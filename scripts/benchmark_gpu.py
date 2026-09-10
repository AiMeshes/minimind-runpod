#!/usr/bin/env python3
"""在指定 GPU 上用真实 minimind 负载实测训练速度。

为什么需要实测：规格表上的 TFLOPS 对 64M 这种小模型没有参考价值 ——
模型太小，瓶颈会跑到数据加载和 kernel 启动开销上。
只有同一套代码、同一份数据、同一个 batch 下测出来的 steps/sec 才能比较。

流程：
  1. 用给定 GPU 类型创建 Pod（复用 64m-full-run 的启动脚本）
  2. 等训练真正跑起来（日志出现 Epoch 行）
  3. 采样两次 step，间隔 --window 秒，算 steps/sec
  4. 终止 Pod（无论成功失败都终止，避免漏计费）
  5. 输出一行 Markdown 表格数据

用法:
    python scripts/benchmark_gpu.py --gpu "NVIDIA H100 NVL" --count 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from launch import (  # noqa: E402
    api_request,
    build_pod_request,
    load_dotenv_if_present,
    load_env_file,
)

SSH_KEY = Path.home() / ".ssh" / "id_rsa"
EPOCH_RE = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)")


def log(msg: str) -> None:
    print(msg, flush=True)


class SSHError(RuntimeError):
    """SSH 执行失败（连接、认证、超时）。"""


def ssh(ip: str, port: str, cmd: str, timeout: int = 20) -> str:
    """在 Pod 上执行命令。

    ⚠️ 失败时**抛异常**而不是返回空串。
    之前的实现用宽泛的 except 返回 ""，把「SSH 认证失败」伪装成
    「命令输出为空」—— 表现为"容器里没有任何进程"，白排查了很久。
    这类静默降级比直接报错危险得多。
    """
    try:
        r = subprocess.run(
            ["ssh", "-i", str(SSH_KEY), "-p", port,
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", f"ConnectTimeout={timeout}", "-o", "LogLevel=ERROR",
             "-o", "BatchMode=yes",          # 禁止交互式密码提示，失败即失败
             f"root@{ip}", cmd],
            capture_output=True, text=True, timeout=timeout + 20,
        )
    except subprocess.TimeoutExpired as exc:
        raise SSHError(f"SSH 超时（{timeout}s）: {cmd[:60]}") from exc
    except Exception as exc:
        raise SSHError(f"SSH 无法执行: {exc}") from exc

    if r.returncode != 0:
        err = (r.stderr or "").strip()[:160]
        raise SSHError(f"SSH 失败 (rc={r.returncode}): {err or '无 stderr'}")
    return r.stdout.strip()


def ssh_ok(ip: str, port: str, cmd: str = "echo ok", timeout: int = 20) -> bool:
    """不抛异常的探活版本。"""
    try:
        return ssh(ip, port, cmd, timeout) == "ok"
    except SSHError:
        return False


def get_ssh_info(pod_id: str, api_key: str) -> tuple[str, str] | None:
    """取 Pod 的 SSH 连接信息。runpodctl 比 REST 更早暴露端口映射。

    ⚠️ 必须逐个候选尝试，不能只挑第一个存在的二进制：
    brew 装的 runpodctl 1.14.3（2024-05）的 ssh 子命令**没有 info**，
    会打印帮助文本并以 0 退出 —— 只看退出码会误判为成功。
    所以判据是「stdout 能解析成含 ip/port 的 JSON」。
    """
    candidates = [
        "/tmp/runpodctl",                       # 本项目下载的新版
        str(Path.home() / ".local/bin/runpodctl"),
        shutil.which("runpodctl") or "",        # PATH 里的可能是老版本
    ]
    for exe in candidates:
        if not exe:
            continue
        if not (shutil.which(exe) or Path(exe).is_file()):
            continue
        try:
            r = subprocess.run([exe, "ssh", "info", pod_id],
                               capture_output=True, text=True, timeout=30,
                               env={**os.environ, "RUNPOD_API_KEY": api_key})
            d = json.loads(r.stdout)
            if d.get("ip") and d.get("port"):
                return d["ip"], str(d["port"])
        except Exception:
            continue
    return None


def parse_progress(log_text: str) -> tuple[int, int, int] | None:
    """从日志取最后一次进度。返回 (global_step, step_in_epoch, iters)。"""
    matches = list(EPOCH_RE.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    epoch_idx, _, step, iters = (int(x) for x in m.groups())
    return epoch_idx * iters + step, step, iters


def wait_for_training(ip: str, port: str, deadline: float) -> bool:
    """等训练真正开始（日志出现 Epoch 行）。"""
    while time.time() < deadline:
        try:
            out = ssh(ip, port,
                      "grep -cE '^Epoch:' /workspace/train.log 2>/dev/null || echo 0")
            if int(out or 0) >= 2:      # 至少两行进度，说明已经稳定在训
                return True
        except (SSHError, ValueError):
            pass
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="实测 GPU 训练速度")
    ap.add_argument("--gpu", required=True, help="GPU 类型 ID，如 'NVIDIA H100 NVL'")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--config", default="configs/64m-full-run.env")
    ap.add_argument("--window", type=int, default=60, help="采样窗口秒数")
    ap.add_argument("--wait-minutes", type=int, default=25, help="等训练启动的上限")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印请求体并做一致性校验，不创建任何 Pod（免费）")
    args = ap.parse_args()

    load_dotenv_if_present()
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    cfg = load_env_file(str(REPO / args.config))

    req = build_pod_request(cfg)
    req.update({
        "name": f"bench-{args.gpu.split()[-1].lower()}",
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.count,
        "interruptible": False,          # 基准测试不能被抢占打断
    })
    # ⚠️ 必须同步改 NUM_GPUS：build_pod_request 是按 cfg 的 GPU_COUNT 生成的，
    # 若只改 gpuCount 而不改它，容器会用 --nproc_per_node=2 去跑单卡 Pod，
    # rank 1 报 "CUDA error: invalid device ordinal" 后整个训练崩溃。
    req["env"]["NUM_GPUS"] = str(args.count)
    # 基准测试不挂卷、不限机房
    req.pop("networkVolumeId", None)
    req.pop("dataCenterIds", None)

    # --- 创建前的一致性校验（免费，且能挡住上一轮那个 $1.27 的错误）---
    problems = []
    if req["env"].get("NUM_GPUS") != str(req["gpuCount"]):
        problems.append(
            f"NUM_GPUS({req['env'].get('NUM_GPUS')}) != gpuCount({req['gpuCount']}) "
            "→ 容器会以错误的世界大小启动 torchrun，rank>0 报 "
            "'CUDA error: invalid device ordinal' 后崩溃")
    if req.get("interruptible"):
        problems.append("基准测试不应使用 Spot —— 被抢占会让测量中断")
    if not req.get("dockerStartCmd"):
        problems.append("dockerStartCmd 为空 → 容器不会执行 entrypoint.sh")
    if problems:
        log("✗ 请求体自检未通过：")
        for p in problems:
            log(f"    · {p}")
        return 2
    log("  ✓ 请求体自检通过（NUM_GPUS 一致 / 按需 / 启动命令就位）")

    if args.dry_run:
        log("\n=== 请求体（--dry-run，未创建任何资源）===")
        log(json.dumps(req, indent=2, ensure_ascii=False))
        return 0

    pod = api_request("POST", "/pods", req)
    pod_id = pod["id"]
    cost = pod.get("costPerHr") or 0
    log(f"Pod {pod_id} 已创建  {args.gpu} ×{args.count}  ${cost}/hr")

    try:
        # --- 等 SSH ---
        deadline = time.time() + args.wait_minutes * 60
        info = None
        while time.time() < deadline:
            info = get_ssh_info(pod_id, api_key)
            if info and ssh_ok(info[0], info[1]):
                break
            info = None
            time.sleep(15)
        if not info:
            log("✗ 超时：SSH 未就绪（镜像拉取过慢或容器启动失败）")
            return 1
        ip, port = info
        log(f"  SSH 就绪: {ip}:{port}  （已耗时 {int(args.wait_minutes*60 - (deadline-time.time()))}s）")

        # --- 等训练 ---
        if not wait_for_training(ip, port, deadline):
            log("✗ 超时：训练未开始")
            try:
                cc = ssh(ip, port, "tail -5 /workspace/train.log 2>/dev/null")
                if cc:
                    log("  日志尾部：\n" + "\n".join("    " + l for l in cc.splitlines()))
            except SSHError as exc:
                log(f"  （读日志也失败: {exc}）")
            return 1
        log("  训练已启动，开始采样")

        # --- 采样 ---
        t0 = time.time()
        a = parse_progress(ssh(ip, port, "cat /workspace/train.log"))
        while time.time() - t0 < args.window:
            time.sleep(5)
        b = parse_progress(ssh(ip, port, "cat /workspace/train.log"))
        elapsed = time.time() - t0

        if not a or not b:
            log("✗ 采样失败：解析不到进度")
            return 1

        steps = b[0] - a[0]
        if steps <= 0:
            log(f"✗ 采样窗口内 step 无推进（{a[0]} → {b[0]}）")
            return 1

        sps = steps / elapsed
        iters = b[2]
        # 2 epochs 的总步数（DDP 下 iters 已是每卡分片后的值）
        total = iters * 2
        full_hours = total / sps / 3600

        log("")
        log(f"  实测: {steps} steps / {elapsed:.0f}s = {sps:.2f} steps/s")
        log(f"  完整跑完 2 epochs（{total} steps）: {full_hours:.2f} 小时, ${full_hours*cost:.2f}")
        log("")
        log("MARKDOWN:")
        log(f"| {args.gpu} ×{args.count} | {cost:.2f} | {sps:.2f} | "
            f"{full_hours:.2f} h | ${full_hours*cost:.2f} |")
        return 0
    finally:
        try:
            api_request("DELETE", f"/pods/{pod_id}")
            log(f"  Pod {pod_id} 已终止")
        except SystemExit as e:
            log(f"  ⚠️ 终止失败，请手动检查: {e}")


if __name__ == "__main__":
    sys.exit(main())
