#!/usr/bin/env python3
"""监控训练 Pod：被抢占后自动重新拉起。

用法（本地 cron 或 launchd 定期调用）:
    export RUNPOD_API_KEY=rpa_...
    python watch.py --config configs/64m-pretrain.env

设计说明:
    RunPod 被抢占的 Pod **不会自己重启** —— 这是本脚本存在的唯一理由。
    发现 Pod 已停/已删时：能 start 就 start（保留容器，更快），
    否则重新 create（entrypoint 会从 network volume 上的 checkpoint 续训）。

    建议每 15–30 分钟跑一次。本地 Mac 睡着时不会触发，但 Pod 照常在跑，
    只是抢占后恢复会被推迟 —— 对跑几天的任务通常可接受。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launch import build_pod_request, load_env_file  # noqa: E402

# 可用环境变量覆盖，便于测试时避免污染用户的 home 目录
STATE_FILE = Path(os.environ.get(
    "WATCH_STATE_FILE",
    Path.home() / ".cache" / "minimind-runpod" / "last_pod_id",
))


def read_last_pod_id():
    try:
        return STATE_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None


def write_last_pod_id(pod_id: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(pod_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="监控并恢复 minimind 训练 Pod")
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="忽略现有 Pod，强制新建")
    args = ap.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        print("错误: 未设置 RUNPOD_API_KEY", file=sys.stderr)
        return 1

    import runpod

    cfg = load_env_file(args.config)
    pod_name = cfg.get("POD_NAME", "minimind-train")

    # --- 找出属于本次训练的所有 Pod ---
    try:
        pods = runpod.get_pods()
    except Exception as exc:
        print(f"查询 Pod 列表失败（网络问题？将在下轮重试）: {exc}", file=sys.stderr)
        return 0  # 不返回 1，避免 cron 误报

    mine = [p for p in pods if p.get("name") == pod_name] if pods else []
    running = [p for p in mine if p.get("desiredStatus") == "RUNNING"]

    if running and not args.force:
        ids = ", ".join(p["id"] for p in running)
        print(f"训练进行中: {ids}")
        write_last_pod_id(running[0]["id"])
        return 0

    # --- 有已停止的 → 优先 start（比重建快，且保留容器盘）---
    if mine and not args.force:
        stopped = [p for p in mine if p.get("desiredStatus") in ("STOPPED", "EXITED")]
        if stopped:
            pod_id = stopped[0]["id"]
            print(f"发现已停止的 Pod {pod_id}，尝试重启...")
            try:
                runpod.start_pod(pod_id)
                print(f"已重启 {pod_id}（entrypoint 将从最新 checkpoint 续训）")
                write_last_pod_id(pod_id)
                return 0
            except Exception as exc:
                print(f"重启失败，将改为新建: {exc}", file=sys.stderr)

    # --- 都没有 → 新建 ---
    print("未发现运行中的 Pod，正在新建...")
    try:
        pod = runpod.create_pod(**build_pod_request(cfg))
    except Exception as exc:
        print(f"创建失败: {exc}", file=sys.stderr)
        return 1

    pod_id = pod.get("id") if isinstance(pod, dict) else pod
    print(f"已新建 Pod: {pod_id}")
    write_last_pod_id(pod_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
