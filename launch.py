#!/usr/bin/env python3
"""拉起一个跑 minimind 的 RunPod Pod（REST API）。

用法:
    cp .env.example .env      # 填入 RUNPOD_API_KEY、NETWORK_VOLUME_ID、IMAGE_NAME
    python launch.py --config configs/64m-pretrain.env --dry-run
    python launch.py --config configs/64m-pretrain.env
    python launch.py --list   # 查看现有 Pod

为什么用 REST 而不是 runpod Python SDK：
    SDK 1.12.0（当前最新）的 create_pod **没有 spot 相关参数** ——
    整个包里搜不到 interruptible / max_bid_price / SPOT。
    而 REST 的 POST /v1/pods 支持 interruptible: true。
    没有它就只能按需付费，成本翻倍。

字段名以 OpenAPI spec 为准（https://rest.runpod.io/v1/openapi.json）：
    camelCase；env 是**对象**不是 {key,value} 列表；ports 是**数组**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("RUNPOD_REST_BASE", "https://rest.runpod.io/v1")
REPO_ROOT = Path(__file__).resolve().parent


def load_env_file(path: str) -> dict:
    """读取 KEY=VALUE 配置文件，忽略注释和空行。

    只剥离「空白 + #」形式的行内注释，这样值里合法出现的 # 不会被误删
    （例如 URL 的 fragment、或不含空格的 token）。
    """
    env = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"配置行缺少 '=': {raw!r}")
        k, v = line.split("=", 1)
        for i, ch in enumerate(v):
            if ch == "#" and i > 0 and v[i - 1] in " \t":
                v = v[:i]
                break
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_dotenv_if_present() -> None:
    """把仓库根目录的 .env 注入环境变量（已存在的同名变量不覆盖）。"""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for key, value in load_env_file(str(env_path)).items():
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# REST 客户端
# ---------------------------------------------------------------------------
def api_request(method: str, path: str, body: dict | None = None) -> object:
    """调用 RunPod REST API。

    User-Agent 是必需的 —— 缺失会被 Cloudflare 以 403 拒绝。
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit(
            "错误: 未设置 RUNPOD_API_KEY。"
            "写入 .env（见 .env.example）或 export 到环境变量。")

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "minimind-runpod/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")[:500]
        raise SystemExit(f"API {method} {path} 失败: HTTP {exc.code}\n{detail}") from exc


# ---------------------------------------------------------------------------
# 请求构造
# ---------------------------------------------------------------------------
def split_list(value: str) -> list[str]:
    """把 "a, b, c" 拆成列表；空字符串返回空列表。"""
    return [x.strip() for x in value.split(",") if x.strip()]


def build_pod_request(cfg: dict) -> dict:
    """按 OpenAPI 的 PodCreateInput 构造请求体。

    字段名与类型严格对齐 spec —— 写错只会在真实调用时才发现，
    所以这里每个字段都对照过 https://rest.runpod.io/v1/openapi.json
    """
    gpu_count = int(cfg.get("GPU_COUNT", 1))

    # env 在 REST 里是**对象**，不是 SDK 那样的 [{key,value}] 列表
    env: dict[str, str] = {
        "MINIMIND_REPO": cfg.get(
            "MINIMIND_REPO", "https://github.com/AiMeshes/minimind.git"),
        "NUM_GPUS": str(gpu_count),
        "RESTART_DELAY": cfg.get("RESTART_DELAY", "15"),
    }
    # 逐个转发给 entrypoint.sh 的可选配置。漏掉一项 = 配置静默失效，
    # 所以 tests/verify_watch.py 里对此有断言。
    for opt in ("DATA_URL", "TRAIN_ARGS", "START_SSHD", "HF_TOKEN", "WANDB_API_KEY",
                "INSTALL_DEPS", "PIP_PACKAGES", "PIP_PROBE"):
        if cfg.get(opt):
            env[opt] = cfg[opt]

    req: dict = {
        "name": cfg.get("POD_NAME", "minimind-train"),
        "imageName": cfg["IMAGE_NAME"],
        "gpuTypeIds": split_list(cfg.get("GPU_TYPE", "NVIDIA GeForce RTX 4090")),
        "gpuCount": gpu_count,
        "cloudType": cfg.get("CLOUD_TYPE", "COMMUNITY"),
        # Spot：便宜约 50%，但随时可能被抢占 —— 抗抢占能力正是本外壳的存在理由
        "interruptible": cfg.get("INTERRUPTIBLE", "1") == "1",
        "containerDiskInGb": int(cfg.get("CONTAINER_DISK_GB", 50)),
        # 显式用 0：volume disk 停机后涨到 $0.20/GB/月，是全平台最贵的存储
        # （见成本指南 §5）。checkpoint 靠 network volume 持久化，不需要它。
        "volumeInGb": 0,
        # ports 是**数组**；默认值含 22/tcp，便于 SSH 调试
        "ports": split_list(cfg.get("PORTS", "8888/http,22/tcp")),
        # Community Cloud 上需要显式请求公网 IP 才能 SSH 进去看日志
        "supportPublicIp": cfg.get("PUBLIC_IP", "1") == "1",
        "env": env,
    }

    if cfg.get("NETWORK_VOLUME_ID"):
        req["networkVolumeId"] = cfg["NETWORK_VOLUME_ID"]
    if cfg.get("DATA_CENTER_IDS"):
        # network volume 是地域锁定的：Pod 必须在 volume 所在机房
        req["dataCenterIds"] = split_list(cfg["DATA_CENTER_IDS"])

    # 用官方镜像时，entrypoint.sh 不在镜像里，需要运行时拉取。
    # 单独用 BOOTSTRAP_URL 而不是通用的 dockerStartCmd，是因为后者按逗号拆分，
    # 会把 URL 或脚本内容切碎。
    if cfg.get("BOOTSTRAP_URL"):
        url = cfg["BOOTSTRAP_URL"]
        req["dockerStartCmd"] = [
            "bash", "-c",
            f"set -e; curl -fsSL {url} -o /opt/mm-entrypoint.sh; "
            f"exec bash /opt/mm-entrypoint.sh",
        ]
    elif cfg.get("DOCKER_START_CMD"):
        req["dockerStartCmd"] = split_list(cfg["DOCKER_START_CMD"])

    if cfg.get("DOCKER_ENTRYPOINT"):
        req["dockerEntrypoint"] = split_list(cfg["DOCKER_ENTRYPOINT"])

    return req


def cmd_list() -> int:
    pods = api_request("GET", "/pods") or []
    if not pods:
        print("无任何 Pod —— 没有在偷偷烧钱")
        return 0
    total = 0.0
    print(f"{'ID':<22} {'名称':<24} {'状态':<12} {'$/hr':>7}  中断")
    print("-" * 76)
    for p in pods:
        cost = p.get("costPerHr") or 0
        if p.get("desiredStatus") == "RUNNING":
            total += cost
        print(f"{p.get('id',''):<22} {p.get('name',''):<24} "
              f"{p.get('desiredStatus',''):<12} {cost:>7.3f}  "
              f"{'是' if p.get('interruptible') else '否'}")
    print("-" * 76)
    print(f"运行中合计: ${total:.3f}/hr  →  ${total * 24:.2f}/天  ${total * 730:.2f}/月")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="拉起 minimind 训练 Pod")
    ap.add_argument("--config", help="配置文件路径")
    ap.add_argument("--dry-run", action="store_true", help="只打印请求，不实际创建")
    ap.add_argument("--list", action="store_true", help="列出所有 Pod 及花费")
    args = ap.parse_args()

    # 必须在任何 API 调用之前
    load_dotenv_if_present()

    if args.list:
        return cmd_list()

    if not args.config:
        ap.error("需要 --config（或用 --list）")

    cfg = load_env_file(args.config)
    for required in ("IMAGE_NAME",):
        if not cfg.get(required):
            print(f"错误: 配置缺少 {required}", file=sys.stderr)
            return 1

    req = build_pod_request(cfg)

    if args.dry_run:
        print(json.dumps(req, indent=2, ensure_ascii=False))
        return 0

    pod = api_request("POST", "/pods", req)
    pod_id = pod.get("id") if isinstance(pod, dict) else pod
    print(f"Pod 已创建: {pod_id}")
    print(f"  计费: ${pod.get('costPerHr', '?')}/hr"
          f"  中断式: {'是（Spot）' if req['interruptible'] else '否'}")
    print(f"  面板: https://www.console.runpod.io/pods")
    if cfg.get("NETWORK_VOLUME_ID"):
        print()
        print(f"  提示: network volume {cfg['NETWORK_VOLUME_ID']} 是地域锁定的，")
        print(f"        若 Pod 起不来，多半是该机房没有 GPU_TYPE 的现货。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
