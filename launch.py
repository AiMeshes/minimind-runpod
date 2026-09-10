#!/usr/bin/env python3
"""拉起一个跑 minimind 的 RunPod Pod。

用法:
    export RUNPOD_API_KEY=rpa_...
    python launch.py --config configs/64m-pretrain.env

前置条件:
    - 已建好 network volume（checkpoint 靠它持久化，这是抗抢占的前提）
    - 镜像已推到可公开拉取的 registry

注意: RunPod API 迭代较快。首次运行前先确认参数名：
    python -c "import runpod; print(runpod.create_pod.__doc__)"
"""
import argparse
import os
import sys
from pathlib import Path


def load_env_file(path: str) -> dict:
    """读取 KEY=VALUE 形式的配置文件，忽略注释和空行。

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
        # 剥离行内注释：' #' 之后的都算注释
        for i, ch in enumerate(v):
            if ch == "#" and i > 0 and v[i - 1] in " \t":
                v = v[:i]
                break
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def build_pod_request(cfg: dict) -> dict:
    gpu_count = int(cfg.get("GPU_COUNT", 1))

    # 传进容器的环境变量（entrypoint.sh 读取）
    container_env = [
        {"key": "DATA_URL", "value": cfg.get("DATA_URL", "")},
        {"key": "MINIMIND_REPO", "value": cfg.get(
            "MINIMIND_REPO", "https://github.com/AiMeshes/minimind.git")},
        {"key": "NUM_GPUS", "value": str(gpu_count)},
        {"key": "RESTART_DELAY", "value": cfg.get("RESTART_DELAY", "15")},
    ]
    # 训练超参走环境变量，不走 docker_args ——
    # docker_args 在 RunPod 里是「容器启动命令」，不是「追加参数」，
    # 用它传超参会把 entrypoint 整个替换掉。entrypoint.sh 会读取本变量。
    if cfg.get("TRAIN_ARGS"):
        container_env.append({"key": "TRAIN_ARGS", "value": cfg["TRAIN_ARGS"]})

    if cfg.get("HF_TOKEN"):
        container_env.append({"key": "HF_TOKEN", "value": cfg["HF_TOKEN"]})
    if cfg.get("WANDB_API_KEY"):
        container_env.append({"key": "WANDB_API_KEY", "value": cfg["WANDB_API_KEY"]})
    if cfg.get("START_SSHD"):
        container_env.append({"key": "START_SSHD", "value": cfg["START_SSHD"]})

    req = {
        "name": cfg.get("POD_NAME", "minimind-train"),
        "image_name": cfg["IMAGE_NAME"],
        "gpu_type_id": cfg.get("GPU_TYPE", "NVIDIA GeForce RTX 4090"),
        "gpu_count": gpu_count,
        "cloud_type": cfg.get("CLOUD_TYPE", "COMMUNITY"),
        # Spot：便宜约 50%，但随时可能被抢占 —— 抗抢占能力正是本外壳的存在理由
        "interruptible": cfg.get("INTERRUPTIBLE", "1") == "1",
        "container_disk_in_gb": int(cfg.get("CONTAINER_DISK_GB", 40)),
        # 不用 volume disk（停机后涨价到 $0.20/GB/月，见 cost-guide §5）
        "volume_in_gb": 0,
        "network_volume_id": cfg["NETWORK_VOLUME_ID"],
        "ports": cfg.get("PORTS", "8888/http"),
        "env": container_env,
    }

    return req


def main() -> int:
    ap = argparse.ArgumentParser(description="拉起 minimind 训练 Pod")
    ap.add_argument("--config", required=True, help="配置文件路径")
    ap.add_argument("--dry-run", action="store_true", help="只打印请求，不实际创建")
    args = ap.parse_args()

    # RUNPOD_API_KEY 必须在 import runpod 之前进入环境（构造函数在实例化时读凭据）
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("错误: 未设置 RUNPOD_API_KEY", file=sys.stderr)
        return 1

    cfg = load_env_file(args.config)
    for required in ("IMAGE_NAME", "NETWORK_VOLUME_ID"):
        if not cfg.get(required):
            print(f"错误: 配置缺少 {required}", file=sys.stderr)
            return 1

    req = build_pod_request(cfg)

    if args.dry_run:
        import json
        print(json.dumps(req, indent=2, ensure_ascii=False))
        return 0

    import runpod  # noqa: E402

    pod = runpod.create_pod(**req)
    pod_id = pod.get("id") if isinstance(pod, dict) else pod
    print(f"Pod 已创建: {pod_id}")
    print(f"监控台: https://www.console.runpod.io/pods")
    print()
    print("注意: 建 Pod 前请确认 network volume 与所选 GPU 在同一个数据中心，")
    print("      否则 RunPod 会报错找不到可用机器（volume 是地域锁定的）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
