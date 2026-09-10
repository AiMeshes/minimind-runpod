#!/usr/bin/env python3
"""监控训练 Pod：被抢占后自动重新拉起。

用法（本地 cron 或 launchd 定期调用）:
    python watch.py --config configs/my-run.env
    python watch.py --config configs/my-run.env --status   # 只看状态，不做变更

设计说明:
    RunPod 被抢占的 Pod **不会自己重启** —— 这是本脚本存在的唯一理由。
    发现 Pod 已停就把它 start 起来；已经被删就只能重新 create
    （entrypoint 会从 network volume 上的 checkpoint 续训）。

    Pod 状态取自 REST 的 desiredStatus，合法值为：
        RUNNING | EXITED | TERMINATED
    注意**没有 STOPPED**。被抢占后是 EXITED，可以 start 恢复；
    TERMINATED 是彻底删除，只能重建。

    建议每 15–30 分钟跑一次。本地 Mac 睡着时不会触发，但 Pod 照常在跑，
    只是抢占后恢复会被推迟 —— 对跑几天的任务通常可接受。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launch import (  # noqa: E402
    api_request,
    build_pod_request,
    load_dotenv_if_present,
    load_env_file,
)

# 可用环境变量覆盖，便于测试时避免污染用户的 home 目录
STATE_FILE = Path(os.environ.get(
    "WATCH_STATE_FILE",
    Path.home() / ".cache" / "minimind-runpod" / "last_pod_id",
))


def read_last_pod_id() -> str | None:
    try:
        return STATE_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None


def write_last_pod_id(pod_id: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(pod_id)


def describe(pod: dict) -> str:
    cost = pod.get("costPerHr")
    cost_s = f"${cost:.3f}/hr" if isinstance(cost, (int, float)) else "?/hr"
    return (f"{pod.get('id','')}  状态={pod.get('desiredStatus','')}  {cost_s}"
            f"  {'Spot' if pod.get('interruptible') else '按需'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="监控并恢复 minimind 训练 Pod")
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="忽略现有 Pod，强制新建")
    ap.add_argument("--status", action="store_true",
                    help="只报告状态，不做任何变更")
    args = ap.parse_args()

    load_dotenv_if_present()
    cfg = load_env_file(args.config)
    pod_name = cfg.get("POD_NAME", "minimind-train")

    # --- 找出属于本次训练的 Pod ---
    try:
        pods = api_request("GET", "/pods")
    except SystemExit as exc:
        # 网络抖动不该让 cron 误报，也不该触发盲目重建
        print(f"查询 Pod 列表失败（将在下轮重试）: {exc}", file=sys.stderr)
        return 0

    mine = [p for p in (pods or [])
            if isinstance(p, dict) and p.get("name") == pod_name]

    running = [p for p in mine if p.get("desiredStatus") == "RUNNING"]
    exited = [p for p in mine if p.get("desiredStatus") == "EXITED"]
    # TERMINATED 是彻底删除，无法恢复，只能重建

    if args.status:
        if not mine:
            print(f"未找到名为 {pod_name!r} 的 Pod")
        for p in mine:
            print(" ", describe(p))
        return 0

    if running and not args.force:
        p = running[0]
        print(f"训练进行中: {describe(p)}")
        write_last_pod_id(p["id"])
        return 0

    # --- 有 EXITED 的 → 优先 start（比重建快，且保留容器盘）---
    if exited and not args.force:
        pod_id = exited[0]["id"]
        print(f"发现已停止的 Pod {pod_id}，尝试重启...")
        try:
            api_request("POST", f"/pods/{pod_id}/start")
            print(f"已重启 {pod_id}（entrypoint 将从最新 checkpoint 续训）")
            write_last_pod_id(pod_id)
            return 0
        except SystemExit as exc:
            print(f"重启失败，将改为新建: {exc}", file=sys.stderr)

    # --- 都没有 → 新建 ---
    print("未发现可恢复的 Pod，正在新建...")
    try:
        pod = api_request("POST", "/pods", build_pod_request(cfg))
    except SystemExit as exc:
        print(f"创建失败: {exc}", file=sys.stderr)
        return 1

    pod_id = pod.get("id") if isinstance(pod, dict) else pod
    print(f"已新建 Pod: {pod_id}")
    write_last_pod_id(pod_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
