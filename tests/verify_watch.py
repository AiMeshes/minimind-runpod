#!/usr/bin/env python3
"""验证 watch.py 的恢复逻辑 —— 不需要 RunPod 账号，不花钱。

watch.py 是让训练活下来的那一环：RunPod 被抢占的 Pod **不会自己重启**，
全靠这个脚本发现并把它拉起来。它判断错了只有两种后果，都很贵：

  - 该恢复时不恢复 → 训练停摆，你不在电脑前就一直停着
  - 不该动时乱动   → 正常训练中重复创建 Pod，白烧两份钱

所以这里用假的 runpod 模块覆盖每种 Pod 状态，断言它做了正确的事。

用法:
    .venv/bin/python tests/verify_watch.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCH_PY = REPO_ROOT / "watch.py"

CONFIG = """\
NETWORK_VOLUME_ID=nv-test123
IMAGE_NAME=test/minimind-runpod:latest
POD_NAME=minimind-64m-pretrain
GPU_TYPE=NVIDIA GeForce RTX 4090
GPU_COUNT=8
"""


class Calls:
    """记录假 runpod 模块收到过哪些调用。"""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.created: list[dict] = []
        self.get_pods_count = 0


def make_fake_runpod(pods, calls: Calls, *, start_raises=False,
                     get_pods_raises=False):
    fake = types.ModuleType("runpod")

    def get_pods():
        calls.get_pods_count += 1
        if get_pods_raises:
            raise RuntimeError("模拟网络故障")
        return pods

    def start_pod(pod_id):
        if start_raises:
            raise RuntimeError("模拟 start 失败（机器已被别人占用）")
        calls.started.append(pod_id)
        return {"id": pod_id}

    def create_pod(**kwargs):
        calls.created.append(kwargs)
        return {"id": "pod-new-0001"}

    fake.get_pods = get_pods
    fake.start_pod = start_pod
    fake.create_pod = create_pod
    return fake


def run_watch(pods, *, start_raises=False, get_pods_raises=False,
              force=False) -> tuple[int, Calls, str]:
    """跑一次 watch.py，返回 (退出码, 调用记录, 输出)。"""
    calls = Calls()
    workdir = Path(tempfile.mkdtemp(prefix="mm-watch-"))
    cfg = workdir / "test.env"
    cfg.write_text(CONFIG)

    with_readme = sys.modules.get("runpod")
    sys.modules["runpod"] = make_fake_runpod(
        pods, calls, start_raises=start_raises, get_pods_raises=get_pods_raises)

    old_state = os.environ.get("WATCH_STATE_FILE")
    old_key = os.environ.get("RUNPOD_API_KEY")
    os.environ["WATCH_STATE_FILE"] = str(workdir / "state")
    os.environ["RUNPOD_API_KEY"] = "fake-key"

    old_argv = sys.argv
    sys.argv = ["watch.py", "--config", str(cfg)] + (["--force"] if force else [])

    import io
    import contextlib
    buf = io.StringIO()

    spec = importlib.util.spec_from_file_location("watch_under_test", WATCH_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)          # main() 在 __main__ 下才跑
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = module.main()
    finally:
        sys.argv = old_argv
        if with_readme is not None:
            sys.modules["runpod"] = with_readme
        else:
            sys.modules.pop("runpod", None)
        if old_state is None:
            os.environ.pop("WATCH_STATE_FILE", None)
        else:
            os.environ["WATCH_STATE_FILE"] = old_state
        if old_key is None:
            os.environ.pop("RUNPOD_API_KEY", None)
        else:
            os.environ["RUNPOD_API_KEY"] = old_key

    return code, calls, buf.getvalue()


def fail(msg: str) -> None:
    print(f"\n  ✗ 断言失败: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


POD = {"id": "pod-abc", "name": "minimind-64m-pretrain"}


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 watch.py")
    ap.parse_args()

    # --- 1. 训练进行中 → 什么都不做 ---
    print("\n── 场景: Pod 正在运行 ──")
    code, calls, out = run_watch([{**POD, "desiredStatus": "RUNNING"}])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if calls.started or calls.created:
        fail(f"正常训练中不应有任何写操作，却调用了 start={calls.started} "
             f"create={len(calls.created)}")
    ok("Pod RUNNING → 不做任何写操作（不会重复创建、白烧钱）")

    # --- 2. Pod 已停止（被抢占）→ 优先 start（比重建快）---
    print("\n── 场景: Pod 被抢占后处于 STOPPED ──")
    code, calls, out = run_watch([{**POD, "desiredStatus": "STOPPED"}])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if calls.started != ["pod-abc"]:
        fail(f"应调用 start_pod(['pod-abc'])，实际 {calls.started}")
    if calls.created:
        fail("已停止的 Pod 可以 start，不应直接新建")
    ok("STOPPED → 调用 start_pod（保留容器，比重建快）")

    # --- 3. Pod 已消失 → 新建 ---
    print("\n── 场景: Pod 已被删除 —— 必须新建 ──")
    code, calls, out = run_watch([])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if len(calls.created) != 1:
        fail(f"应新建 1 个 Pod，实际 {len(calls.created)}")
    req = calls.created[0]
    if req.get("network_volume_id") != "nv-test123":
        fail(f"新建时未挂载 network volume: {req.get('network_volume_id')}")
    if req.get("interruptible") is not True:
        fail("新建的 Pod 未启用 spot（interruptible=True）")
    ok("无 Pod → 新建，且正确挂载 volume + 启用 spot")

    # --- 4. start 失败 → 退回新建 ---
    print("\n── 场景: start 失败（原机器已被占用）──")
    code, calls, out = run_watch([{**POD, "desiredStatus": "STOPPED"}],
                                 start_raises=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if calls.started:
        fail("fake 应在 start 时抛错")
    if len(calls.created) != 1:
        fail(f"start 失败后应退回新建，实际新建 {len(calls.created)} 次")
    ok("start 失败 → 自动退回新建（不会卡死在无法启动的 Pod 上）")

    # --- 5. 查询失败 → 退出 0，不误报 ---
    print("\n── 场景: API 查询失败（网络抖动）──")
    code, calls, out = run_watch([], get_pods_raises=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0 —— 返回非 0 会让 cron 误报故障并可能"
             "触发重复告警")
    if calls.created:
        fail("查询失败时不应盲目新建 Pod")
    ok("查询失败 → 退出 0 且不新建（下轮重试，避免网络抖动导致重复创建）")

    # --- 6. --force 覆盖运行中的 Pod ---
    print("\n── 场景: --force（强制重来）──")
    code, calls, out = run_watch([{**POD, "desiredStatus": "RUNNING"}], force=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if len(calls.created) != 1:
        fail(f"--force 时应无视运行中的 Pod 直接新建，实际新建 {len(calls.created)}")
    ok("--force → 无视运行中的 Pod 新建（用于换配置重跑）")

    print("\n" + "=" * 62)
    print("watch.py 恢复逻辑全部符合预期")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
