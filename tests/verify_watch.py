#!/usr/bin/env python3
"""验证 watch.py 的恢复逻辑与 launch.py 的请求体格式 —— 不需要 RunPod 账号，不花钱。

两项验证：

  A. 恢复逻辑。watch.py 是让训练活下来的那一环（被抢占的 Pod 不会自己重启）。
     它判断错了只有两种后果，都很贵：该恢复时不恢复（训练停摆）、
     不该动时乱动（重复创建，白烧两份钱）。

  B. 请求体字段名。REST API 与 SDK 的字段格式完全不同（camelCase、
     env 是对象不是列表、ports 是数组），写错只会在真实调用时才发现。
     这里对照 OpenAPI spec 抽取的字段清单做一致性检查。

用法:
    .venv/bin/python tests/verify_watch.py
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import contextlib
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
CONTAINER_DISK_GB=40
INTERRUPTIBLE=1
CLOUD_TYPE=COMMUNITY
PORTS=8888/http,22/tcp
DATA_URL=https://example.com/data.jsonl
INSTALL_DEPS=1
PIP_PACKAGES=transformers==4.57.6 datasets==3.6.0
TRAIN_ARGS=--epochs 1 --save_interval 50
"""

# 必须被转发进容器 env 的配置项。漏掉任何一项 = 配置静默失效
# （entrypoint.sh 读不到，会退回默认值而没有任何报错）
MUST_FORWARD = ["DATA_URL", "INSTALL_DEPS", "PIP_PACKAGES", "TRAIN_ARGS"]

# 取自 https://rest.runpod.io/v1/openapi.json 的 PodCreateInput.properties
VALID_CREATE_FIELDS = {
    "allowedCudaVersions", "cloudType", "computeType", "containerDiskInGb",
    "containerRegistryAuthId", "countryCodes", "cpuFlavorIds",
    "cpuFlavorPriority", "dataCenterIds", "dataCenterPriority",
    "dockerEntrypoint", "dockerStartCmd", "env", "globalNetworking",
    "gpuCount", "gpuTypeIds", "gpuTypePriority", "imageName", "interruptible",
    "locked", "minDiskBandwidthMBps", "minDownloadMbps", "minRAMPerGPU",
    "minUploadMbps", "minVCPUPerGPU", "name", "networkVolumeId", "ports",
    "supportPublicIp", "templateId", "vcpuCount", "volumeInGb",
    "volumeMountPath",
}

# desiredStatus 的合法值 —— 注意**没有 STOPPED**
VALID_STATUSES = {"RUNNING", "EXITED", "TERMINATED"}

POD = {"id": "pod-abc", "name": "minimind-64m-pretrain",
       "desiredStatus": "RUNNING", "costPerHr": 2.72, "interruptible": True}


class FakeAPI:
    """记录调用并按脚本返回。"""

    def __init__(self, pods, *, get_raises=False, start_raises=False):
        self.pods = pods
        self.get_raises = get_raises
        self.start_raises = start_raises
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method: str, path: str, body: dict | None = None):
        self.calls.append((method, path, body))
        if method == "GET" and path == "/pods":
            if self.get_raises:
                raise SystemExit("模拟网络故障")
            return self.pods
        if method == "POST" and path.endswith("/start"):
            if self.start_raises:
                raise SystemExit("模拟 start 失败（机器已被别人占用）")
            return {"id": path.split("/")[2]}
        if method == "POST" and path == "/pods":
            return {"id": "pod-new-0001", "costPerHr": 2.72}
        raise AssertionError(f"未预期的调用: {method} {path}")

    @property
    def writes(self):
        return [c for c in self.calls if c[0] != "GET"]


def run_watch(pods, *, get_raises=False, start_raises=False,
              force=False, status=False) -> tuple[int, FakeAPI, str]:
    api = FakeAPI(pods, get_raises=get_raises, start_raises=start_raises)
    workdir = Path(tempfile.mkdtemp(prefix="mm-watch-"))
    cfg = workdir / "test.env"
    cfg.write_text(CONFIG)

    spec = importlib.util.spec_from_file_location("watch_under_test", WATCH_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.api_request = api          # 替换掉真实网络调用

    argv = ["watch.py", "--config", str(cfg)]
    if force:
        argv.append("--force")
    if status:
        argv.append("--status")

    old_argv, old_state = sys.argv, os.environ.get("WATCH_STATE_FILE")
    sys.argv = argv
    os.environ["WATCH_STATE_FILE"] = str(workdir / "state")

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = module.main()
    finally:
        sys.argv = old_argv
        if old_state is None:
            os.environ.pop("WATCH_STATE_FILE", None)
        else:
            os.environ["WATCH_STATE_FILE"] = old_state

    return code, api, buf.getvalue()


def fail(msg: str) -> None:
    print(f"\n  ✗ 断言失败: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


def check_body_schema(body: dict) -> list[str]:
    """对照 OpenAPI spec 检查请求体。返回问题列表。"""
    problems = []
    for k in body:
        if k not in VALID_CREATE_FIELDS:
            problems.append(f"未知字段 {k!r}（camelCase 拼写错误？）")
    if not isinstance(body.get("env"), dict):
        problems.append("env 必须是对象（不是 [{key,value}] 列表）")
    if not isinstance(body.get("ports"), list):
        problems.append("ports 必须是数组（不是逗号分隔的字符串）")
    if not isinstance(body.get("gpuTypeIds"), list):
        problems.append("gpuTypeIds 必须是数组")
    if not isinstance(body.get("interruptible"), bool):
        problems.append("interruptible 必须是布尔值")
    return problems


def main() -> int:
    argparse.ArgumentParser(description="验证 watch.py").parse_args()

    # --- 1. 训练进行中 → 什么都不做 ---
    print("\n── 场景: Pod 正在运行 ──")
    code, api, out = run_watch([POD])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if api.writes:
        fail(f"正常训练中不应有写操作，却有 {api.writes}")
    ok("RUNNING → 不做任何写操作（不会重复创建、白烧钱）")

    # --- 2. Pod 被抢占（EXITED）→ 优先 start ---
    print("\n── 场景: Pod 被抢占后处于 EXITED ──")
    code, api, out = run_watch([{**POD, "desiredStatus": "EXITED"}])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if ("POST", "/pods/pod-abc/start", None) not in api.calls:
        fail(f"应调用 start，实际写操作 {api.writes}")
    if any(c[1] == "/pods" for c in api.writes):
        fail("已停止的 Pod 可以 start，不应直接新建")
    ok("EXITED → 调用 start（保留容器，比重建快）")

    # --- 3. Pod 已被删除 → 新建 + 请求体合规 ---
    print("\n── 场景: Pod 已被删除 —— 必须新建 ──")
    code, api, out = run_watch([])
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    creates = [c for c in api.writes if c[1] == "/pods"]
    if len(creates) != 1:
        fail(f"应新建 1 个 Pod，实际 {len(creates)}")
    body = creates[0][2]

    problems = check_body_schema(body)
    if problems:
        fail("请求体不符合 OpenAPI schema:\n     " + "\n     ".join(problems))
    ok("请求体字段名全部合法（对照 OpenAPI spec 的 33 个字段）")

    if body.get("networkVolumeId") != "nv-test123":
        fail(f"未挂载 network volume: {body.get('networkVolumeId')}")
    if body.get("interruptible") is not True:
        fail("未启用 spot（interruptible 应为 true）")
    if body.get("volumeInGb") != 0:
        fail(f"volumeInGb 应为 0（停机后 $0.20/GB/月），实际 {body.get('volumeInGb')}")
    if body.get("gpuTypeIds") != ["NVIDIA GeForce RTX 4090"]:
        fail(f"gpuTypeIds 不对: {body.get('gpuTypeIds')}")
    if body.get("env", {}).get("NUM_GPUS") != "8":
        fail(f"env 中 NUM_GPUS 不对: {body.get('env')}")
    ok("挂载 volume + 启用 spot + volumeInGb=0 + GPU/env 正确")

    # 配置必须真的进到容器里 —— 漏转发只会静默退回默认值，不会报错
    missing = [k for k in MUST_FORWARD if k not in body.get("env", {})]
    if missing:
        fail(f"这些配置项没被转发进容器 env: {missing} —— "
             "entrypoint.sh 会读不到并静默使用默认值")
    ok(f"{len(MUST_FORWARD)} 项关键配置均已转发进容器 env")

    # --- 4. start 失败 → 退回新建 ---
    print("\n── 场景: start 失败（原机器已被占用）──")
    code, api, out = run_watch([{**POD, "desiredStatus": "EXITED"}],
                               start_raises=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if len([c for c in api.writes if c[1] == "/pods"]) != 1:
        fail("start 失败后应退回新建")
    ok("start 失败 → 自动退回新建（不会卡死在无法启动的 Pod 上）")

    # --- 5. 查询失败 → 退出 0，不误报 ---
    print("\n── 场景: API 查询失败（网络抖动）──")
    code, api, out = run_watch([], get_raises=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0 —— 非 0 会让 cron 误报故障并可能触发重复告警")
    if api.writes:
        fail("查询失败时不应盲目新建 Pod")
    ok("查询失败 → 退出 0 且不新建（下轮重试，避免网络抖动导致重复创建）")

    # --- 6. --force ---
    print("\n── 场景: --force（强制重来）──")
    code, api, out = run_watch([POD], force=True)
    if len([c for c in api.writes if c[1] == "/pods"]) != 1:
        fail("--force 时应无视运行中的 Pod 直接新建")
    ok("--force → 无视运行中的 Pod 新建（用于换配置重跑）")

    # --- 7. --status 只读 ---
    print("\n── 场景: --status（只报告）──")
    code, api, out = run_watch([POD], status=True)
    if code != 0:
        fail(f"退出码 {code}，期望 0")
    if api.writes:
        fail(f"--status 不应有任何写操作，却有 {api.writes}")
    ok("--status → 只报告，不做任何变更")

    # --- 8. 状态枚举自查 ---
    print("\n── 检查: 状态枚举是否与 API 一致 ──")
    import re
    src = WATCH_PY.read_text()
    if '"STOPPED"' in src:
        fail("watch.py 里出现了 'STOPPED' —— REST API 的 desiredStatus "
             "只有 RUNNING/EXITED/TERMINATED，不存在 STOPPED")
    ok(f"watch.py 未使用不存在的状态值（API 合法值: {sorted(VALID_STATUSES)}）")

    print("\n" + "=" * 62)
    print("watch.py 恢复逻辑 + 请求体格式全部符合预期")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
