#!/usr/bin/env python3
"""验证 entrypoint.sh 的 supervisor 行为 —— 不需要 GPU，不花钱。

用假的 `torchrun`（放在 PATH 最前面）精确编排每次训练的退出行为，
断言 supervisor 在四种情形下的反应：

  1. 训练正常结束（exit 0）           → supervisor 退出 0，不再重启
  2. 训练崩溃后恢复（fail, fail, ok） → 重启并最终正常结束
  3. 快速连续失败（配置写错）         → 触发熔断，退出 1，避免无限烧钱
  4. 长时间运行后失败（被抢占）       → **不**触发熔断，继续重启

第 3、4 条的区别是关键：抢占的表现是「跑了几小时才挂」，配置错误是「秒挂」。
分不清这两者的话，要么被抢占后不恢复，要么配置写错时无限烧钱。

用法:
    .venv/bin/python tests/verify_supervisor.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"

# 假 torchrun：按 FAKE_SEQUENCE 依次返回退出码，并可通过 FAKE_SLEEP 模拟长跑。
# 每次调用都往 FAKE_LOG 追加一行，便于断言调用次数。
FAKE_TORCHRUN = """#!/bin/bash
count=0
[ -f "$FAKE_COUNT_FILE" ] && count=$(cat "$FAKE_COUNT_FILE")
count=$((count + 1))
echo "$count" > "$FAKE_COUNT_FILE"
echo "[fake-torchrun] 第 $count 次调用, args=$*" >> "$FAKE_LOG"

# FAKE_SLEEP_SEQ 形如 "0 0 90"：第 n 次调用 sleep 对应秒数（模拟长跑）
sleep_for=$(echo "${FAKE_SLEEP_SEQ:-}" | cut -d' ' -f"$count")
[ -n "$sleep_for" ] && [ "$sleep_for" != "0" ] && sleep "$sleep_for"

code=$(echo "${FAKE_SEQUENCE:-0}" | cut -d' ' -f"$count")
# 序列耗尽后默认成功（0），这样「最后一次调用」可以留空不写
[ -z "$code" ] && code=0
echo "[fake-torchrun] 第 $count 次结束, exit=$code" >> "$FAKE_LOG"
exit "$code"
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def lint_no_ascii_adjacent_vars(path: Path) -> list[str]:
    """找出 `$VAR` 紧跟非 ASCII 字符的写法。

    为什么这是 bug：bash 3.2（macOS 自带）不做 UTF-8 感知，会把后续多字节
    字符当成变量名的一部分，例如 `$code（` 会被解析为变量 `code\\xef\\xbc\\x88`
    → 报 unbound variable。加花括号 `${code}` 即可明确界定边界。
    """
    import re
    pat = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)([^\x00-\x7f])")
    problems = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for m in pat.finditer(line):
            problems.append(f"{path.name}:{i}  ${m.group(1)} 后接 {m.group(2)!r}")
    return problems


def fail(msg: str) -> None:
    print(f"\n  ✗ 断言失败: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


def run_case(name: str, sequence: str, *, sleep_seq: str = "",
             fast_fail_seconds: str = "60", fast_fail_limit: str = "5",
             expected_code: int | None = None,
             expected_calls: int | None = None,
             timeout: float = 90) -> tuple[int, str, int]:
    """跑一个场景，返回 (退出码, 日志, torchrun 调用次数)。"""
    workdir = Path(tempfile.mkdtemp(prefix=f"mm-sup-{name}-"))
    workspace = workdir / "workspace"
    bindir = workdir / "bin"

    # 伪造 minimind 目录，让 prepare_code 跳过克隆
    (workspace / "minimind" / "trainer").mkdir(parents=True)
    (workspace / "minimind" / ".git").mkdir(parents=True)

    bindir.mkdir()
    fake = bindir / "torchrun"
    fake.write_text(FAKE_TORCHRUN)
    fake.chmod(0o755)

    fake_log = workdir / "torchrun.log"
    count_file = workdir / "count"
    fake_log.touch()

    env = os.environ.copy()
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "WORKSPACE": str(workspace),
        "NUM_GPUS": "1",
        "RESTART_DELAY": "0",
        "FAST_FAIL_SECONDS": fast_fail_seconds,
        "FAST_FAIL_LIMIT": fast_fail_limit,
        "FAKE_SEQUENCE": sequence,
        "FAKE_LOG": str(fake_log),
        "FAKE_COUNT_FILE": str(count_file),
    })
    if sleep_seq:
        env["FAKE_SLEEP_SEQ"] = sleep_seq

    # 输出重定向到文件而非 capture_output：即使有孤儿进程短暂持有 fd，
    # 也不会让 subprocess.run 一直等 EOF
    outfile = workdir / "supervisor.log"
    with outfile.open("w") as fh:
        proc = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            env=env, stdout=fh, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
    calls = int(count_file.read_text().strip()) if count_file.exists() else 0
    out = outfile.read_text(errors="ignore")

    log(f"\n── 场景: {name} ──")
    log(f"   torchrun 序列: {sequence!r}"
        + (f", sleep: {sleep_seq!r}" if sleep_seq else ""))
    log(f"   supervisor 退出码: {proc.returncode}, torchrun 调用次数: {calls}")

    if expected_code is not None and proc.returncode != expected_code:
        fail(f"退出码 {proc.returncode}，期望 {expected_code}\n"
             f"     supervisor 输出:\n{out}")
    if expected_calls is not None and calls != expected_calls:
        fail(f"torchrun 调用 {calls} 次，期望 {expected_calls} 次\n"
             f"     supervisor 输出:\n{out}")

    return proc.returncode, out, calls


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 supervisor 行为")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not ENTRYPOINT.exists():
        fail(f"找不到 {ENTRYPOINT}")
    if subprocess.run(["bash", "-n", str(ENTRYPOINT)]).returncode != 0:
        fail("entrypoint.sh 语法错误")

    log("entrypoint.sh 语法 OK")

    # 静态检查：$VAR 紧跟非 ASCII 字符在 bash 3.2 下会解析错误
    problems = lint_no_ascii_adjacent_vars(ENTRYPOINT)
    if problems:
        fail("发现 $VAR 紧跟非 ASCII 字符（请改用 ${VAR}）:\n     "
             + "\n     ".join(problems))
    log("无 $VAR 后紧跟非 ASCII 字符的写法")

    # --- 场景 1: 正常结束，不重启 ---
    _, out, _ = run_case("正常结束", "0", expected_code=0, expected_calls=1)
    if "训练正常结束" not in out:
        fail("未打印「训练正常结束」")
    ok("正常结束 → supervisor 退出 0，不再重启")

    # --- 场景 2: 崩两次后正常结束 ---
    run_case("崩溃后恢复", "1 1 0", expected_code=0, expected_calls=3)
    ok("崩溃两次后恢复 → 重启到成功为止")

    # --- 场景 3: 快速连续失败 → 熔断 ---
    _, out, _ = run_case(
        "快速失败熔断", "1 1 1 1 1 1 1 1",
        fast_fail_seconds="60", fast_fail_limit="5",
        expected_code=1, expected_calls=5,
    )
    if "避免继续烧钱" not in out and "疑似配置错误" not in out:
        fail("熔断时未给出解释性日志")
    ok("连续 5 次秒挂 → 熔断退出 1（不无限重试烧钱）")

    # --- 场景 4: 长时间运行后失败 → 不熔断 ---
    # 模拟被抢占：每次跑 2 秒（> FAST_FAIL_SECONDS=1），连续失败 8 次后成功。
    # 失败次数（8）刻意超过 FAST_FAIL_LIMIT（5）—— 若误判为配置错误就会熔断
    _, out, _ = run_case(
        "长跑后失败不熔断", "1 1 1 1 1 1 1 1 0",
        sleep_seq="2 2 2 2 2 2 2 2",
        fast_fail_seconds="1", fast_fail_limit="5",
        expected_code=0, expected_calls=9,
    )
    if "避免继续烧钱" in out or "疑似配置错误" in out:
        fail("长时间运行后的失败被误判为配置错误并熔断 —— 抢占后将无法恢复")
    ok("8 次「跑 2s 后失败」→ 不熔断（正确识别为抢占，而非配置错误）")

    log("\n" + "=" * 62)
    log("supervisor 行为全部符合预期")
    log("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
