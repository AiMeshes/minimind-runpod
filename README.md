# minimind-runpod

给 [minimind](https://github.com/jingyaogong/minimind) 套一层 RunPod 运行外壳：**抗抢占 checkpoint、supervisor 自动续训、network volume 持久化**。

**核心设计：不改动 minimind 一行代码。** 上游明天推新 commit 也不影响你。

---

## 为什么是外壳而不是 fork 改造

minimind 上游非常活跃（60k stars，几乎每天有 commit）。直接改 `trainer/train_pretrain.py` 会让每次同步上游都在处理合并冲突；而外壳模式下，上游改了什么都不影响这层。

这个方案能成立，是因为 minimind 自带了两项关键能力：

| 能力 | 位置 | 意义 |
|---|---|---|
| **原子写 checkpoint** | `trainer_utils.py:74-76` | `.tmp` + `os.replace`，抢占不会留下损坏文件 |
| **`--from_resume 1`** | `train_pretrain.py:104` | 自动找最新 checkpoint 续训 |
| **GPU 数自适应** | `trainer_utils.py:110-114` | ⭐ 卡数变化时自动换算 step |
| **确定性 shuffle** | `train_pretrain.py:160` | `seed + epoch` 播种，恢复后数据顺序一致 |

其中 **GPU 数自适应**对 Spot 特别关键：8 卡被抢占后重开的机器可能只有 4 卡甚至 1 卡，minimind 能直接接上继续训，不需要任何改造。

---

## 持久化怎么实现的（关键）

看 `train_pretrain.py:69` 和 `:118`：

```python
lm_checkpoint(lm_config, weight=args.save_weight, ..., save_dir='../checkpoints')
```

`'../checkpoints'` 是**硬编码**的，而且 minimind 的约定是**从 `trainer/` 目录内运行**（所以还有 `../out`、`../model`、`../dataset`）。

于是只要把一切放在 network volume 上，checkpoint 就自动持久化了：

```
/workspace/                      ← network volume 挂载点
└── minimind/                    ← entrypoint 克隆到这里
    ├── trainer/                 ← cd 到这里运行
    ├── dataset/                 ← 数据
    ├── checkpoints/             ← '../checkpoints' 自动解析到此 ← 持久化
    └── out/                     ← 权重输出
```

**一行 minimind 代码都不用改。**

---

## 快速开始

### 1. 构建镜像

**在 RunPod 的 Pod 里构建，不要在 Apple Silicon 本地 build**（qemu 模拟编译极慢甚至失败）。

```bash
docker build -t <你的dockerhub>/minimind-runpod:latest .
docker push <你的dockerhub>/minimind-runpod:latest
```

### 2. 建 network volume

在 RunPod console 创建，**记住机房** —— volume 是地域锁定的，只有同机房的 GPU 能用它。

### 3. 配置

```bash
cp configs/64m-pretrain.env configs/my-run.env
# 填入 NETWORK_VOLUME_ID 和 IMAGE_NAME
```

### 4. 启动

```bash
export RUNPOD_API_KEY=rpa_...
python launch.py --config configs/my-run.env
```

先用 `--dry-run` 检查请求内容，首次运行前建议确认 API 参数名：

```bash
python -c "import runpod; print(runpod.create_pod.__doc__)"
```

### 5. 挂上监控（强烈建议）

```bash
# crontab -e，每 20 分钟检查一次
*/20 * * * * cd ~/minimind-runpod && RUNPOD_API_KEY=rpa_... /usr/bin/python3 watch.py --config configs/my-run.env >> ~/minimind-watch.log 2>&1
```

**RunPod 被抢占的 Pod 不会自己重启** —— 这是 `watch.py` 存在的唯一理由。

> ⚠️ **`watch.py` 只在 Mac 醒着时运行。** 睡着时 Pod 照常在跑，只是抢占后恢复会被推迟到你唤醒电脑。对跑几天的任务通常可接受；要更可靠就用 $5/月的 VPS 跑这个 cron。
>
> 另外，仓库 60 天无活动会让 GitHub Actions 的定时任务自动禁用 —— 所以这里刻意用本地 cron 而非 Actions。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 只装依赖，不打代码（镜像小、启动快、总是拿最新代码） |
| `entrypoint.sh` | 数据准备 + supervisor 主循环 |
| `launch.py` | 拉起 Pod |
| `watch.py` | 监控 + 抢占后自动重建 |
| `configs/*.env` | 配置 |
| `tests/` | 本地验证（CPU，不花钱）—— 见上节 |

### supervisor 的几个设计点

- **快速失败退避**：连续 `FAST_FAIL_LIMIT`（默认 5）次在 `FAST_FAIL_SECONDS`（默认 60）内失败就停止重试并退出。这区分了"被抢占"（跑了几小时才挂）和"配置写错了"（秒挂），避免后者无限重试烧钱
- **心跳文件**：`$WORKSPACE/.supervisor/heartbeat`，供需要更精确存活判断时使用
- **数据幂等**：`.ready` 哨兵 + 临时文件原子 rename，抢占后重跑不会留下半份数据

---

## 本地验证（不需要 GPU，不花钱）

整个抗抢占设计建立在两个假设上。它们都可以**在本地用 CPU 证伪** ——
如果错了，上云跑几天才发现就太贵了。

```bash
# 首次建环境（uv）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers==4.57.6 datasets==3.6.0

# 跑全套
bash tests/run_all.sh
```

约 40 秒跑完，两项验证：

### 1. 断点续训真的成立吗？

`tests/verify_resume.py` 用 0.53M 参数的极小模型在 CPU 上训练，
在**训练进行到 60% 时 SIGKILL**（不给任何优雅退出的机会），然后用
`--from_resume 1` 重启，断言：

| 断言 | 意义 |
|---|---|
| checkpoint 落在 `../checkpoints/` | 硬编码路径约定成立 → 放 network volume 即可持久化 |
| checkpoint 含 model/optimizer/epoch/step | 恢复所需状态完整 |
| 续训跳过正确的 step 数、从正确的 epoch 开始 | 真的接续了，不是重跑 |
| **学习率与余弦调度曲线吻合** | ⭐ scheduler 状态被恢复 |
| 续训处 lr 已比初始值低 70%+ | 保证上一条有鉴别力（否则接续和重置无法区分） |

最后两条是关键。如果在 60% 处续训后 lr 跳回初始值，就说明只恢复了模型权重、
调度器被重置 —— 训练动力学已经错乱，而这种错误不会报错，只会让结果变差。

> **为什么要跑到 60% 才杀**：minimind 用余弦调度 `lr·(0.1+0.45·(1+cos(π·t/T)))`。
> 在训练前 10% 处 lr 几乎等于初始值，此时"是否恢复"根本无从区分。60% 处
> lr 已降到约 41%，重置会立刻暴露。

### 2. supervisor 能区分「被抢占」和「配置错误」吗？

`tests/verify_supervisor.py` 用假的 `torchrun`（放在 PATH 最前）精确编排退出行为：

| 场景 | 期望 | 为什么重要 |
|---|---|---|
| 训练正常结束 | 退出 0，不重启 | — |
| 崩两次后成功 | 重启到成功为止 | 基本恢复能力 |
| 连续 5 次秒挂 | **熔断退出 1** | 配置写错时无限重试 = 无限烧钱 |
| 跑 2s 后失败 8 次 | **不熔断，持续重启** | 这才是抢占的表现，误判则永远无法恢复 |

第 3、4 条的区别是整个熔断逻辑的关键：抢占是"跑了几小时才挂"，
配置错误是"秒挂"。分不清这两者，要么被抢占后不恢复，要么配置写错时烧钱。

同时做静态检查：`$VAR` 后紧跟非 ASCII 字符在 bash 3.2 下会解析错误（见下）。

---

## 基础镜像

`runpod/pytorch:1.1.0-cu1281-torch260-ubuntu2204`（已核实内容）：

| 项 | 值 |
|---|---|
| Python | 3.12 |
| torch | 2.6.0（与 minimind 的 requirements 对齐） |
| CUDA / cuDNN | 12.8.1 / 9.8 |
| 镜像大小 | 10.6 GB |

**该镜像已预设两件对我们有用的事**：

- `HF_HUB_ENABLE_HF_TRANSFER=1` —— 并行下载已开箱启用
- `HF_HOME=/workspace/.cache/huggingface/` —— HF 缓存落在 network volume 上 ⚠️

---

## 已知限制

- **上游对 `../out/*.pth` 是非原子写**（`train_pretrain.py:68` 直接 `torch.save` 到目标路径）。若抢占恰好落在该保存窗口，这个文件可能损坏。
  **不影响续训** —— `--from_resume 1` 只读 `../checkpoints/*_resume.pth`（该路径由 `lm_checkpoint` 用 `.tmp` + `os.replace` 原子写入）。且训练继续后下次保存会覆盖它。只有在"抢占后**永久停止**训练、又想把 `out/` 权重用于下一阶段"这个组合下才会踩到。
- **`$VAR` 后紧跟非 ASCII 字符在 bash 3.2 下会解析错误**。macOS 自带的就是 bash 3.2，它不做 UTF-8 感知，会把中文字节的当成变量名的一部分（如 `$code（` → 变量 `code\xef\xbc\x88` → unbound variable）。**一律写 `${VAR}`**。容器里是 bash 5 不会踩到，但本地测试会 —— `tests/verify_supervisor.py` 里有静态检查防回归。
- **HF 缓存会吃 network volume 空间**。基座镜像把 `HF_HOME` 指向 `/workspace/.cache/huggingface/`，而 `/workspace` 就是 volume 挂载点。如果用 `datasets` 从 HF 拉数据，缓存会常驻并计费（$0.07/GB/月）。**不需要跨 pod 复用时，在 entrypoint 里 `rm -rf` 掉或把 `HF_HOME` 改到容器盘。**
- **tokenizer 从 `../model/` 加载**（`trainer_utils.py:120`）—— 已包含在 minimind 仓库内，无需额外准备
- **`PretrainDataset` 全量载入内存**（`dataset/lm_dataset.py`，用 `load_dataset('json', ...)`）。64M–350M 量级没问题；放大到 1B / 200B tokens 时需要换成流式或预 tokenize 的 memmap（见成本指南 §9.4）
- **默认不启动 sshd**。本镜像覆盖了基座 ENTRYPOINT，会丢掉 RunPod 的 `/start.sh`（它负责 sshd 和 Jupyter）。设 `START_SSHD=1` 并在 console 配置公钥可恢复 SSH。取舍：保持默认则启动更快（按秒计费），需要调试时再开
- **`watch.py` v1 用 Pod 状态判断存活**，不读心跳文件。Pod 显示 running 但进程卡死的边缘情况检测不到 —— 需要更精确判断时可接 RunPod S3 API 读心跳
- **训练超参走 `TRAIN_ARGS` 环境变量，不走 `docker_args`**。RunPod 的 `docker_args` 是「容器启动命令」而非「追加参数」，用它传超参会把 entrypoint 整个替换掉

---

## 相关

- [runpod-training-cost-guide](https://github.com/AiMeshes/runpod-training-guide) —— 成本模型、GPU 选型、Spot 决策阈值、数据搬运
- [minimind](https://github.com/jingyaogong/minimind) —— 上游项目（Apache-2.0）
- [AiMeshes/minimind](https://github.com/AiMeshes/minimind) —— 本项目的 fork，用于同步上游
