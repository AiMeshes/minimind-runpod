# minimind 训练外壳镜像
#
# 只装依赖，不打包 minimind 代码 —— 代码由 entrypoint.sh 在运行时克隆，
# 这样镜像小、启动快（按秒计费，启动时间就是钱），且总是拿到最新代码。
#
# 构建（在 RunPod 的 Pod 里构建，不要在 Apple Silicon 本地 build）：
#   docker build -t <user>/minimind-runpod:latest .
#   docker push <user>/minimind-runpod:latest
# 基础镜像说明（已核实）:
#   python 3.12 / torch 2.6.0 / CUDA 12.8.1 / cuDNN 9.8
#   已预设 HF_HUB_ENABLE_HF_TRANSFER=1（hf_transfer 无需额外配置）
#   已预设 HF_HOME=/workspace/.cache/huggingface/ ← 注意：HF 缓存会占用
#     network volume 空间，见 README「已知限制」
FROM runpod/pytorch:1.1.0-cu1281-torch260-ubuntu2204

# INSTALL_SERVING=1 时额外装 web_demo / serve_openai_api 需要的依赖
ARG INSTALL_SERVING=0

WORKDIR /opt/shell

# --- 依赖层（变化少，放前面利用缓存）---
# 注意：minimind 的 requirements.txt 里 torch 是注释掉的，由基础镜像提供
RUN pip install --no-cache-dir \
        "transformers==4.57.6" \
        "datasets==3.6.0" \
        "numpy==1.26.4" \
        "einops==0.8.1" \
        "wandb==0.22.3" \
        "swanlab==0.9.8" \
        "jsonlines==4.0.0" \
        "modelscope==1.37.0" \
        "hf_transfer" \
        "rich==13.7.1"

# trl 仅 RLHF/GRPO 阶段需要，且依赖较重，单独一层便于单独失效
RUN pip install --no-cache-dir "trl==0.13.0"

RUN if [ "$INSTALL_SERVING" = "1" ]; then \
        pip install --no-cache-dir \
            "Flask==3.0.3" "Flask_Cors==4.0.0" "streamlit==1.50.0" \
            "ngrok==1.4.0" "openai==1.59.6" "sentencepiece==0.2.0"; \
    fi

COPY entrypoint.sh /opt/shell/entrypoint.sh
RUN chmod +x /opt/shell/entrypoint.sh

# 工作区挂载点：network volume 会挂在这里，checkpoint 由此持久化
WORKDIR /workspace

ENTRYPOINT ["/opt/shell/entrypoint.sh"]
