FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/work
COPY requirements-core.txt requirements-models.txt ./
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install --no-cache-dir -r requirements-core.txt -r requirements-models.txt
COPY . .
ENV HF_HOME=/workspace/work/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
CMD ["bash"]
