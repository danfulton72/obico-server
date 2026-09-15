FROM openvino/ubuntu22_runtime:2025.1.0

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        onnxruntime-openvino==1.23.0 \
        opencv-python-headless \
    && python3 -m pip install --no-cache-dir \
        -r /app/requirements.txt

COPY . /app

RUN mkdir -p /model_cache/ml_api/onnx \
    && echo "Downloading Obico ONNX failure detection model..." \
    && curl --fail --location \
        --output /model_cache/ml_api/onnx/model-weights.onnx \
        "$(tr -d '\r\n' < /app/model/model-weights.onnx.url)"

EXPOSE 3333
