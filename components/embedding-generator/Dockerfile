FROM python:3.11-slim

LABEL maintainer="embedding-generator"
LABEL description="Embedding generator KFP component for RAG pipeline"

WORKDIR /app

# Install system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY component.py .

# Pre-create output and cache directories
RUN mkdir -p /tmp/outputs /root/.cache/huggingface

# Environment variables for model caching
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["python", "component.py"]
