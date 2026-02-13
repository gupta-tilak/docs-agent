FROM python:3.11-slim

LABEL maintainer="milvus-indexer"
LABEL description="Milvus indexer KFP component for RAG pipeline"

WORKDIR /app

# Install system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY component.py .

# Pre-create output directory
RUN mkdir -p /tmp/outputs

ENTRYPOINT ["python", "component.py"]
