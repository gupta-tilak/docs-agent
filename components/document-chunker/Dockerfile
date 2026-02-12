FROM python:3.11-slim

LABEL maintainer="document-chunker"
LABEL description="Document chunker KFP component for RAG pipeline"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY component.py .

# Create output directory
RUN mkdir -p /tmp/outputs

ENTRYPOINT ["python", "component.py"]
