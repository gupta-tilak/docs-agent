FROM python:3.11-slim

LABEL maintainer="kubeflow-docs-crawler"
LABEL description="Kubeflow documentation crawler KFP component"

# Install git (required by gitpython for cloning repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY component.py .

# Create output directory
RUN mkdir -p /tmp/outputs

ENTRYPOINT ["python", "component.py"]
