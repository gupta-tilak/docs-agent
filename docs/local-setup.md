# Local Kubeflow Development Setup Guide

> **Run Kubeflow Pipelines + KServe on your laptop** using Kind (Kubernetes in Docker).
> No cloud account required.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Step-by-Step Setup](#step-by-step-setup)
  - [1. Create the Kind Cluster](#1-create-the-kind-cluster)
  - [2. Install Kubeflow Pipelines](#2-install-kubeflow-pipelines)
  - [3. Install KServe](#3-install-kserve)
- [Accessing Components](#accessing-components)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Tear Down & Rebuild](#tear-down--rebuild)
- [Next Steps for Development](#next-steps-for-development)

---

## Prerequisites

### Required Software

| Tool | Min Version | Purpose | Install Link |
|------|-------------|---------|--------------|
| **Docker Desktop** (or Docker Engine) | 24.0+ | Container runtime for Kind nodes | [Get Docker](https://docs.docker.com/get-docker/) |
| **kubectl** | 1.29+ | Kubernetes CLI | [Install kubectl](https://kubernetes.io/docs/tasks/tools/) |
| **Kind** | 0.22+ | Local Kubernetes clusters in Docker | [Install Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) |

### Minimum System Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **RAM** | 8 GB | 12+ GB | Allocated to Docker, not just system total |
| **CPU** | 4 cores | 6+ cores | Allocated to Docker |
| **Disk** | 20 GB free | 40+ GB free | Container images are large |
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 11 (WSL2) | — | Native Linux is fastest |

> **Important (macOS / Windows):** Docker Desktop has its own resource limits.
> Go to **Docker Desktop → Settings → Resources** and increase RAM to 12 GB and CPUs to 6.

### Verify prerequisites

```bash
# Check Docker
docker --version        # Should be 24.0+
docker info             # Should not error (daemon must be running)

# Check kubectl
kubectl version --client  # Should be 1.29+

# Check Kind
kind --version          # Should be 0.22+
```

---

## Quick Start

If you just want to get everything running as fast as possible:

```bash
# Clone the repo
git clone <your-repo-url> && cd docs-agent

# 1. Create the Kind cluster (Kubernetes v1.29, 3 nodes)
./scripts/setup-kind-cluster.sh

# 2. Install Kubeflow Pipelines
./scripts/install-kubeflow-pipelines.sh

# 3. Install KServe (includes cert-manager + Knative)
./scripts/install-kserve.sh

# 4. Verify everything
./scripts/install-kubeflow-pipelines.sh verify
./scripts/install-kserve.sh verify
```

After the scripts complete:

| Component | URL | How to access |
|-----------|-----|---------------|
| Kubeflow Pipelines UI | http://localhost:8080 | Auto port-forward, or `./scripts/install-kubeflow-pipelines.sh portfwd` |
| KServe Inference | http://localhost:8081 | `kubectl port-forward -n kourier-system svc/kourier 8081:80` |
| MLflow (if installed) | http://localhost:5000 | Separate install required |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                             │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ ./data   │  │ ./models │  │ scripts/ │                      │
│  └────┬─────┘  └────┬─────┘  └──────────┘                      │
│       │              │                                          │
│       │  Docker      │                                          │
│  ┌────┼──────────────┼──────────────────────────────────────┐   │
│  │    │   Kind Cluster ("kubeflow-local")                   │   │
│  │    │   Kubernetes v1.29                                  │   │
│  │    ▼              ▼                                      │   │
│  │  /mnt/data    /mnt/models                                │   │
│  │                                                          │   │
│  │  ┌──────────────────┐  ┌───────────┐  ┌───────────┐     │   │
│  │  │  Control Plane   │  │  Worker 1 │  │  Worker 2 │     │   │
│  │  │                  │  │           │  │           │     │   │
│  │  │  Port mappings:  │  │  Runs     │  │  Runs     │     │   │
│  │  │  :8080 → :80     │  │  workload │  │  workload │     │   │
│  │  │  :8081 → :8081   │  │  pods     │  │  pods     │     │   │
│  │  │  :5000 → :5000   │  │           │  │           │     │   │
│  │  └──────────────────┘  └───────────┘  └───────────┘     │   │
│  │                                                          │   │
│  │  Namespaces:                                             │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │ kubeflow          Kubeflow Pipelines             │     │   │
│  │  │   ├─ ml-pipeline        (API server)             │     │   │
│  │  │   ├─ ml-pipeline-ui     (Dashboard)              │     │   │
│  │  │   ├─ workflow-controller (Argo workflows)        │     │   │
│  │  │   ├─ metadata-grpc      (ML Metadata)            │     │   │
│  │  │   ├─ mysql              (Metadata store)         │     │   │
│  │  │   └─ minio              (Artifact store)         │     │   │
│  │  ├─────────────────────────────────────────────────┤     │   │
│  │  │ kserve            KServe Controller              │     │   │
│  │  ├─────────────────────────────────────────────────┤     │   │
│  │  │ knative-serving   Knative (serverless runtime)   │     │   │
│  │  ├─────────────────────────────────────────────────┤     │   │
│  │  │ kourier-system    Kourier (ingress / networking) │     │   │
│  │  ├─────────────────────────────────────────────────┤     │   │
│  │  │ cert-manager      TLS certificate management     │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component dependency chain

```
cert-manager
  └─► Knative Serving + Kourier
        └─► KServe
              └─► InferenceServices (your models)

Kubeflow Pipelines (independent)
  └─► ML Pipeline API, UI, Argo, ML Metadata
```

---

## Step-by-Step Setup

### 1. Create the Kind Cluster

The Kind cluster is configured with 1 control-plane node and 2 worker nodes running Kubernetes v1.29.

**What gets created:**
- 3 Docker containers acting as Kubernetes nodes
- Port mappings from localhost to the cluster
- Volume mounts for `./data` and `./models`

```bash
./scripts/setup-kind-cluster.sh
```

**What the script does:**
1. Verifies Docker is running and has sufficient resources
2. Creates `./data` and `./models` directories
3. Prompts to delete an existing cluster if one exists
4. Creates the cluster using `configs/kind-config.yaml`
5. Waits for all nodes and system pods to be ready
6. Sets the `kubectl` context

**Verify the cluster:**

```bash
# Check nodes (expect 3: 1 control-plane + 2 workers)
kubectl get nodes

# Expected output:
# NAME                           STATUS   ROLES           AGE   VERSION
# kubeflow-local-control-plane   Ready    control-plane   2m    v1.29.2
# kubeflow-local-worker          Ready    <none>          2m    v1.29.2
# kubeflow-local-worker2         Ready    <none>          2m    v1.29.2

# Check system pods
kubectl get pods -n kube-system
```

### 2. Install Kubeflow Pipelines

Installs the standalone Kubeflow Pipelines v2.3.0 deployment.

```bash
./scripts/install-kubeflow-pipelines.sh
```

**What gets installed:**
- ML Pipeline API server and UI
- Argo Workflow Controller
- ML Metadata (gRPC + Envoy)
- MySQL (metadata storage)
- MinIO (artifact storage)
- Persistence Agent, Scheduled Workflow, Viewer CRD controllers

**Estimated time:** 3–8 minutes depending on internet speed and system resources.

**Verify:**

```bash
./scripts/install-kubeflow-pipelines.sh verify

# Or manually:
kubectl get pods -n kubeflow
```

### 3. Install KServe

Installs KServe and all its dependencies (cert-manager, Knative Serving, Kourier).

```bash
./scripts/install-kserve.sh
```

**What gets installed (in order):**
1. **cert-manager v1.14.5** — TLS certificate management
2. **Knative Serving v1.14.1** — Serverless runtime with Kourier networking
3. **KServe v0.13.1** — Model inference platform

**Configuration applied:**
- Inference domain: `example.com`
- Scale-to-zero: enabled (`minReplicas: 0`, 30s grace period)
- GPU affinity labels: pre-configured for `accelerator: nvidia-gpu`
- Networking: Kourier (lightweight alternative to Istio)

**Estimated time:** 5–10 minutes.

**Verify:**

```bash
./scripts/install-kserve.sh verify
```

---

## Accessing Components

### Kubeflow Pipelines UI

The install script automatically sets up port-forwarding. If it stops:

```bash
# Restart port-forwarding
./scripts/install-kubeflow-pipelines.sh portfwd

# Or manually:
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

Open: **http://localhost:8080**

### KServe Inference Endpoint

```bash
# Port-forward Kourier gateway
kubectl port-forward -n kourier-system svc/kourier 8081:80
```

Then send requests with the appropriate `Host` header:

```bash
curl -H "Host: <model-name>.<namespace>.example.com" \
     -H "Content-Type: application/json" \
     -d '{"instances": [[1.0, 2.0, 3.0, 4.0]]}' \
     http://localhost:8081/v1/models/<model-name>:predict
```

### Deploy & Test a Sample Model

```bash
# Deploy the sklearn Iris test model
./scripts/install-kserve.sh test

# Or manually:
kubectl create namespace kserve-test
kubectl apply -f configs/test-inference-service.yaml -n kserve-test

# Wait for it to be ready
kubectl get inferenceservice sklearn-iris -n kserve-test -w

# Test inference
kubectl port-forward -n kourier-system svc/kourier 8081:80 &
curl -v \
  -H "Host: sklearn-iris.kserve-test.example.com" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]}' \
  http://localhost:8081/v1/models/sklearn-iris:predict

# Expected response: {"predictions": [1, 1]}
```

### kubectl Context

All scripts set the context automatically. To switch manually:

```bash
kubectl config use-context kind-kubeflow-local
```

---

## Validation

Run these checks to confirm all components are healthy:

### 1. Cluster health

```bash
# All nodes should be "Ready"
kubectl get nodes

# No pods in error state across all namespaces
kubectl get pods -A | grep -vE "Running|Completed"
```

### 2. Kubeflow Pipelines

```bash
# Run the built-in verification
./scripts/install-kubeflow-pipelines.sh verify

# Key pods to check
kubectl get pods -n kubeflow

# Test API connectivity
kubectl port-forward svc/ml-pipeline -n kubeflow 8888:8888 &
curl http://localhost:8888/apis/v2beta1/healthz
# Expected: {"status":"HEALTHY"}
```

### 3. KServe

```bash
# Run the built-in verification
./scripts/install-kserve.sh verify

# Check CRDs are installed
kubectl get crd | grep kserve

# Verify cert-manager is issuing certificates
kubectl get certificates -A

# Check Knative Serving is responding
kubectl get ksvc -A
```

### 4. End-to-end test

```bash
# Deploy and test the sklearn model
./scripts/install-kserve.sh test
```

---

## Troubleshooting

### Docker is not running

```
[ERROR] Docker is not running. Please start Docker Desktop or the Docker daemon.
```

**Solution:**
- **macOS/Windows:** Open Docker Desktop and wait for it to start
- **Linux:** `sudo systemctl start docker`

### Insufficient Docker resources

```
[WARN] Docker has 4GB RAM allocated. Recommended: 12GB+
```

**Solution:**
Open **Docker Desktop → Settings → Resources** and set:
- Memory: **12 GB** (minimum 8 GB)
- CPUs: **6** (minimum 4)
- Disk: **40 GB**

Restart Docker Desktop after changing settings.

### Port conflicts

```
Error: listen tcp 127.0.0.1:8080: bind: address already in use
```

**Solution:** Find and stop the process using the port:

```bash
# Find what's using port 8080
lsof -i :8080

# Kill it (replace <PID> with the actual PID)
kill <PID>

# Or use a different port — edit configs/kind-config.yaml:
#   hostPort: 9080  (instead of 8080)
```

Common port conflicts:
| Port | Common culprit | Fix |
|------|----------------|-----|
| 8080 | Other web servers, Jenkins | Change `hostPort` in kind-config.yaml |
| 8081 | Other services | Change `hostPort` in kind-config.yaml |
| 5000 | macOS AirPlay Receiver | Disable in System Settings → AirDrop & Handoff |

### Mount path errors (macOS)

```
docker: Error response from daemon: path /host_mnt/... is not a shared or slave mount
```

**Solution:** This is caused by `propagation: HostToContainer` in mount configs. The current `configs/kind-config.yaml` already has this fixed. If you see it, ensure no `propagation:` field is set in the mount config — the default (`None`) works on macOS.

### Pods stuck in Pending state

```bash
# Check why a pod is pending
kubectl describe pod <pod-name> -n <namespace>
```

**Common causes:**

| Reason | Solution |
|--------|----------|
| `Insufficient memory` | Increase Docker Desktop RAM |
| `Insufficient cpu` | Increase Docker Desktop CPUs |
| `0/3 nodes are available: 3 node(s) had untolerated taint` | Wait for nodes to be ready, or remove taints |
| `no persistent volumes available` | MinIO/MySQL needs PVs — Kind auto-provisions them; wait a moment |

### Pods in CrashLoopBackOff

```bash
# Check pod logs
kubectl logs <pod-name> -n <namespace> --previous

# Common fixes:
# 1. Restart the pod
kubectl delete pod <pod-name> -n <namespace>

# 2. Restart the deployment
kubectl rollout restart deployment/<deployment-name> -n <namespace>

# 3. Check resource limits
kubectl describe pod <pod-name> -n <namespace> | grep -A5 "Limits\|Requests"
```

### ImagePullBackOff

```bash
# Check the event
kubectl describe pod <pod-name> -n <namespace> | grep -A3 "Events"
```

**Solution:** Usually a network issue. Check:
1. Internet connectivity: `curl -I https://registry.k8s.io`
2. Docker Hub rate limits: Log in with `docker login`
3. DNS resolution inside the cluster: `kubectl run test --rm -it --image=busybox -- nslookup google.com`

### KFP manifests fail to apply

```
error: unable to recognize "...": no matches for kind "..." in version "..."
```

**Solution:** CRDs may not have finished registering. Wait 30 seconds and retry:

```bash
sleep 30
./scripts/install-kubeflow-pipelines.sh
```

### cert-manager webhook not ready

```
Internal error occurred: failed calling webhook "webhook.cert-manager.io"
```

**Solution:** The cert-manager webhook needs time to start. The install script waits for it, but if it occurs:

```bash
kubectl wait --for=condition=available deployment/cert-manager-webhook \
  -n cert-manager --timeout=180s

# Then retry
./scripts/install-kserve.sh
```

### kubectl context is wrong

```bash
# List all contexts
kubectl config get-contexts

# Switch to the Kind cluster
kubectl config use-context kind-kubeflow-local
```

### Nuclear option: start over

If everything is broken beyond repair:

```bash
# Delete the entire cluster
kind delete cluster --name kubeflow-local

# Remove leftover Docker resources
docker system prune -f

# Start fresh
./scripts/setup-kind-cluster.sh
./scripts/install-kubeflow-pipelines.sh
./scripts/install-kserve.sh
```

---

## Tear Down & Rebuild

### Remove individual components

```bash
# Remove KServe (keeps Kubeflow Pipelines)
./scripts/install-kserve.sh uninstall

# Remove Kubeflow Pipelines (keeps KServe)
./scripts/install-kubeflow-pipelines.sh uninstall
```

### Delete everything

```bash
# Delete the entire Kind cluster
kind delete cluster --name kubeflow-local

# Verify
kind get clusters                # Should not list kubeflow-local
docker ps | grep kubeflow-local  # Should be empty
```

### Rebuild from scratch

```bash
./scripts/setup-kind-cluster.sh
./scripts/install-kubeflow-pipelines.sh
./scripts/install-kserve.sh
```

### Clean up disk space

```bash
# Remove unused Docker images (recovers several GB)
docker image prune -a

# Full Docker system cleanup
docker system prune -a --volumes
```

---

## Next Steps for Development

### 1. Build and deploy your own models

Create an `InferenceService` YAML (use `configs/test-inference-service.yaml` as a template):

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
spec:
  predictor:
    minReplicas: 0
    model:
      modelFormat:
        name: sklearn    # or: tensorflow, pytorch, xgboost, onnx
      storageUri: "gs://my-bucket/my-model"
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
```

### 2. Create Kubeflow Pipelines

Use the KFP SDK to build pipelines:

```bash
pip install kfp==2.3.0
```

```python
from kfp import dsl

@dsl.component
def train_model(data_path: str) -> str:
    # Your training logic
    return "model_path"

@dsl.pipeline(name="my-pipeline")
def my_pipeline():
    train_task = train_model(data_path="/mnt/data/training.csv")
```

### 3. Use local mounts for data

Place data files in `./data/` and model artifacts in `./models/` — they are available at `/mnt/data` and `/mnt/models` inside all cluster nodes.

### 4. Load local Docker images into Kind

If you build custom container images locally:

```bash
# Build your image
docker build -t my-custom-image:latest .

# Load it into the Kind cluster (no registry needed)
kind load docker-image my-custom-image:latest --name kubeflow-local

# Reference it in your YAML
# image: my-custom-image:latest
# imagePullPolicy: Never
```

### 5. Set up MLflow (optional)

The cluster has port 5000 pre-mapped for MLflow. Deploy it as a Kubernetes service or run it locally and point to the cluster's MinIO for artifact storage.

### 6. Monitor cluster resources

```bash
# Node resource usage
kubectl top nodes

# Pod resource usage (requires metrics-server)
kubectl top pods -A

# Install metrics-server for Kind:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

---

## File Reference

| File | Purpose |
|------|---------|
| `configs/kind-config.yaml` | Kind cluster configuration (nodes, ports, mounts) |
| `configs/test-inference-service.yaml` | Sample sklearn InferenceService for testing KServe |
| `scripts/setup-kind-cluster.sh` | Creates the Kind Kubernetes cluster |
| `scripts/install-kubeflow-pipelines.sh` | Installs Kubeflow Pipelines (standalone v2.3.0) |
| `scripts/install-kserve.sh` | Installs KServe + cert-manager + Knative + Kourier |

---

## Version Reference

| Component | Version |
|-----------|---------|
| Kubernetes | v1.29.2 |
| Kubeflow Pipelines | v2.3.0 |
| KServe | v0.13.1 |
| Knative Serving | v1.14.1 |
| cert-manager | v1.14.5 |
| Networking | Kourier |
