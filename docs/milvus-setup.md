# Milvus Setup & Management Guide

This document covers deploying **Milvus Standalone** on a local Kind cluster, initialising the `kubeflow_docs` collection, and day-to-day management tasks.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Accessing Milvus](#accessing-milvus)
6. [Health Checks & Verification](#health-checks--verification)
7. [Collection Schema](#collection-schema)
8. [Management Tasks](#management-tasks)
9. [Troubleshooting](#troubleshooting)
10. [Uninstalling](#uninstalling)

---

## Prerequisites

| Tool        | Minimum Version | Install                                   |
| ----------- | --------------- | ----------------------------------------- |
| **Kind**    | 0.20+           | `brew install kind`                        |
| **kubectl** | 1.28+           | `brew install kubectl`                     |
| **Helm**    | 3.12+           | `brew install helm`                        |
| **curl**    | any             | pre-installed on macOS                     |

Your Kind cluster must be running (`kind get clusters` should show `kubeflow-local`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Kind Cluster                         │
│  namespace: kubeflow                                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Milvus Standalone Pod                   │  │
│  │                                                   │  │
│  │  ┌─────────┐  ┌───────┐  ┌────────┐             │  │
│  │  │ Milvus  │  │ etcd  │  │ MinIO  │             │  │
│  │  │ :19530  │  │ :2379 │  │ :9000  │             │  │
│  │  │ :9091   │  │       │  │        │             │  │
│  │  └────┬────┘  └───┬───┘  └───┬────┘             │  │
│  │       │           │          │                    │  │
│  │  ┌────┴────┐ ┌────┴────┐ ┌───┴─────┐            │  │
│  │  │ PVC     │ │ PVC     │ │ PVC     │            │  │
│  │  │ 10Gi    │ │ 2Gi     │ │ 5Gi     │            │  │
│  │  └─────────┘ └─────────┘ └─────────┘            │  │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────┐                             │
│  │ ClusterIP Service      │                             │
│  │ milvus:19530 (gRPC)    │                             │
│  │ milvus:9091  (HTTP)    │                             │
│  └────────────────────────┘                             │
└─────────────────────────────────────────────────────────┘
          │ port-forward
    localhost:19530 / localhost:9091
```

---

## Quick Start

```bash
# One-command deploy (Helm + PVCs + init job)
./scripts/deploy-milvus.sh

# Access Milvus from your machine
./scripts/port-forward-milvus.sh
```

---

## Step-by-Step Deployment

### 1. Add the Helm repo

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

### 2. Create the namespace (if not present)

```bash
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Apply PVCs

```bash
kubectl apply -f infrastructure/milvus/pvc.yaml
```

### 4. Install Milvus via Helm

```bash
helm upgrade --install milvus milvus/milvus \
  --namespace kubeflow \
  --values infrastructure/milvus/helm-values.yaml \
  --timeout 600s \
  --wait
```

### 5. Verify pods

```bash
kubectl get pods -n kubeflow -l app.kubernetes.io/instance=milvus -o wide
```

Expected output: all pods `Running` and `1/1 Ready`.

### 6. Initialise the collection

```bash
kubectl apply -f infrastructure/milvus/init-collection.yaml
kubectl wait --for=condition=Complete job/milvus-init-collection -n kubeflow --timeout=300s
kubectl logs job/milvus-init-collection -n kubeflow
```

---

## Accessing Milvus

### Port-forward (foreground)

```bash
./scripts/port-forward-milvus.sh
# gRPC: localhost:19530
# HTTP:  localhost:9091
```

### Port-forward (background)

```bash
./scripts/port-forward-milvus.sh --bg    # start
./scripts/port-forward-milvus.sh --stop  # stop
```

### From inside the cluster

Other pods in the `kubeflow` namespace can connect directly:

```
host: milvus.kubeflow.svc.cluster.local
port: 19530
```

---

## Health Checks & Verification

### HTTP health endpoint

```bash
curl http://localhost:9091/healthz
# Expected: {"status":"ok"}
```

### Milvus metrics

```bash
curl http://localhost:9091/metrics | head -20
```

### Verify collection with Python

```bash
pip install pymilvus
python -c "
from pymilvus import connections, utility, Collection
connections.connect(host='localhost', port='19530')
print('Server version:', utility.get_server_version())
print('Collections:', utility.list_collections())
col = Collection('kubeflow_docs')
print('Schema:', col.schema)
print('Indexes:', [i.params for i in col.indexes])
print('Entities:', col.num_entities)
"
```

### Kubernetes-level checks

```bash
# Pod status
kubectl get pods -n kubeflow -l app.kubernetes.io/instance=milvus

# Events (useful for debugging)
kubectl get events -n kubeflow --sort-by=.lastTimestamp | grep milvus

# PVC status
kubectl get pvc -n kubeflow -l app=milvus
```

---

## Collection Schema

The `kubeflow_docs` collection has the following schema:

| Field       | Type           | Details                                     |
| ----------- | -------------- | ------------------------------------------- |
| `chunk_id`  | VARCHAR(256)   | Primary key – unique chunk identifier       |
| `embedding` | FLOAT_VECTOR   | 384 dimensions (all-MiniLM-L6-v2)           |
| `text`      | VARCHAR(65535) | Raw text content of the chunk               |
| `source`    | VARCHAR(1024)  | Source URL or file path                     |
| `metadata`  | JSON           | Arbitrary key-value metadata                |

**Index:** IVF_FLAT on `embedding`, metric = `COSINE`, nlist = 128.

---

## Management Tasks

### Re-run the init job

```bash
kubectl delete job milvus-init-collection -n kubeflow
kubectl apply -f infrastructure/milvus/init-collection.yaml
```

### Drop and recreate the collection

```python
from pymilvus import connections, utility
connections.connect(host='localhost', port='19530')
utility.drop_collection('kubeflow_docs')
# Then re-run the init job
```

### Scale resources

Edit `infrastructure/milvus/helm-values.yaml` and upgrade:

```bash
helm upgrade milvus milvus/milvus \
  --namespace kubeflow \
  --values infrastructure/milvus/helm-values.yaml \
  --timeout 600s
```

### Backup data (flush + export)

```python
from pymilvus import connections, Collection
connections.connect(host='localhost', port='19530')
col = Collection('kubeflow_docs')
col.flush()
print(f"Flushed {col.num_entities} entities")
```

MinIO data is persisted in the `milvus-minio-data` PVC. For a full backup, snapshot the PVCs.

### View logs

```bash
# Milvus standalone
kubectl logs -n kubeflow -l app.kubernetes.io/instance=milvus -c standalone --tail=100 -f

# etcd
kubectl logs -n kubeflow -l app.kubernetes.io/instance=milvus -c etcd --tail=50

# MinIO
kubectl logs -n kubeflow -l app.kubernetes.io/instance=milvus -c minio --tail=50
```

---

## Troubleshooting

### Pod stuck in `Pending`

```bash
kubectl describe pod -n kubeflow -l app.kubernetes.io/instance=milvus
```

Common causes on Kind:
- **Insufficient resources** – increase Docker Desktop memory to ≥ 8 Gi.
- **PVC not binding** – Kind's default StorageClass should auto-provision. Check with `kubectl get sc`.

### Init job fails to connect

```bash
kubectl logs job/milvus-init-collection -n kubeflow
```

Usually means Milvus isn't ready yet. Delete the job and re-apply – it has retry logic built in (30 attempts × 10 s).

### Port-forward drops

The `kubectl port-forward` process can die if the pod restarts. Simply re-run:

```bash
./scripts/port-forward-milvus.sh
```

### Out of disk space

```bash
kubectl get pvc -n kubeflow
```

If PVCs are full, increase the size in `infrastructure/milvus/pvc.yaml` and in `helm-values.yaml`, then re-apply.

---

## Uninstalling

```bash
# Automated teardown
./scripts/deploy-milvus.sh --uninstall

# Or manual steps
helm uninstall milvus -n kubeflow
kubectl delete -f infrastructure/milvus/init-collection.yaml
kubectl delete -f infrastructure/milvus/pvc.yaml
```

> **Warning:** Deleting PVCs destroys all stored vectors and data.
