#!/usr/bin/env bash
#
# setup-kind-cluster.sh
# Creates a Kind (Kubernetes in Docker) cluster configured for Kubeflow local development.
#
# Usage:
#   ./scripts/setup-kind-cluster.sh
#
set -euo pipefail

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CLUSTER_NAME="kubeflow-local"
KIND_CONFIG="configs/kind-config.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIRED_DOCKER_MEM_GB=6   # Minimum: 6 GB for Kubeflow stack (8+ recommended)
REQUIRED_DOCKER_CPUS=4     # Minimum: 4 CPUs

# ──────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ──────────────────────────────────────────────
# Prerequisite checks
# ──────────────────────────────────────────────
check_command() {
  if ! command -v "$1" &>/dev/null; then
    error "'$1' is not installed. Please install it first."
    echo "  Install guide: $2"
    exit 1
  fi
  success "$1 found: $(command -v "$1")"
}

check_docker_running() {
  info "Checking if Docker daemon is running..."
  if ! docker info &>/dev/null; then
    error "Docker is not running. Please start Docker Desktop or the Docker daemon."
    exit 1
  fi
  success "Docker is running."

  # Check Docker resource allocation (best-effort on Docker Desktop)
  local mem_bytes
  mem_bytes=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
  if [[ "$mem_bytes" -gt 0 ]]; then
    local mem_gb=$(( mem_bytes / 1073741824 ))
    if [[ "$mem_gb" -lt "$REQUIRED_DOCKER_MEM_GB" ]]; then
      warn "Docker has ${mem_gb}GB RAM allocated. Recommended: ${REQUIRED_DOCKER_MEM_GB}GB+ for Kubeflow."
      warn "Increase memory in Docker Desktop → Settings → Resources."
    else
      success "Docker memory: ${mem_gb}GB (meets ${REQUIRED_DOCKER_MEM_GB}GB recommendation)."
    fi
  fi

  local cpus
  cpus=$(docker info --format '{{.NCPU}}' 2>/dev/null || echo "0")
  if [[ "$cpus" -gt 0 ]]; then
    if [[ "$cpus" -lt "$REQUIRED_DOCKER_CPUS" ]]; then
      warn "Docker has ${cpus} CPUs. Recommended: ${REQUIRED_DOCKER_CPUS}+ for Kubeflow."
      warn "Increase CPUs in Docker Desktop → Settings → Resources."
    else
      success "Docker CPUs: ${cpus} (meets ${REQUIRED_DOCKER_CPUS} recommendation)."
    fi
  fi
}

# ──────────────────────────────────────────────
# Cluster setup
# ──────────────────────────────────────────────
create_local_dirs() {
  info "Ensuring local mount directories exist..."
  mkdir -p "${PROJECT_ROOT}/data"
  mkdir -p "${PROJECT_ROOT}/models"
  success "Created ./data and ./models directories."
}

delete_existing_cluster() {
  if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    warn "Cluster '${CLUSTER_NAME}' already exists."
    read -rp "Delete and recreate? [y/N]: " confirm
    if [[ "${confirm,,}" == "y" ]]; then
      info "Deleting existing cluster..."
      kind delete cluster --name "${CLUSTER_NAME}"
      success "Cluster deleted."
    else
      info "Keeping existing cluster. Exiting."
      exit 0
    fi
  fi
}

create_cluster() {
  info "Creating Kind cluster '${CLUSTER_NAME}' with Kubernetes v1.29..."
  echo ""
  kind create cluster --config "${PROJECT_ROOT}/${KIND_CONFIG}" --wait 5m
  echo ""
  success "Kind cluster '${CLUSTER_NAME}' created successfully."
}

wait_for_cluster_ready() {
  info "Waiting for all nodes to be Ready..."
  local retries=30
  local delay=10
  for (( i=1; i<=retries; i++ )); do
    local not_ready
    not_ready=$(kubectl get nodes --no-headers 2>/dev/null | grep -cv "Ready" || true)
    if [[ "$not_ready" -eq 0 ]]; then
      success "All nodes are Ready."
      kubectl get nodes -o wide
      return 0
    fi
    info "  Attempt ${i}/${retries}: waiting for nodes... (retrying in ${delay}s)"
    sleep "$delay"
  done
  error "Timed out waiting for nodes to become Ready."
  kubectl get nodes -o wide
  exit 1
}

wait_for_system_pods() {
  info "Waiting for kube-system pods to be running..."
  local retries=30
  local delay=10
  for (( i=1; i<=retries; i++ )); do
    local pending
    pending=$(kubectl get pods -n kube-system --no-headers 2>/dev/null \
      | grep -cvE "Running|Completed" || true)
    if [[ "$pending" -eq 0 ]]; then
      success "All kube-system pods are running."
      return 0
    fi
    info "  Attempt ${i}/${retries}: ${pending} pod(s) not ready... (retrying in ${delay}s)"
    sleep "$delay"
  done
  warn "Some kube-system pods may still be starting. Check with:"
  echo "  kubectl get pods -n kube-system"
}

configure_kubectl_context() {
  info "Configuring kubectl context..."
  kubectl config use-context "kind-${CLUSTER_NAME}"
  success "kubectl context set to 'kind-${CLUSTER_NAME}'."
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
main() {
  echo ""
  echo "=========================================="
  echo "  Kind Cluster Setup for Kubeflow"
  echo "=========================================="
  echo ""

  # Check prerequisites
  check_command "docker" "https://docs.docker.com/get-docker/"
  check_command "kind"   "https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
  check_command "kubectl" "https://kubernetes.io/docs/tasks/tools/"
  echo ""

  check_docker_running
  echo ""

  # Verify config file exists
  if [[ ! -f "${PROJECT_ROOT}/${KIND_CONFIG}" ]]; then
    error "Kind config not found at '${PROJECT_ROOT}/${KIND_CONFIG}'."
    exit 1
  fi
  success "Kind config found: ${KIND_CONFIG}"
  echo ""

  # Setup
  create_local_dirs
  delete_existing_cluster
  echo ""

  create_cluster
  echo ""

  wait_for_cluster_ready
  echo ""

  wait_for_system_pods
  echo ""

  configure_kubectl_context
  echo ""

  # ──────────────────────────────────────────
  # Print next steps
  # ──────────────────────────────────────────
  echo ""
  echo -e "${GREEN}=========================================="
  echo "  Cluster is ready!"
  echo -e "==========================================${NC}"
  echo ""
  echo "Cluster details:"
  echo "  Name:       ${CLUSTER_NAME}"
  echo "  Nodes:      1 control-plane + 1 worker"
  echo "  K8s:        v1.29"
  echo "  Context:    kind-${CLUSTER_NAME}"
  echo ""
  echo "Port mappings (localhost):"
  echo "  Kubeflow Dashboard : http://localhost:8083"
  echo "  KServe Inference   : http://localhost:8082"
  echo "  MLflow (optional)  : http://localhost:5000"
  echo ""
  echo "Local mounts:"
  echo "  ./data   → /mnt/data   (on all nodes)"
  echo "  ./models → /mnt/models (on all nodes)"
  echo ""
  echo -e "${BLUE}Next steps:${NC}"
  echo "  1. Install Kubeflow:"
  echo "       kubectl apply -k \"github.com/kubeflow/manifests/example\""
  echo ""
  echo "  2. Install KServe (if needed):"
  echo "       kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml"
  echo ""
  echo "  3. Verify pods are running:"
  echo "       kubectl get pods -A"
  echo ""
  echo "  4. Access the Kubeflow dashboard:"
  echo "       kubectl port-forward --address 127.0.0.1 svc/ml-pipeline-ui -n kubeflow 8083:80"
  echo "       Open: http://localhost:8083"
  echo ""
  echo "  5. To delete the cluster later:"
  echo "       kind delete cluster --name ${CLUSTER_NAME}"
  echo ""
}

main "$@"
