#!/usr/bin/env bash
#
# install-kubeflow-pipelines-lite.sh
# Lightweight Kubeflow Pipelines install for resource-constrained machines (≤8 GB RAM).
#
# What this does differently from the full installer:
#   1. Uses a single-node Kind cluster (kind-config-lite.yaml)
#   2. Disables non-essential KFP components (cache-server, UI, viewer-crd)
#   3. Patches remaining pods with minimal resource requests/limits
#   4. Connects via SDK only (no UI port-forward needed)
#
# Usage:
#   ./scripts/install-kubeflow-pipelines-lite.sh              # Full lightweight install
#   ./scripts/install-kubeflow-pipelines-lite.sh cluster       # Create Kind cluster only
#   ./scripts/install-kubeflow-pipelines-lite.sh kfp           # Install KFP only (cluster exists)
#   ./scripts/install-kubeflow-pipelines-lite.sh portfwd       # Restart API port-forward
#   ./scripts/install-kubeflow-pipelines-lite.sh verify        # Verify services
#   ./scripts/install-kubeflow-pipelines-lite.sh uninstall     # Remove KFP + cluster
#
set -euo pipefail

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
KUBEFLOW_NAMESPACE="kubeflow"
KFP_VERSION="2.3.0"
KFP_MANIFEST_BASE="https://github.com/kubeflow/pipelines/archive/refs/tags/${KFP_VERSION}.tar.gz"
CLUSTER_NAME="kubeflow-local"
KUBECTL_CONTEXT="kind-${CLUSTER_NAME}"
API_PORT=8080
PORTFWD_PID_FILE="/tmp/kfp-api-portforward.pid"
INSTALL_TIMEOUT=600
POD_READY_INTERVAL=15
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIND_CONFIG="${SCRIPT_DIR}/../configs/kind-config-lite.yaml"

# Essential deployments only (no cache-server, UI, viewer-crd)
EXPECTED_DEPLOYMENTS=(
  "ml-pipeline"
  "ml-pipeline-persistenceagent"
  "ml-pipeline-scheduledworkflow"
  "metadata-grpc-deployment"
  "metadata-envoy-deployment"
  "workflow-controller"
  "mysql"
  "minio"
)

# Components to scale down to zero (saves ~1-2 GB)
DISABLE_DEPLOYMENTS=(
  "cache-server"
  "ml-pipeline-ui"
  "ml-pipeline-viewer-crd"
)

# ──────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
step()    { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ──────────────────────────────────────────────
# Prerequisite checks
# ──────────────────────────────────────────────
check_prerequisites() {
  step "Checking prerequisites..."
  local missing=0
  for cmd in kubectl kind; do
    if ! command -v "$cmd" &>/dev/null; then
      error "'$cmd' is not installed."
      missing=1
    else
      success "$cmd found."
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    error "Install missing tools before proceeding."
    exit 1
  fi
}

check_cluster() {
  if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    error "Kind cluster '${CLUSTER_NAME}' not found. Run: $0 cluster"
    exit 1
  fi
  kubectl config use-context "${KUBECTL_CONTEXT}" &>/dev/null || {
    error "Cannot switch to kubectl context '${KUBECTL_CONTEXT}'."
    exit 1
  }
  success "kubectl context: ${KUBECTL_CONTEXT}"
}

# ──────────────────────────────────────────────
# Create lightweight single-node Kind cluster
# ──────────────────────────────────────────────
create_cluster() {
  step "Creating lightweight single-node Kind cluster..."

  if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    warn "Cluster '${CLUSTER_NAME}' already exists."
    info "Delete it first with: kind delete cluster --name ${CLUSTER_NAME}"
    kubectl config use-context "${KUBECTL_CONTEXT}" &>/dev/null
    return 0
  fi

  if [[ ! -f "${KIND_CONFIG}" ]]; then
    error "Kind config not found at ${KIND_CONFIG}"
    exit 1
  fi

  kind create cluster --config "${KIND_CONFIG}" --wait 120s
  success "Single-node Kind cluster '${CLUSTER_NAME}' created."

  # Remove control-plane taint so workloads can schedule on it
  kubectl taint nodes --all node-role.kubernetes.io/control-plane- 2>/dev/null || true
  success "Control-plane taint removed (workloads can schedule on it)."
  echo ""
}

# ──────────────────────────────────────────────
# Namespace setup
# ──────────────────────────────────────────────
create_namespace() {
  step "Creating namespace '${KUBEFLOW_NAMESPACE}'..."
  if kubectl get namespace "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
    warn "Namespace '${KUBEFLOW_NAMESPACE}' already exists."
  else
    kubectl create namespace "${KUBEFLOW_NAMESPACE}"
    success "Namespace '${KUBEFLOW_NAMESPACE}' created."
  fi
  echo ""
}

# ──────────────────────────────────────────────
# Download & extract manifests
# ──────────────────────────────────────────────
download_manifests() {
  step "Downloading Kubeflow Pipelines v${KFP_VERSION} manifests..."
  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "${tmpdir}"' EXIT
  curl -sL "${KFP_MANIFEST_BASE}" | tar -xz -C "${tmpdir}"
  MANIFEST_DIR="${tmpdir}/pipelines-${KFP_VERSION}/manifests/kustomize"
  if [[ ! -d "${MANIFEST_DIR}" ]]; then
    error "Manifest directory not found after extraction."
    exit 1
  fi
  success "Manifests extracted."
  echo ""
}

# ──────────────────────────────────────────────
# Install KFP (apply manifests)
# ──────────────────────────────────────────────
install_kfp() {
  step "Installing Kubeflow Pipelines components..."

  local kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic-emissary"
  if [[ ! -d "${kustomize_dir}" ]]; then
    kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic"
    if [[ ! -d "${kustomize_dir}" ]]; then
      error "Cannot find kustomize overlay directory."
      exit 1
    fi
  fi

  info "Applying kustomize overlay: $(basename "${kustomize_dir}")"
  if ! kubectl apply -k "${kustomize_dir}" --namespace "${KUBEFLOW_NAMESPACE}" 2>&1; then
    error "kubectl apply failed."
    exit 1
  fi
  success "KFP manifests applied."
  echo ""
}

# ──────────────────────────────────────────────
# Disable non-essential components
# ──────────────────────────────────────────────
disable_non_essential() {
  step "Scaling down non-essential components to save memory..."

  for deploy in "${DISABLE_DEPLOYMENTS[@]}"; do
    if kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
      kubectl scale deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" --replicas=0 2>/dev/null || true
      success "  Scaled ${deploy} → 0 replicas"
    else
      info "  ${deploy} not found (skipping)"
    fi
  done
  echo ""
}

# ──────────────────────────────────────────────
# Patch resource requests/limits to minimal values
# ──────────────────────────────────────────────
patch_resources() {
  step "Patching resource requests to fit in 8 GB RAM..."

  # Minimal resource patches for each essential component
  # Format: deployment_name cpu_request mem_request cpu_limit mem_limit
  local patches=(
    "ml-pipeline 50m 128Mi 200m 256Mi"
    "ml-pipeline-persistenceagent 20m 64Mi 100m 128Mi"
    "ml-pipeline-scheduledworkflow 20m 64Mi 100m 128Mi"
    "metadata-grpc-deployment 30m 64Mi 100m 192Mi"
    "metadata-envoy-deployment 20m 64Mi 100m 128Mi"
    "workflow-controller 50m 128Mi 200m 256Mi"
    "mysql 50m 128Mi 200m 512Mi"
    "minio 30m 128Mi 100m 256Mi"
  )

  for entry in "${patches[@]}"; do
    read -r name cpu_req mem_req cpu_lim mem_lim <<< "${entry}"

    if ! kubectl get deployment "${name}" -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
      info "  ${name} not found, skipping."
      continue
    fi

    # Get the first container name
    local container
    container=$(kubectl get deployment "${name}" -n "${KUBEFLOW_NAMESPACE}" \
      -o jsonpath='{.spec.template.spec.containers[0].name}' 2>/dev/null)

    if [[ -z "${container}" ]]; then
      warn "  Could not get container name for ${name}, skipping."
      continue
    fi

    kubectl patch deployment "${name}" -n "${KUBEFLOW_NAMESPACE}" --type='json' -p="[
      {
        \"op\": \"replace\",
        \"path\": \"/spec/template/spec/containers/0/resources\",
        \"value\": {
          \"requests\": {\"cpu\": \"${cpu_req}\", \"memory\": \"${mem_req}\"},
          \"limits\":   {\"cpu\": \"${cpu_lim}\", \"memory\": \"${mem_lim}\"}
        }
      }
    ]" 2>/dev/null && success "  ${name}: cpu=${cpu_req}/${cpu_lim}, mem=${mem_req}/${mem_lim}" \
       || warn "  ${name}: patch failed (may use defaults)"
  done
  echo ""
}

# ──────────────────────────────────────────────
# Wait for essential pods only
# ──────────────────────────────────────────────
wait_for_pods() {
  step "Waiting for essential pods to be ready (timeout: ${INSTALL_TIMEOUT}s)..."

  local elapsed=0
  while [[ "$elapsed" -lt "$INSTALL_TIMEOUT" ]]; do
    local all_ready=true

    for deploy in "${EXPECTED_DEPLOYMENTS[@]}"; do
      local ready
      ready=$(kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
      if [[ -z "${ready}" || "${ready}" == "0" ]]; then
        all_ready=false
        break
      fi
    done

    if [[ "${all_ready}" == true ]]; then
      success "All essential pods are running!"
      echo ""
      kubectl get pods -n "${KUBEFLOW_NAMESPACE}"
      echo ""
      return 0
    fi

    # Show which pods are still not ready
    local not_ready
    not_ready=$(kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null \
      | grep -vE "Running|Completed|Succeeded" | grep -v "0/0" || true)
    local pending_count
    pending_count=$(echo "${not_ready}" | grep -c . || true)

    info "  ${pending_count} pod(s) not ready yet... (${elapsed}s elapsed)"
    sleep "$POD_READY_INTERVAL"
    elapsed=$((elapsed + POD_READY_INTERVAL))
  done

  error "Timed out after ${INSTALL_TIMEOUT}s. Pods not ready:"
  kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers | grep -vE "Running|Completed|Succeeded" || true
  echo ""
  warn "Check logs: kubectl logs -n ${KUBEFLOW_NAMESPACE} <pod-name>"
  warn "Resource usage: kubectl top nodes (if metrics-server is installed)"
  exit 1
}

# ──────────────────────────────────────────────
# Verify services
# ──────────────────────────────────────────────
verify_services() {
  step "Verifying essential KFP services..."
  echo ""
  local all_ok=true

  for deploy in "${EXPECTED_DEPLOYMENTS[@]}"; do
    local status
    status=$(kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" \
      --no-headers -o custom-columns=":status.readyReplicas" 2>/dev/null || echo "NOT_FOUND")
    if [[ "${status}" == "NOT_FOUND" || -z "${status}" || "${status}" == "<none>" || "${status}" == "0" ]]; then
      warn "  [NOT READY] ${deploy}"
      all_ok=false
    else
      success "  ${deploy} (${status} replica(s))"
    fi
  done

  echo ""
  info "Disabled components (scaled to 0):"
  for deploy in "${DISABLE_DEPLOYMENTS[@]}"; do
    info "  ${deploy} (disabled to save resources)"
  done

  echo ""
  info "Checking service endpoints..."
  for svc_name in ml-pipeline metadata-grpc-service; do
    local cluster_ip
    cluster_ip=$(kubectl get svc "${svc_name}" -n "${KUBEFLOW_NAMESPACE}" \
      -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "NOT_FOUND")
    if [[ "${cluster_ip}" != "NOT_FOUND" ]]; then
      success "  svc/${svc_name} → ${cluster_ip}"
    else
      warn "  svc/${svc_name} not found"
      all_ok=false
    fi
  done
  echo ""

  if [[ "${all_ok}" == true ]]; then
    success "All essential services are running."
  else
    warn "Some services are not ready. Re-run: $0 verify"
  fi
  echo ""
}

# ──────────────────────────────────────────────
# Port-forward for KFP API (SDK access)
# ──────────────────────────────────────────────
stop_port_forward() {
  if [[ -f "${PORTFWD_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${PORTFWD_PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      info "Stopping existing port-forward (PID ${pid})..."
      kill "${pid}" 2>/dev/null || true
      sleep 1
    fi
    rm -f "${PORTFWD_PID_FILE}"
  fi
}

start_port_forward() {
  step "Setting up port-forward for KFP API server..."
  stop_port_forward

  info "Waiting for ml-pipeline to be ready..."
  kubectl wait --for=condition=available deployment/ml-pipeline \
    -n "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || {
    warn "ml-pipeline not available yet. Port-forward may fail."
  }

  kubectl port-forward svc/ml-pipeline -n "${KUBEFLOW_NAMESPACE}" \
    "${API_PORT}:8888" &>/dev/null &
  local pfpid=$!
  echo "${pfpid}" > "${PORTFWD_PID_FILE}"

  sleep 2
  if kill -0 "${pfpid}" 2>/dev/null; then
    success "Port-forward active (PID ${pfpid})."
    success "KFP API: http://localhost:${API_PORT}"
  else
    warn "Port-forward exited. Start manually:"
    echo "  kubectl port-forward svc/ml-pipeline -n ${KUBEFLOW_NAMESPACE} ${API_PORT}:8888"
  fi
  echo ""
}

# ──────────────────────────────────────────────
# Uninstall
# ──────────────────────────────────────────────
uninstall() {
  warn "Removing Kubeflow Pipelines and Kind cluster..."
  stop_port_forward

  info "Deleting all resources in '${KUBEFLOW_NAMESPACE}'..."
  kubectl delete --all deployments,services,statefulsets,configmaps,secrets,serviceaccounts,roles,rolebindings,jobs \
    -n "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || true

  info "Cleaning up CRDs..."
  kubectl get crds -o name 2>/dev/null | grep -E "argoproj|kubeflow|pipelines" | while read -r crd; do
    kubectl delete "$crd" --timeout=60s 2>/dev/null || true
  done

  info "Cleaning up cluster roles..."
  kubectl get clusterroles,clusterrolebindings -o name 2>/dev/null \
    | grep -iE "ml-pipeline|argo|kubeflow|pipeline-runner" \
    | while read -r r; do kubectl delete "$r" 2>/dev/null || true; done

  info "Deleting namespace '${KUBEFLOW_NAMESPACE}'..."
  kubectl delete namespace "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || true

  info "Deleting Kind cluster..."
  kind delete cluster --name "${CLUSTER_NAME}" 2>/dev/null || true

  success "Cleanup complete."
  echo ""
}

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print_summary() {
  echo ""
  echo -e "${GREEN}=========================================="
  echo "  KFP Lite Installed (Low-Resource Mode)"
  echo -e "==========================================${NC}"
  echo ""
  echo "  Version:     v${KFP_VERSION}"
  echo "  Namespace:   ${KUBEFLOW_NAMESPACE}"
  echo "  Context:     ${KUBECTL_CONTEXT}"
  echo "  KFP API:     http://localhost:${API_PORT}"
  echo ""
  echo "  Disabled (to save RAM):"
  echo "    - cache-server"
  echo "    - ml-pipeline-ui          (use SDK instead)"
  echo "    - ml-pipeline-viewer-crd"
  echo ""
  echo "Submit a pipeline via SDK:"
  echo "  from kfp.client import Client"
  echo "  client = Client(host='http://localhost:${API_PORT}')"
  echo ""
  echo "Commands:"
  echo "  kubectl get pods -n ${KUBEFLOW_NAMESPACE}"
  echo "  $0 portfwd     # restart API port-forward"
  echo "  $0 verify      # check services"
  echo "  $0 uninstall   # remove everything"
  echo ""
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
main() {
  local action="${1:-install}"

  echo ""
  echo "=========================================="
  echo "  KFP Lite – Low-Resource Installer"
  echo "  Version: v${KFP_VERSION}"
  echo "=========================================="
  echo ""

  case "${action}" in
    install)
      check_prerequisites
      create_cluster
      create_namespace
      download_manifests
      install_kfp
      disable_non_essential
      patch_resources
      wait_for_pods
      verify_services
      start_port_forward
      print_summary
      ;;
    cluster)
      check_prerequisites
      create_cluster
      ;;
    kfp)
      check_prerequisites
      check_cluster
      create_namespace
      download_manifests
      install_kfp
      disable_non_essential
      patch_resources
      wait_for_pods
      verify_services
      start_port_forward
      print_summary
      ;;
    portfwd)
      check_prerequisites
      check_cluster
      start_port_forward
      ;;
    verify)
      check_prerequisites
      check_cluster
      verify_services
      ;;
    uninstall)
      check_prerequisites
      uninstall
      ;;
    *)
      echo "Usage: $0 {install|cluster|kfp|portfwd|verify|uninstall}"
      exit 1
      ;;
  esac
}

main "$@"
