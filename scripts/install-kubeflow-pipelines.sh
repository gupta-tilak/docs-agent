#!/usr/bin/env bash
#
# install-kubeflow-pipelines.sh
# Installs Kubeflow Pipelines (standalone, v2.0+) on a local Kind cluster.
#
# Usage:
#   ./scripts/install-kubeflow-pipelines.sh          # Full install
#   ./scripts/install-kubeflow-pipelines.sh portfwd  # Restart port-forwarding only
#   ./scripts/install-kubeflow-pipelines.sh verify    # Verify services only
#   ./scripts/install-kubeflow-pipelines.sh uninstall # Remove Kubeflow Pipelines
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
UI_PORT=8080
PORTFWD_PID_FILE="/tmp/kfp-portforward.pid"
INSTALL_TIMEOUT=600   # 10 minutes total timeout for pod readiness
POD_READY_INTERVAL=10 # seconds between readiness checks

# Components expected to be running after install
EXPECTED_DEPLOYMENTS=(
  "ml-pipeline"
  "ml-pipeline-ui"
  "ml-pipeline-persistenceagent"
  "ml-pipeline-scheduledworkflow"
  "ml-pipeline-viewer-crd"
  "metadata-grpc-deployment"
  "metadata-envoy-deployment"
  "workflow-controller"
  "mysql"
  "minio"
)

# ──────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    echo "  kubectl: https://kubernetes.io/docs/tasks/tools/"
    echo "  kind:    https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
  fi

  # Verify the cluster exists and kubectl context is correct
  if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    error "Kind cluster '${CLUSTER_NAME}' not found. Run setup-kind-cluster.sh first."
    exit 1
  fi

  kubectl config use-context "${KUBECTL_CONTEXT}" &>/dev/null || {
    error "Cannot switch to kubectl context '${KUBECTL_CONTEXT}'."
    exit 1
  }
  success "kubectl context: ${KUBECTL_CONTEXT}"

  # Verify nodes are ready
  local not_ready
  not_ready=$(kubectl get nodes --no-headers 2>/dev/null | grep -cv "Ready" || true)
  if [[ "$not_ready" -ne 0 ]]; then
    error "Not all cluster nodes are Ready. Fix the cluster first."
    kubectl get nodes
    exit 1
  fi
  success "All cluster nodes are Ready."
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
  trap 'rm -rf "${tmpdir}"' EXIT  # clean up on script exit

  curl -sL "${KFP_MANIFEST_BASE}" | tar -xz -C "${tmpdir}"
  MANIFEST_DIR="${tmpdir}/pipelines-${KFP_VERSION}/manifests/kustomize"

  if [[ ! -d "${MANIFEST_DIR}" ]]; then
    error "Manifest directory not found after extraction."
    error "Expected: ${MANIFEST_DIR}"
    exit 1
  fi
  success "Manifests extracted to temp directory."
  echo ""
}

# ──────────────────────────────────────────────
# Install Kubeflow Pipelines via kustomize
# ──────────────────────────────────────────────
install_kfp() {
  step "Installing Kubeflow Pipelines components..."

  # The standalone overlay includes:
  #   - ML Pipeline API server
  #   - ML Pipeline UI
  #   - Workflow Controller (Argo)
  #   - ML Metadata (gRPC + Envoy)
  #   - Persistence Agent
  #   - Scheduled Workflow controller
  #   - Viewer CRD controller
  #   - MySQL (metadata store)
  #   - MinIO (artifact store)
  local kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic-emissary"

  if [[ ! -d "${kustomize_dir}" ]]; then
    # Fallback for older/different directory layouts
    kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic"
    if [[ ! -d "${kustomize_dir}" ]]; then
      error "Cannot find kustomize overlay directory."
      error "Tried: env/platform-agnostic-emissary and env/platform-agnostic"
      ls -la "${MANIFEST_DIR}/env/" 2>/dev/null || true
      exit 1
    fi
  fi

  info "Using kustomize overlay: $(basename "${kustomize_dir}")"

  # Build and apply in one step; capture output for rollback
  if ! kubectl apply -k "${kustomize_dir}" --namespace "${KUBEFLOW_NAMESPACE}" 2>&1; then
    error "kubectl apply failed. Attempting rollback..."
    rollback
    exit 1
  fi

  success "Kubeflow Pipelines manifests applied."
  echo ""
}

# ──────────────────────────────────────────────
# Wait for pods to be ready
# ──────────────────────────────────────────────
wait_for_pods() {
  step "Waiting for all pods in '${KUBEFLOW_NAMESPACE}' to be ready (timeout: ${INSTALL_TIMEOUT}s)..."

  local elapsed=0
  while [[ "$elapsed" -lt "$INSTALL_TIMEOUT" ]]; do
    local total pending
    total=$(kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    pending=$(kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null \
      | grep -cvE "Running|Completed|Succeeded" || true)

    if [[ "$total" -gt 0 && "$pending" -eq 0 ]]; then
      success "All ${total} pods are running."
      echo ""
      kubectl get pods -n "${KUBEFLOW_NAMESPACE}" -o wide
      echo ""
      return 0
    fi

    info "  ${pending}/${total} pod(s) not ready yet... (${elapsed}s elapsed)"
    sleep "$POD_READY_INTERVAL"
    elapsed=$((elapsed + POD_READY_INTERVAL))
  done

  error "Timed out after ${INSTALL_TIMEOUT}s. Some pods are not ready:"
  kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers | grep -vE "Running|Completed|Succeeded" || true
  echo ""
  warn "You can continue waiting manually:"
  echo "  kubectl get pods -n ${KUBEFLOW_NAMESPACE} -w"
  echo ""
  warn "Or rollback with:"
  echo "  $0 uninstall"
  exit 1
}

# ──────────────────────────────────────────────
# Verify all expected services are running
# ──────────────────────────────────────────────
verify_services() {
  step "Verifying Kubeflow Pipelines services..."
  echo ""

  local all_ok=true

  for deploy in "${EXPECTED_DEPLOYMENTS[@]}"; do
    local status
    status=$(kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" \
      --no-headers -o custom-columns=":status.readyReplicas" 2>/dev/null || echo "NOT_FOUND")

    if [[ "${status}" == "NOT_FOUND" || -z "${status}" || "${status}" == "<none>" || "${status}" == "0" ]]; then
      warn "  [MISSING/NOT READY] ${deploy}"
      all_ok=false
    else
      success "  ${deploy} (${status} replica(s) ready)"
    fi
  done

  echo ""

  # Verify key services have ClusterIP endpoints
  info "Checking service endpoints..."
  local services
  services=$(kubectl get svc -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null)

  for svc_name in ml-pipeline ml-pipeline-ui metadata-grpc-service; do
    if echo "${services}" | grep -q "${svc_name}"; then
      local cluster_ip
      cluster_ip=$(kubectl get svc "${svc_name}" -n "${KUBEFLOW_NAMESPACE}" \
        -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
      success "  svc/${svc_name} → ${cluster_ip}"
    else
      warn "  svc/${svc_name} not found"
      all_ok=false
    fi
  done

  echo ""

  if [[ "${all_ok}" == true ]]; then
    success "All Kubeflow Pipelines services verified."
  else
    warn "Some components are missing or not ready."
    warn "They may still be starting. Re-run verification with:"
    echo "  $0 verify"
  fi
  echo ""
}

# ──────────────────────────────────────────────
# Port-forwarding
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
  step "Setting up port-forwarding for Kubeflow Pipelines UI..."

  stop_port_forward

  # Wait until the ml-pipeline-ui pod is ready
  info "Waiting for ml-pipeline-ui to be ready..."
  kubectl wait --for=condition=available deployment/ml-pipeline-ui \
    -n "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || {
    warn "ml-pipeline-ui deployment not available yet. Port-forward may fail."
  }

  # Start port-forward in background
  kubectl port-forward svc/ml-pipeline-ui -n "${KUBEFLOW_NAMESPACE}" \
    "${UI_PORT}:80" &>/dev/null &
  local pfpid=$!
  echo "${pfpid}" > "${PORTFWD_PID_FILE}"

  # Brief wait to check it didn't crash immediately
  sleep 2
  if kill -0 "${pfpid}" 2>/dev/null; then
    success "Port-forward active (PID ${pfpid})."
    success "Kubeflow Pipelines UI: http://localhost:${UI_PORT}"
  else
    warn "Port-forward process exited. You can start it manually:"
    echo "  kubectl port-forward svc/ml-pipeline-ui -n ${KUBEFLOW_NAMESPACE} ${UI_PORT}:80"
  fi
  echo ""
}

# Helper function users can source into their shell
print_portfwd_helper() {
  cat <<'HELPER_EOF'

# ─── Helper: paste into your shell or add to ~/.bashrc ───
kfp-portfwd() {
  local ns="${KFP_NAMESPACE:-kubeflow}"
  local port="${KFP_UI_PORT:-8080}"
  local pidfile="/tmp/kfp-portforward.pid"

  # Kill existing
  if [[ -f "$pidfile" ]]; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
    rm -f "$pidfile"
  fi

  echo "Starting port-forward: localhost:${port} → ml-pipeline-ui (ns: ${ns})"
  kubectl port-forward svc/ml-pipeline-ui -n "$ns" "${port}:80" &>/dev/null &
  echo $! > "$pidfile"
  sleep 2

  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "✓ Kubeflow Pipelines UI: http://localhost:${port}"
    echo "  PID: $(cat "$pidfile")"
  else
    echo "✗ Port-forward failed. Check pods:"
    echo "  kubectl get pods -n $ns"
  fi
}

kfp-portfwd-stop() {
  local pidfile="/tmp/kfp-portforward.pid"
  if [[ -f "$pidfile" ]]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "Stopped." || echo "Already stopped."
    rm -f "$pidfile"
  else
    echo "No active port-forward found."
  fi
}
# ─────────────────────────────────────────────

HELPER_EOF
}

# ──────────────────────────────────────────────
# Rollback / Uninstall
# ──────────────────────────────────────────────
rollback() {
  warn "Rolling back Kubeflow Pipelines installation..."

  stop_port_forward

  # Delete all resources in the namespace
  info "Deleting all resources in namespace '${KUBEFLOW_NAMESPACE}'..."
  kubectl delete --all deployments,services,statefulsets,configmaps,secrets,serviceaccounts,roles,rolebindings,jobs \
    -n "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || true

  # Delete CRDs installed by Argo / KFP
  info "Cleaning up CRDs..."
  kubectl get crds -o name 2>/dev/null | grep -E "argoproj|kubeflow|pipelines" | while read -r crd; do
    kubectl delete "$crd" --timeout=60s 2>/dev/null || true
  done

  # Delete cluster-scoped resources
  info "Cleaning up cluster roles..."
  kubectl get clusterroles,clusterrolebindings -o name 2>/dev/null \
    | grep -iE "ml-pipeline|argo|kubeflow|pipeline-runner" \
    | while read -r resource; do
        kubectl delete "$resource" 2>/dev/null || true
      done

  # Delete namespace
  info "Deleting namespace '${KUBEFLOW_NAMESPACE}'..."
  kubectl delete namespace "${KUBEFLOW_NAMESPACE}" --timeout=120s 2>/dev/null || true

  success "Rollback complete."
  echo ""
}

# ──────────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────────
print_summary() {
  echo ""
  echo -e "${GREEN}=========================================="
  echo "  Kubeflow Pipelines Installed!"
  echo -e "==========================================${NC}"
  echo ""
  echo "  Version:    v${KFP_VERSION}"
  echo "  Namespace:  ${KUBEFLOW_NAMESPACE}"
  echo "  Context:    ${KUBECTL_CONTEXT}"
  echo "  UI:         http://localhost:${UI_PORT}"
  echo ""
  echo "Useful commands:"
  echo "  kubectl get pods -n ${KUBEFLOW_NAMESPACE}       # Check pod status"
  echo "  kubectl logs -n ${KUBEFLOW_NAMESPACE} <pod>     # View pod logs"
  echo "  $0 portfwd                                      # Restart port-forward"
  echo "  $0 verify                                       # Re-verify services"
  echo "  $0 uninstall                                    # Remove everything"
  echo ""
  echo "Shell helper functions (copy to your ~/.bashrc):"
  print_portfwd_helper
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
main() {
  local action="${1:-install}"

  echo ""
  echo "=========================================="
  echo "  Kubeflow Pipelines Installer"
  echo "  Version: v${KFP_VERSION}"
  echo "=========================================="
  echo ""

  case "${action}" in
    install)
      check_prerequisites
      create_namespace
      download_manifests
      install_kfp
      wait_for_pods
      verify_services
      start_port_forward
      print_summary
      ;;
    portfwd)
      check_prerequisites
      start_port_forward
      ;;
    verify)
      check_prerequisites
      verify_services
      ;;
    uninstall)
      check_prerequisites
      rollback
      ;;
    *)
      echo "Usage: $0 {install|portfwd|verify|uninstall}"
      exit 1
      ;;
  esac
}

main "$@"
