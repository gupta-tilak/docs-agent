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
KFP_VERSION="2.5.0"
KFP_MANIFEST_BASE="https://github.com/kubeflow/pipelines/archive/refs/tags/${KFP_VERSION}.tar.gz"
CLUSTER_NAME="kubeflow-local"
KUBECTL_CONTEXT="kind-${CLUSTER_NAME}"
API_PORT=8080
PORTFWD_PID_FILE="/tmp/kfp-api-portforward.pid"
INSTALL_TIMEOUT=600
POD_READY_INTERVAL=15
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIND_CONFIG="${SCRIPT_DIR}/../configs/kind-config-lite.yaml"

# Global temp directory for manifests (set in download_manifests, cleaned up on EXIT)
MANIFEST_TMPDIR=""

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
cleanup_tmpdir() {
  if [[ -n "${MANIFEST_TMPDIR:-}" && -d "${MANIFEST_TMPDIR}" ]]; then
    rm -rf "${MANIFEST_TMPDIR}"
  fi
}

download_manifests() {
  step "Downloading Kubeflow Pipelines v${KFP_VERSION} manifests..."
  MANIFEST_TMPDIR=$(mktemp -d)
  trap cleanup_tmpdir EXIT
  curl -sL "${KFP_MANIFEST_BASE}" | tar -xz -C "${MANIFEST_TMPDIR}"
  MANIFEST_DIR="${MANIFEST_TMPDIR}/pipelines-${KFP_VERSION}/manifests/kustomize"
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

  # First: install cluster-scoped resources (CRDs, ClusterRoles)
  local cluster_scoped_dir="${MANIFEST_DIR}/cluster-scoped-resources"
  if [[ -d "${cluster_scoped_dir}" ]]; then
    info "Applying cluster-scoped resources (CRDs, ClusterRoles)..."
    if ! kubectl apply -k "${cluster_scoped_dir}" 2>&1; then
      error "Failed to apply cluster-scoped resources."
      exit 1
    fi
    info "Waiting for CRDs to be established..."
    kubectl wait --for condition=established --timeout=60s \
      crd/workflows.argoproj.io \
      crd/cronworkflows.argoproj.io \
      crd/clusterworkflowtemplates.argoproj.io \
      crd/workflowtemplates.argoproj.io 2>/dev/null || \
      warn "Some CRDs may not be ready yet, continuing..."
    success "Cluster-scoped resources applied."
  else
    warn "cluster-scoped-resources/ not found; CRDs may be bundled in the overlay."
  fi

  # Second: install namespace-scoped resources
  local kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic-emissary"
  if [[ ! -d "${kustomize_dir}" ]]; then
    kustomize_dir="${MANIFEST_DIR}/env/platform-agnostic"
    if [[ ! -d "${kustomize_dir}" ]]; then
      error "Cannot find kustomize overlay directory."
      exit 1
    fi
  fi

  info "Applying kustomize overlay: $(basename "${kustomize_dir}")"
  if ! kubectl apply -k "${kustomize_dir}" 2>&1; then
    error "kubectl apply failed."
    exit 1
  fi
  success "KFP manifests applied."
  echo ""
}

# ──────────────────────────────────────────────
# Patch manifest YAML files BEFORE kubectl apply
# Fixes known broken image references directly in the downloaded
# YAML files so pods are created with correct images from the start.
# ──────────────────────────────────────────────
patch_manifest_images() {
  step "Patching manifest images before apply..."

  # ── MinIO: gcr.io/ml-pipeline/minio has been removed from GCR ──
  local minio_deploy="${MANIFEST_DIR}/third-party/minio/base/minio-deployment.yaml"
  local minio_new="minio/minio:RELEASE.2024-06-13T22-53-53Z"
  if [[ -f "${minio_deploy}" ]]; then
    sed "s|gcr.io/ml-pipeline/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance|${minio_new}|g" \
      "${minio_deploy}" > "${minio_deploy}.tmp" && mv "${minio_deploy}.tmp" "${minio_deploy}"
    # Add MINIO_ROOT_USER/PASSWORD aliases (newer MinIO images require them)
    if ! grep -q 'MINIO_ROOT_USER' "${minio_deploy}"; then
      sed '/name: MINIO_ACCESS_KEY/a\        - name: MINIO_ROOT_USER\n          valueFrom:\n            secretKeyRef:\n              name: mlpipeline-minio-artifact\n              key: accesskey' \
        "${minio_deploy}" > "${minio_deploy}.tmp" && mv "${minio_deploy}.tmp" "${minio_deploy}"
      sed '/name: MINIO_SECRET_KEY/a\        - name: MINIO_ROOT_PASSWORD\n          valueFrom:\n            secretKeyRef:\n              name: mlpipeline-minio-artifact\n              key: secretkey' \
        "${minio_deploy}" > "${minio_deploy}.tmp" && mv "${minio_deploy}.tmp" "${minio_deploy}"
    fi
    success "  minio → ${minio_new}"
  else
    warn "  minio-deployment.yaml not at expected path; scanning all manifests..."
    find "${MANIFEST_DIR}" -name '*.yaml' -print0 | while IFS= read -r -d '' f; do
      if grep -q 'gcr.io/ml-pipeline/minio' "$f"; then
        sed "s|gcr.io/ml-pipeline/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance|${minio_new}|g" \
          "$f" > "$f.tmp" && mv "$f.tmp" "$f"
        success "  Patched minio in: $(basename "$f")"
      fi
    done
  fi

  # ── MySQL: gcr.io/ml-pipeline/mysql → Docker Hub fallback ──
  # Use mysql:8.0 (floating tag) instead of 8.0.26 — the pinned tag lacks ARM64 manifests
  local mysql_deploy="${MANIFEST_DIR}/third-party/mysql/base/mysql-deployment.yaml"
  if [[ -f "${mysql_deploy}" ]]; then
    sed 's|gcr.io/ml-pipeline/mysql:8.0.26|mysql:8.0|g' \
      "${mysql_deploy}" > "${mysql_deploy}.tmp" && mv "${mysql_deploy}.tmp" "${mysql_deploy}"
    success "  mysql → mysql:8.0 (Docker Hub, multi-arch)"
  fi

  # ── Best-effort pre-pull key images into Kind ──
  info "  Pre-pulling images into Kind (best-effort)..."
  for img in \
    "${minio_new}" \
    "mysql:8.0" \
    "ghcr.io/kubeflow/kfp-api-server:${KFP_VERSION}"
  do
    if timeout 120 docker pull "${img}" &>/dev/null; then
      kind load docker-image "${img}" --name "${CLUSTER_NAME}" &>/dev/null || true
      success "    Pre-loaded: ${img}"
    else
      info "    Could not pre-pull ${img} (kubelet will pull directly)"
    fi
  done

  echo ""
}

# ──────────────────────────────────────────────
# Post-apply image verification & safety-net
# ──────────────────────────────────────────────
fix_image_references() {
  step "Verifying container image references..."

  # Safety-net: if minio somehow still has the broken gcr.io image, force-patch
  if kubectl get deployment minio -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
    local minio_img
    minio_img=$(kubectl get deployment minio -n "${KUBEFLOW_NAMESPACE}" \
      -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "")
    if [[ "${minio_img}" == *"gcr.io/ml-pipeline/minio"* ]]; then
      warn "  minio still using broken gcr.io image — force-patching..."
      kubectl set image deployment/minio -n "${KUBEFLOW_NAMESPACE}" \
        "minio=minio/minio:RELEASE.2024-06-13T22-53-53Z" 2>/dev/null || true
    else
      success "  minio: ${minio_img}"
    fi
  fi

  # Safety-net: if mysql still has gcr.io, force-patch
  if kubectl get deployment mysql -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
    local mysql_img
    mysql_img=$(kubectl get deployment mysql -n "${KUBEFLOW_NAMESPACE}" \
      -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "")
    if [[ "${mysql_img}" == *"gcr.io/ml-pipeline/mysql"* ]]; then
      warn "  mysql still using gcr.io image — force-patching..."
      kubectl set image deployment/mysql -n "${KUBEFLOW_NAMESPACE}" \
        "mysql=mysql:8.0" 2>/dev/null || true
    else
      success "  mysql: ${mysql_img}"
    fi
  fi

  # Report ghcr.io component images
  for deploy in ml-pipeline workflow-controller; do
    if kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
      local img
      img=$(kubectl get deployment "${deploy}" -n "${KUBEFLOW_NAMESPACE}" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")
      success "  ${deploy}: ${img}"
    fi
  done

  echo ""
}

# ──────────────────────────────────────────────
# Clean up old ReplicaSets left after patching
# ──────────────────────────────────────────────
cleanup_stale_replicasets() {
  step "Cleaning up stale ReplicaSets..."
  local old_rs
  old_rs=$(kubectl get rs -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null \
    | awk '$2 == 0 && $3 == 0 && $4 == 0 {print $1}') || true
  for rs in ${old_rs}; do
    kubectl delete rs "${rs}" -n "${KUBEFLOW_NAMESPACE}" --timeout=30s 2>/dev/null || true
  done
  success "  Stale ReplicaSets cleaned up."
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
      patch_manifest_images
      install_kfp
      fix_image_references
      disable_non_essential
      patch_resources
      cleanup_stale_replicasets
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
      patch_manifest_images
      install_kfp
      fix_image_references
      disable_non_essential
      patch_resources
      cleanup_stale_replicasets
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
