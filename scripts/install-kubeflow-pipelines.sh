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
KFP_VERSION="2.5.0"
KFP_MANIFEST_BASE="https://github.com/kubeflow/pipelines/archive/refs/tags/${KFP_VERSION}.tar.gz"
CLUSTER_NAME="kubeflow-local"
KUBECTL_CONTEXT="kind-${CLUSTER_NAME}"
UI_PORT=8083
PORTFWD_PID_FILE="/tmp/kfp-portforward.pid"
INSTALL_TIMEOUT=900   # 15 minutes total timeout for pod readiness
POD_READY_INTERVAL=15 # seconds between readiness checks

# Global temp directory for manifests (set in download_manifests, cleaned up on EXIT)
MANIFEST_TMPDIR=""

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

# Resource patches: deployment cpu_req mem_req cpu_lim mem_lim
# These override the upstream defaults which are too generous for local/Docker setups
RESOURCE_PATCHES=(
  "ml-pipeline 100m 256Mi 500m 512Mi"
  "ml-pipeline-ui 50m 256Mi 200m 512Mi"
  "ml-pipeline-persistenceagent 50m 64Mi 200m 256Mi"
  "ml-pipeline-scheduledworkflow 50m 64Mi 200m 256Mi"
  "ml-pipeline-viewer-crd 30m 64Mi 100m 128Mi"
  "metadata-grpc-deployment 50m 128Mi 200m 256Mi"
  "metadata-envoy-deployment 50m 64Mi 200m 128Mi"
  "workflow-controller 100m 256Mi 500m 512Mi"
  "mysql 100m 256Mi 500m 512Mi"
  "minio 50m 128Mi 200m 512Mi"
  "cache-server 30m 64Mi 100m 128Mi"
  "cache-deployer 30m 64Mi 100m 128Mi"
  "metadata-writer 30m 64Mi 100m 128Mi"
  "ml-pipeline-visualization-server 30m 64Mi 100m 128Mi"
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
cleanup_tmpdir() {
  if [[ -n "${MANIFEST_TMPDIR:-}" && -d "${MANIFEST_TMPDIR}" ]]; then
    rm -rf "${MANIFEST_TMPDIR}"
  fi
}

download_manifests() {
  step "Downloading Kubeflow Pipelines v${KFP_VERSION} manifests..."

  MANIFEST_TMPDIR=$(mktemp -d)
  trap cleanup_tmpdir EXIT  # clean up on script exit

  curl -sL "${KFP_MANIFEST_BASE}" | tar -xz -C "${MANIFEST_TMPDIR}"
  MANIFEST_DIR="${MANIFEST_TMPDIR}/pipelines-${KFP_VERSION}/manifests/kustomize"

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
ensure_control_plane_schedulable() {
  step "Ensuring control-plane node accepts workloads..."
  kubectl taint nodes --all node-role.kubernetes.io/control-plane- 2>/dev/null || true
  success "Control-plane scheduling enabled."
  echo ""
}

install_kfp() {
  step "Installing Kubeflow Pipelines components..."

  # First: install cluster-scoped resources (CRDs, ClusterRoles, etc.)
  # These must be applied BEFORE the namespace-scoped resources because
  # components like workflow-controller depend on CRDs (workflows.argoproj.io)
  # and crash-loop if the CRD doesn't exist.
  local cluster_scoped_dir="${MANIFEST_DIR}/cluster-scoped-resources"
  if [[ -d "${cluster_scoped_dir}" ]]; then
    info "Applying cluster-scoped resources (CRDs, ClusterRoles)..."
    if ! kubectl apply -k "${cluster_scoped_dir}" 2>&1; then
      error "Failed to apply cluster-scoped resources."
      exit 1
    fi
    # Wait for CRDs to be established before continuing
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

  # Apply namespace-scoped resources
  if ! kubectl apply -k "${kustomize_dir}" 2>&1; then
    error "kubectl apply failed. Attempting rollback..."
    rollback
    exit 1
  fi

  success "Kubeflow Pipelines manifests applied."
  echo ""
}

# ──────────────────────────────────────────────
# Patch manifest YAML files BEFORE kubectl apply
# Fixes known broken image references directly in the downloaded
# YAML files so pods are created with correct images from the start.
# This avoids the race condition of post-apply patching and prevents
# duplicate ReplicaSets.
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
    # Fallback: search all YAML files for the broken image
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
  # which causes ImagePullBackOff on Apple Silicon machines.
  local mysql_deploy="${MANIFEST_DIR}/third-party/mysql/base/mysql-deployment.yaml"
  if [[ -f "${mysql_deploy}" ]]; then
    sed 's|gcr.io/ml-pipeline/mysql:8.0.26|mysql:8.0|g' \
      "${mysql_deploy}" > "${mysql_deploy}.tmp" && mv "${mysql_deploy}.tmp" "${mysql_deploy}"
    success "  mysql → mysql:8.0 (Docker Hub, multi-arch)"
  fi

  # ── Best-effort pre-pull key images into Kind ──
  # This speeds up pod startup; kubelet still pulls if this fails.
  info "  Pre-pulling images into Kind (best-effort)..."
  local prepull_images=(
    "${minio_new}"
    "mysql:8.0"
    "ghcr.io/kubeflow/kfp-api-server:${KFP_VERSION}"
    "ghcr.io/kubeflow/kfp-frontend:${KFP_VERSION}"
    "ghcr.io/kubeflow/kfp-cache-server:${KFP_VERSION}"
  )
  for img in "${prepull_images[@]}"; do
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
# Runs after kubectl apply to confirm images are correct.
# If pre-apply patching missed something, force-patches here.
# ──────────────────────────────────────────────
fix_image_references() {
  step "Verifying container image references..."

  # Safety-net: if minio somehow still has the broken gcr.io image, force-patch it
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

  # Safety-net: if mysql still has gcr.io, force-patch to Docker Hub
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

  # Report ghcr.io component images (should be correct from manifests)
  for deploy in ml-pipeline ml-pipeline-ui cache-server workflow-controller; do
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
# Generate TLS secret for cache-server webhook
# KFP cache-server requires a TLS secret "webhook-server-tls"
# that is normally created by a cert-generation Job. If the job
# didn't run or failed, we generate a self-signed cert manually.
# ──────────────────────────────────────────────
generate_cache_server_tls() {
  step "Ensuring cache-server TLS secret exists..."

  if kubectl get secret webhook-server-tls -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
    success "  Secret 'webhook-server-tls' already exists."
    echo ""
    return 0
  fi

  info "  Generating self-signed TLS cert for cache-server webhook..."

  local tmpdir
  tmpdir=$(mktemp -d)

  # Generate CA
  openssl genrsa -out "${tmpdir}/ca.key" 2048 2>/dev/null
  openssl req -x509 -new -nodes -key "${tmpdir}/ca.key" \
    -subj "/CN=webhook-server-ca" -days 3650 \
    -out "${tmpdir}/ca.crt" 2>/dev/null

  # Generate server cert
  openssl genrsa -out "${tmpdir}/tls.key" 2048 2>/dev/null
  cat > "${tmpdir}/csr.conf" <<EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name
[req_distinguished_name]
[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
subjectAltName = @alt_names
[alt_names]
DNS.1 = cache-server.${KUBEFLOW_NAMESPACE}.svc
DNS.2 = cache-server.${KUBEFLOW_NAMESPACE}.svc.cluster.local
EOF

  openssl req -new -key "${tmpdir}/tls.key" \
    -subj "/CN=cache-server.${KUBEFLOW_NAMESPACE}.svc" \
    -config "${tmpdir}/csr.conf" \
    -out "${tmpdir}/tls.csr" 2>/dev/null

  openssl x509 -req -in "${tmpdir}/tls.csr" \
    -CA "${tmpdir}/ca.crt" -CAkey "${tmpdir}/ca.key" -CAcreateserial \
    -out "${tmpdir}/tls.crt" -days 3650 \
    -extensions v3_req -extfile "${tmpdir}/csr.conf" 2>/dev/null

  # Create the Kubernetes secret with cert.pem/key.pem keys
  # (cache-server mounts at /etc/webhook/certs/ and expects cert.pem + key.pem)
  kubectl create secret generic webhook-server-tls \
    -n "${KUBEFLOW_NAMESPACE}" \
    --from-file=cert.pem="${tmpdir}/tls.crt" \
    --from-file=key.pem="${tmpdir}/tls.key" 2>/dev/null \
    && success "  Created secret 'webhook-server-tls'." \
    || warn "  Failed to create TLS secret. cache-server may not start."

  rm -rf "${tmpdir}"
  echo ""
}

# ──────────────────────────────────────────────
# Clean up old ReplicaSets left after image/resource patches
# to avoid duplicate pods consuming resources
# ──────────────────────────────────────────────
cleanup_stale_replicasets() {
  step "Cleaning up stale ReplicaSets from patched deployments..."

  for deploy in $(kubectl get deployments -n "${KUBEFLOW_NAMESPACE}" -o name 2>/dev/null); do
    # Get old ReplicaSets with 0 desired replicas
    local old_rs
    old_rs=$(kubectl get rs -n "${KUBEFLOW_NAMESPACE}" --no-headers 2>/dev/null \
      | awk '$2 == 0 && $3 == 0 && $4 == 0 {print $1}') || true
    for rs in ${old_rs}; do
      kubectl delete rs "${rs}" -n "${KUBEFLOW_NAMESPACE}" --timeout=30s 2>/dev/null || true
    done
  done
  success "  Stale ReplicaSets cleaned up."
  echo ""
}

# ──────────────────────────────────────────────
# Patch resource requests/limits to fit in Docker-allocated RAM
# ──────────────────────────────────────────────
patch_resources() {
  step "Patching pod resource requests to fit local Docker environment..."

  for entry in "${RESOURCE_PATCHES[@]}"; do
    read -r name cpu_req mem_req cpu_lim mem_lim <<< "${entry}"

    if ! kubectl get deployment "${name}" -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
      info "  ${name} not found, skipping."
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
       || warn "  ${name}: patch failed (will use upstream defaults)"
  done

  # Brief wait for rolling updates triggered by patches
  info "Waiting 10s for patched deployments to begin rolling update..."
  sleep 10

  # Relax ml-pipeline-ui probes — the default 3s initialDelay is too aggressive
  # for resource-constrained local clusters, causing CrashLoopBackOff
  if kubectl get deployment ml-pipeline-ui -n "${KUBEFLOW_NAMESPACE}" &>/dev/null; then
    info "Adjusting ml-pipeline-ui probe timings..."
    kubectl patch deployment ml-pipeline-ui -n "${KUBEFLOW_NAMESPACE}" --type='json' -p='[
      {"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/initialDelaySeconds","value":30},
      {"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/periodSeconds","value":10},
      {"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/timeoutSeconds","value":5},
      {"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/failureThreshold","value":6},
      {"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/initialDelaySeconds","value":15},
      {"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/periodSeconds","value":10},
      {"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/timeoutSeconds","value":5}
    ]' 2>/dev/null && success "  ml-pipeline-ui probes adjusted" \
       || warn "  ml-pipeline-ui probe patch failed"
  fi

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

  # Show resource diagnostics to help debug
  warn "Checking for resource-related scheduling failures..."
  local pending_pods
  pending_pods=$(kubectl get pods -n "${KUBEFLOW_NAMESPACE}" --no-headers --field-selector=status.phase=Pending 2>/dev/null | awk '{print $1}')
  for pod in ${pending_pods}; do
    echo ""
    warn "--- Events for ${pod} ---"
    kubectl describe pod "${pod}" -n "${KUBEFLOW_NAMESPACE}" 2>/dev/null | tail -15
  done

  echo ""
  warn "Node resource usage:"
  kubectl describe nodes | grep -A5 "Allocated resources" || true
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
  kubectl port-forward --address 127.0.0.1 svc/ml-pipeline-ui -n "${KUBEFLOW_NAMESPACE}" \
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
    echo "  kubectl port-forward --address 127.0.0.1 svc/ml-pipeline-ui -n ${KUBEFLOW_NAMESPACE} ${UI_PORT}:80"
  fi
  echo ""
}

# Helper function users can source into their shell
print_portfwd_helper() {
  cat <<'HELPER_EOF'

# ─── Helper: paste into your shell or add to ~/.bashrc ───
kfp-portfwd() {
  local ns="${KFP_NAMESPACE:-kubeflow}"
  local port="${KFP_UI_PORT:-8083}"
  local pidfile="/tmp/kfp-portforward.pid"

  # Kill existing
  if [[ -f "$pidfile" ]]; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
    rm -f "$pidfile"
  fi

  echo "Starting port-forward: localhost:${port} → ml-pipeline-ui (ns: ${ns})"
  kubectl port-forward --address 127.0.0.1 svc/ml-pipeline-ui -n "$ns" "${port}:80" &>/dev/null &
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
      ensure_control_plane_schedulable
      create_namespace
      download_manifests
      patch_manifest_images
      install_kfp
      generate_cache_server_tls
      fix_image_references
      patch_resources
      cleanup_stale_replicasets
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
