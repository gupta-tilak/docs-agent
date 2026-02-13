#!/usr/bin/env bash
# ==========================================================================
# deploy-milvus.sh – Deploy Milvus Standalone on a local Kind cluster
# ==========================================================================
# Usage:
#   ./scripts/deploy-milvus.sh              # full deploy
#   ./scripts/deploy-milvus.sh --skip-helm  # apply PVCs + init only
#   ./scripts/deploy-milvus.sh --uninstall  # tear down
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MILVUS_DIR="${ROOT_DIR}/infrastructure/milvus"

NAMESPACE="kubeflow"
HELM_RELEASE="milvus"
HELM_REPO_NAME="milvus"
HELM_REPO_URL="https://zilliztech.github.io/milvus-helm/"
TIMEOUT="600s"

# ── Colours ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight checks ───────────────────────────────────────────────────
check_prerequisites() {
    local missing=()
    for cmd in kubectl helm; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    # Verify cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster. Is your Kind cluster running?"
        exit 1
    fi
    info "Pre-flight checks passed."
}

# ── Ensure namespace exists ──────────────────────────────────────────────
ensure_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        info "Creating namespace '${NAMESPACE}' ..."
        kubectl create namespace "$NAMESPACE"
    else
        info "Namespace '${NAMESPACE}' already exists."
    fi
}

# ── Add / update Helm repo ──────────────────────────────────────────────
setup_helm_repo() {
    info "Adding Milvus Helm repo ..."
    helm repo add "$HELM_REPO_NAME" "$HELM_REPO_URL" 2>/dev/null || true
    helm repo update
}

# ── Apply PVCs ──────────────────────────────────────────────────────────
apply_pvcs() {
    info "Applying persistent volume claims ..."
    kubectl apply -f "${MILVUS_DIR}/pvc.yaml"
}

# ── Install / upgrade Milvus via Helm ────────────────────────────────────
install_milvus() {
    info "Installing Milvus (Helm release '${HELM_RELEASE}') in namespace '${NAMESPACE}' ..."
    helm upgrade --install "$HELM_RELEASE" "${HELM_REPO_NAME}/milvus" \
        --namespace "$NAMESPACE" \
        --values "${MILVUS_DIR}/helm-values.yaml" \
        --timeout "$TIMEOUT" \
        --wait
    info "Helm install complete."
}

# ── Wait for Milvus readiness ───────────────────────────────────────────
wait_for_milvus() {
    info "Waiting for Milvus pods to become ready ..."
    kubectl wait --for=condition=Ready pod \
        -l "app.kubernetes.io/instance=${HELM_RELEASE},app.kubernetes.io/component=standalone" \
        -n "$NAMESPACE" \
        --timeout="$TIMEOUT" 2>/dev/null || \
    kubectl wait --for=condition=Ready pod \
        -l "app=milvus" \
        -n "$NAMESPACE" \
        --timeout="$TIMEOUT" 2>/dev/null || true

    # Fallback: poll the health endpoint via a temporary port-forward
    info "Checking Milvus health endpoint ..."
    local max_attempts=30
    local attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        # Find the milvus pod
        local pod
        pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=${HELM_RELEASE}" \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [[ -z "$pod" ]]; then
            pod=$(kubectl get pods -n "$NAMESPACE" -l "app=milvus" \
                  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        fi

        if [[ -n "$pod" ]]; then
            local phase
            phase=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
            if [[ "$phase" == "Running" ]]; then
                info "Milvus pod '${pod}' is Running."
                break
            fi
        fi

        ((attempt++))
        warn "Attempt ${attempt}/${max_attempts} – waiting 10s ..."
        sleep 10
    done

    if [[ $attempt -ge $max_attempts ]]; then
        error "Milvus did not become ready within the timeout."
        kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=${HELM_RELEASE}" -o wide
        exit 1
    fi
}

# ── Run collection init job ─────────────────────────────────────────────
init_collection() {
    info "Deploying collection initialization job ..."

    # Delete previous job run if it exists (jobs are immutable)
    kubectl delete job milvus-init-collection -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete configmap milvus-init-script -n "$NAMESPACE" 2>/dev/null || true

    kubectl apply -f "${MILVUS_DIR}/init-collection.yaml"

    info "Waiting for init job to complete (up to 5 min) ..."
    kubectl wait --for=condition=Complete job/milvus-init-collection \
        -n "$NAMESPACE" \
        --timeout=300s || {
            error "Init job did not complete successfully."
            warn "Job logs:"
            kubectl logs job/milvus-init-collection -n "$NAMESPACE" --tail=50
            exit 1
        }

    info "Init job finished. Logs:"
    kubectl logs job/milvus-init-collection -n "$NAMESPACE"
}

# ── Verify deployment ───────────────────────────────────────────────────
verify() {
    echo ""
    info "=== Milvus Deployment Summary ==="
    echo ""
    info "Pods:"
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=${HELM_RELEASE}" -o wide 2>/dev/null || \
    kubectl get pods -n "$NAMESPACE" -l "app=milvus" -o wide 2>/dev/null
    echo ""
    info "Services:"
    kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/instance=${HELM_RELEASE}" 2>/dev/null || \
    kubectl get svc -n "$NAMESPACE" -l "app=milvus" 2>/dev/null
    echo ""
    info "PVCs:"
    kubectl get pvc -n "$NAMESPACE" -l "app=milvus"
    echo ""
    info "To access Milvus locally, run:"
    echo "    ./scripts/port-forward-milvus.sh"
    echo ""
    info "gRPC endpoint: localhost:19530"
    info "HTTP  endpoint: localhost:9091"
}

# ── Uninstall ────────────────────────────────────────────────────────────
uninstall() {
    warn "Uninstalling Milvus from namespace '${NAMESPACE}' ..."
    kubectl delete job milvus-init-collection -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete configmap milvus-init-script -n "$NAMESPACE" 2>/dev/null || true
    helm uninstall "$HELM_RELEASE" -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete -f "${MILVUS_DIR}/pvc.yaml" 2>/dev/null || true
    info "Milvus uninstalled."
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
    case "${1:-}" in
        --uninstall)
            check_prerequisites
            uninstall
            exit 0
            ;;
        --skip-helm)
            check_prerequisites
            ensure_namespace
            apply_pvcs
            init_collection
            verify
            exit 0
            ;;
        *)
            check_prerequisites
            ensure_namespace
            setup_helm_repo
            apply_pvcs
            install_milvus
            wait_for_milvus
            init_collection
            verify
            ;;
    esac
}

main "$@"
