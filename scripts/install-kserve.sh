#!/usr/bin/env bash
#
# install-kserve.sh
# Installs KServe and its dependencies (cert-manager, Knative Serving, Kourier)
# on a local Kind cluster.
#
# Usage:
#   ./scripts/install-kserve.sh              # Full install
#   ./scripts/install-kserve.sh verify       # Verify installation only
#   ./scripts/install-kserve.sh test         # Deploy test InferenceService
#   ./scripts/install-kserve.sh uninstall    # Remove KServe + dependencies
#
set -euo pipefail

# ──────────────────────────────────────────────
# Versions
# ──────────────────────────────────────────────
CERT_MANAGER_VERSION="v1.14.5"
KNATIVE_VERSION="v1.14.1"
KSERVE_VERSION="v0.13.1"

# Kourier follows minor-only releases (no patch releases like Knative Serving)
KOURIER_VERSION="v1.14.0"

# ──────────────────────────────────────────────
# Cluster / Namespace config
# ──────────────────────────────────────────────
CLUSTER_NAME="kubeflow-local"
KUBECTL_CONTEXT="kind-${CLUSTER_NAME}"
KSERVE_NS="kserve"
KSERVE_TEST_NS="kserve-test"
KNATIVE_SERVING_NS="knative-serving"
CERT_MANAGER_NS="cert-manager"

INFERENCE_DOMAIN="example.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ──────────────────────────────────────────────
# URLs
# ──────────────────────────────────────────────
CERT_MANAGER_URL="https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"

KNATIVE_SERVING_CRD_URL="https://github.com/knative/serving/releases/download/knative-${KNATIVE_VERSION}/serving-crds.yaml"
KNATIVE_SERVING_CORE_URL="https://github.com/knative/serving/releases/download/knative-${KNATIVE_VERSION}/serving-core.yaml"

# Kourier is lighter than Istio — preferred for local Kind clusters
# Note: repo moved from knative/net-kourier to knative-extensions/net-kourier
KOURIER_URL="https://github.com/knative-extensions/net-kourier/releases/download/knative-${KOURIER_VERSION}/kourier.yaml"

KSERVE_URL="https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
KSERVE_RUNTIMES_URL="https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml"

# ──────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
step()    { echo -e "\n${CYAN}── $* ──${NC}"; }

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
wait_for_namespace_pods() {
  local ns="$1"
  local timeout="${2:-300}"
  local interval=10
  local elapsed=0

  info "Waiting for pods in '${ns}' to be ready (timeout: ${timeout}s)..."
  while [[ "$elapsed" -lt "$timeout" ]]; do
    local total pending
    total=$(kubectl get pods -n "${ns}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    pending=$(kubectl get pods -n "${ns}" --no-headers 2>/dev/null \
      | grep -cvE "Running|Completed|Succeeded" || true)

    if [[ "$total" -gt 0 && "$pending" -eq 0 ]]; then
      success "All ${total} pod(s) in '${ns}' are ready."
      return 0
    fi
    info "  ${pending}/${total} pod(s) pending... (${elapsed}s)"
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done

  error "Timeout waiting for pods in '${ns}'."
  kubectl get pods -n "${ns}" --no-headers 2>/dev/null | grep -vE "Running|Completed|Succeeded" || true
  return 1
}

apply_and_wait() {
  local url="$1"
  local description="$2"

  info "Applying: ${description}..."
  if ! kubectl apply -f "${url}"; then
    error "Failed to apply ${description}."
    return 1
  fi
  success "${description} applied."
}

# ──────────────────────────────────────────────
# Prerequisites
# ──────────────────────────────────────────────
check_prerequisites() {
  step "Checking prerequisites"

  for cmd in kubectl kind; do
    if ! command -v "$cmd" &>/dev/null; then
      error "'$cmd' not installed."
      exit 1
    fi
    success "$cmd found."
  done

  if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    error "Kind cluster '${CLUSTER_NAME}' not found. Run setup-kind-cluster.sh first."
    exit 1
  fi

  kubectl config use-context "${KUBECTL_CONTEXT}" &>/dev/null || {
    error "Cannot set kubectl context to '${KUBECTL_CONTEXT}'."
    exit 1
  }
  success "kubectl context: ${KUBECTL_CONTEXT}"
}

# ──────────────────────────────────────────────
# 1. cert-manager
# ──────────────────────────────────────────────
install_cert_manager() {
  step "Installing cert-manager ${CERT_MANAGER_VERSION}"

  if kubectl get namespace "${CERT_MANAGER_NS}" &>/dev/null; then
    local ready
    ready=$(kubectl get pods -n "${CERT_MANAGER_NS}" --no-headers 2>/dev/null \
      | grep -c "Running" || true)
    if [[ "$ready" -ge 3 ]]; then
      success "cert-manager already installed and running. Skipping."
      return 0
    fi
  fi

  apply_and_wait "${CERT_MANAGER_URL}" "cert-manager ${CERT_MANAGER_VERSION}"

  # cert-manager needs its webhook to be ready before other components can use it
  info "Waiting for cert-manager webhook to be ready..."
  kubectl wait --for=condition=available deployment/cert-manager-webhook \
    -n "${CERT_MANAGER_NS}" --timeout=180s

  wait_for_namespace_pods "${CERT_MANAGER_NS}" 180
}

# ──────────────────────────────────────────────
# 2. Knative Serving + Kourier networking
# ──────────────────────────────────────────────
install_knative_serving() {
  step "Installing Knative Serving ${KNATIVE_VERSION}"

  if kubectl get namespace "${KNATIVE_SERVING_NS}" &>/dev/null; then
    local ready
    ready=$(kubectl get pods -n "${KNATIVE_SERVING_NS}" --no-headers 2>/dev/null \
      | grep -c "Running" || true)
    if [[ "$ready" -ge 3 ]]; then
      success "Knative Serving already installed. Skipping."
      return 0
    fi
  fi

  # CRDs first
  apply_and_wait "${KNATIVE_SERVING_CRD_URL}" "Knative Serving CRDs"

  # Core components
  apply_and_wait "${KNATIVE_SERVING_CORE_URL}" "Knative Serving Core"

  wait_for_namespace_pods "${KNATIVE_SERVING_NS}" 300

  step "Installing Kourier networking layer"

  apply_and_wait "${KOURIER_URL}" "Kourier ingress"

  # Configure Knative to use Kourier
  info "Setting Kourier as the default ingress..."
  kubectl patch configmap/config-network \
    --namespace "${KNATIVE_SERVING_NS}" \
    --type merge \
    -p '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'

  success "Kourier configured as Knative networking layer."

  # Configure DNS (use sslip.io for local development)
  info "Configuring DNS for local development..."
  kubectl patch configmap/config-domain \
    --namespace "${KNATIVE_SERVING_NS}" \
    --type merge \
    -p "{\"data\":{\"${INFERENCE_DOMAIN}\":\"\"}}"

  # Disable tag-to-digest resolution (Kind can't reach external registries reliably)
  info "Configuring deployment settings for Kind..."
  kubectl patch configmap/config-deployment \
    --namespace "${KNATIVE_SERVING_NS}" \
    --type merge \
    -p '{"data":{"registries-skipping-tag-resolving":"kind.local,ko.local,dev.local,localhost:5000"}}'

  wait_for_namespace_pods "kourier-system" 120
  success "Knative Serving with Kourier is ready."
}

# ──────────────────────────────────────────────
# 3. KServe
# ──────────────────────────────────────────────
install_kserve() {
  step "Installing KServe ${KSERVE_VERSION}"

  if kubectl get namespace "${KSERVE_NS}" &>/dev/null; then
    local ready
    ready=$(kubectl get pods -n "${KSERVE_NS}" --no-headers 2>/dev/null \
      | grep -c "Running" || true)
    if [[ "$ready" -ge 1 ]]; then
      success "KServe already installed. Skipping."
      return 0
    fi
  fi

  # Install KServe CRDs + controller
  apply_and_wait "${KSERVE_URL}" "KServe CRDs and controller"

  # Install default serving runtimes
  apply_and_wait "${KSERVE_RUNTIMES_URL}" "KServe ClusterServingRuntimes"

  wait_for_namespace_pods "${KSERVE_NS}" 300
  success "KServe controller is running."
}

# ──────────────────────────────────────────────
# 4. Configure KServe for local development
# ──────────────────────────────────────────────
configure_kserve() {
  step "Configuring KServe for local Kind development"

  # Configure inferenceservice-config
  info "Setting inference domain, scale-to-zero, and GPU affinity..."

  # Patch the inferenceservice-config ConfigMap in kserve namespace
  # This sets scale-to-zero, domain, and GPU affinity labels
  kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: inferenceservice-config
  namespace: kserve
data:
  # Scale-to-zero: allow minReplicas=0 for InferenceServices
  autoscaler: |-
    {
      "class": "kpa.autoscaling.knative.dev",
      "minReplicas": 0,
      "targetUtilizationPercentage": "70",
      "enableScaleToZero": true,
      "scaleToZeroGracePeriod": "30s"
    }
  # Deploy settings
  deploy: |-
    {
      "defaultDeploymentMode": "Serverless"
    }
  # Ingress settings
  ingress: |-
    {
      "ingressGateway": "knative-serving/knative-ingress-gateway",
      "ingressService": "kourier",
      "localGateway": "knative-serving/knative-local-gateway",
      "localGatewayService": "kourier-internal",
      "ingressDomain": "example.com",
      "ingressClassName": "kourier.ingress.networking.knative.dev",
      "urlScheme": "http"
    }
  # Node affinity for GPU workloads (labels for future use)
  nodeSelector: |-
    {
      "gpu": {
        "nodeSelector": {
          "accelerator": "nvidia-gpu"
        },
        "tolerations": [
          {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule"
          }
        ]
      }
    }
EOF

  success "KServe configured:"
  success "  - Domain: ${INFERENCE_DOMAIN}"
  success "  - Scale-to-zero: enabled (minReplicas: 0, grace: 30s)"
  success "  - GPU affinity labels: accelerator=nvidia-gpu (for future use)"
  success "  - Networking: Kourier"

  # Restart KServe controller to pick up config changes
  info "Restarting KServe controller to apply configuration..."
  kubectl rollout restart deployment kserve-controller-manager -n "${KSERVE_NS}" 2>/dev/null || true
  kubectl rollout status deployment kserve-controller-manager -n "${KSERVE_NS}" --timeout=120s 2>/dev/null || true

  success "KServe configuration applied."
}

# ──────────────────────────────────────────────
# Verify installation
# ──────────────────────────────────────────────
verify_installation() {
  step "Verifying KServe installation"

  local all_ok=true

  echo ""
  info "cert-manager:"
  for deploy in cert-manager cert-manager-cainjector cert-manager-webhook; do
    local replicas
    replicas=$(kubectl get deployment "${deploy}" -n "${CERT_MANAGER_NS}" \
      -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [[ "${replicas}" -gt 0 ]]; then
      success "  ${deploy} (${replicas} replica(s))"
    else
      warn "  ${deploy} NOT READY"
      all_ok=false
    fi
  done

  echo ""
  info "Knative Serving:"
  for deploy in activator autoscaler controller webhook; do
    local replicas
    replicas=$(kubectl get deployment "${deploy}" -n "${KNATIVE_SERVING_NS}" \
      -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [[ "${replicas}" -gt 0 ]]; then
      success "  ${deploy} (${replicas} replica(s))"
    else
      warn "  ${deploy} NOT READY"
      all_ok=false
    fi
  done

  echo ""
  info "Kourier:"
  local kourier_ready
  kourier_ready=$(kubectl get deployment 3scale-kourier-gateway -n "kourier-system" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  if [[ "${kourier_ready}" -gt 0 ]]; then
    success "  3scale-kourier-gateway (${kourier_ready} replica(s))"
  else
    warn "  3scale-kourier-gateway NOT READY"
    all_ok=false
  fi

  echo ""
  info "KServe:"
  local kserve_ready
  kserve_ready=$(kubectl get deployment kserve-controller-manager -n "${KSERVE_NS}" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  if [[ "${kserve_ready}" -gt 0 ]]; then
    success "  kserve-controller-manager (${kserve_ready} replica(s))"
  else
    warn "  kserve-controller-manager NOT READY"
    all_ok=false
  fi

  echo ""
  info "KServe CRDs:"
  for crd in inferenceservices.serving.kserve.io clusterservingruntimes.serving.kserve.io servingruntimes.serving.kserve.io; do
    if kubectl get crd "${crd}" &>/dev/null; then
      success "  ${crd}"
    else
      warn "  ${crd} MISSING"
      all_ok=false
    fi
  done

  echo ""
  info "KServe config (inferenceservice-config):"
  if kubectl get configmap inferenceservice-config -n "${KSERVE_NS}" &>/dev/null; then
    local scale_to_zero
    scale_to_zero=$(kubectl get configmap inferenceservice-config -n "${KSERVE_NS}" \
      -o jsonpath='{.data.autoscaler}' 2>/dev/null | grep -o '"enableScaleToZero": *[a-z]*' || echo "not set")
    success "  Scale-to-zero: ${scale_to_zero}"
  else
    warn "  inferenceservice-config not found"
  fi

  echo ""
  if [[ "${all_ok}" == true ]]; then
    success "All KServe components verified successfully."
  else
    warn "Some components are not ready. They may still be starting."
    warn "Re-run: $0 verify"
  fi
}

# ──────────────────────────────────────────────
# Deploy test InferenceService
# ──────────────────────────────────────────────
deploy_test() {
  step "Deploying test InferenceService"

  local test_yaml="${PROJECT_ROOT}/configs/test-inference-service.yaml"

  if [[ ! -f "${test_yaml}" ]]; then
    error "Test manifest not found: ${test_yaml}"
    exit 1
  fi

  # Create test namespace
  kubectl create namespace "${KSERVE_TEST_NS}" 2>/dev/null || true

  info "Applying test InferenceService..."
  kubectl apply -f "${test_yaml}" -n "${KSERVE_TEST_NS}"

  info "Waiting for InferenceService to be ready (this may take a few minutes)..."
  local retries=30
  local delay=10
  for (( i=1; i<=retries; i++ )); do
    local status
    status=$(kubectl get inferenceservice sklearn-iris -n "${KSERVE_TEST_NS}" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")

    if [[ "${status}" == "True" ]]; then
      success "Test InferenceService 'sklearn-iris' is Ready!"
      echo ""
      kubectl get inferenceservice sklearn-iris -n "${KSERVE_TEST_NS}"
      echo ""

      # Show how to test it
      local url
      url=$(kubectl get inferenceservice sklearn-iris -n "${KSERVE_TEST_NS}" \
        -o jsonpath='{.status.url}' 2>/dev/null)

      echo ""
      info "To test the inference endpoint:"
      echo ""
      echo "  # Port-forward Kourier gateway"
      echo "  kubectl port-forward -n kourier-system svc/kourier 8081:80 &"
      echo ""
      echo "  # Send a prediction request"
      echo "  curl -v \\"
      echo "    -H \"Host: sklearn-iris.${KSERVE_TEST_NS}.${INFERENCE_DOMAIN}\" \\"
      echo "    -H \"Content-Type: application/json\" \\"
      echo "    -d '{\"instances\": [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]}' \\"
      echo "    http://localhost:8081/v1/models/sklearn-iris:predict"
      echo ""
      return 0
    fi

    info "  Attempt ${i}/${retries}: status='${status}' (waiting ${delay}s)..."
    sleep "${delay}"
  done

  warn "InferenceService not ready yet. Check status:"
  echo "  kubectl get inferenceservice sklearn-iris -n ${KSERVE_TEST_NS} -o yaml"
  echo "  kubectl get pods -n ${KSERVE_TEST_NS}"
}

# ──────────────────────────────────────────────
# Uninstall
# ──────────────────────────────────────────────
uninstall() {
  step "Uninstalling KServe and dependencies"

  warn "This will remove KServe, Knative Serving, Kourier, and cert-manager."
  read -rp "Continue? [y/N]: " confirm
  if [[ "${confirm,,}" != "y" ]]; then
    info "Aborted."
    exit 0
  fi

  # Test namespace
  info "Removing test namespace..."
  kubectl delete namespace "${KSERVE_TEST_NS}" --timeout=60s 2>/dev/null || true

  # KServe
  info "Removing KServe..."
  kubectl delete -f "${KSERVE_RUNTIMES_URL}" 2>/dev/null || true
  kubectl delete -f "${KSERVE_URL}" 2>/dev/null || true
  kubectl delete namespace "${KSERVE_NS}" --timeout=120s 2>/dev/null || true

  # Kourier
  info "Removing Kourier..."
  kubectl delete -f "${KOURIER_URL}" 2>/dev/null || true
  kubectl delete namespace "kourier-system" --timeout=60s 2>/dev/null || true

  # Knative Serving
  info "Removing Knative Serving..."
  kubectl delete -f "${KNATIVE_SERVING_CORE_URL}" 2>/dev/null || true
  kubectl delete -f "${KNATIVE_SERVING_CRD_URL}" 2>/dev/null || true
  kubectl delete namespace "${KNATIVE_SERVING_NS}" --timeout=120s 2>/dev/null || true

  # cert-manager
  info "Removing cert-manager..."
  kubectl delete -f "${CERT_MANAGER_URL}" 2>/dev/null || true
  kubectl delete namespace "${CERT_MANAGER_NS}" --timeout=120s 2>/dev/null || true

  # Clean up KServe CRDs
  info "Cleaning up CRDs..."
  kubectl get crds -o name 2>/dev/null \
    | grep -E "kserve|knative|certmanager|cert-manager|serving\.knative" \
    | while read -r crd; do
        kubectl delete "$crd" --timeout=60s 2>/dev/null || true
      done

  success "Uninstall complete."
}

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print_summary() {
  echo ""
  echo -e "${GREEN}=========================================="
  echo "  KServe Installation Complete!"
  echo -e "==========================================${NC}"
  echo ""
  echo "  Versions:"
  echo "    cert-manager:    ${CERT_MANAGER_VERSION}"
  echo "    Knative Serving: ${KNATIVE_VERSION}"
  echo "    Kourier:         ${KOURIER_VERSION}"
  echo "    KServe:          ${KSERVE_VERSION}"
  echo ""
  echo "  Configuration:"
  echo "    Inference domain: ${INFERENCE_DOMAIN}"
  echo "    Scale-to-zero:    enabled (minReplicas: 0)"
  echo "    GPU affinity:     accelerator=nvidia-gpu (labels ready)"
  echo ""
  echo "  Namespaces:"
  echo "    ${CERT_MANAGER_NS}     - cert-manager"
  echo "    ${KNATIVE_SERVING_NS} - Knative Serving"
  echo "    kourier-system     - Kourier ingress"
  echo "    ${KSERVE_NS}           - KServe controller"
  echo ""
  echo -e "${BLUE}Next steps:${NC}"
  echo "  1. Deploy a test InferenceService:"
  echo "       $0 test"
  echo ""
  echo "  2. Or deploy your own model:"
  echo "       kubectl apply -f configs/test-inference-service.yaml -n ${KSERVE_TEST_NS}"
  echo ""
  echo "  3. Port-forward Kourier to access inference endpoints:"
  echo "       kubectl port-forward -n kourier-system svc/kourier 8081:80"
  echo ""
  echo "  4. Verify installation anytime:"
  echo "       $0 verify"
  echo ""
  echo "  5. Uninstall:"
  echo "       $0 uninstall"
  echo ""
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
main() {
  local action="${1:-install}"

  echo ""
  echo "=========================================="
  echo "  KServe Installer for Kind"
  echo "  KServe ${KSERVE_VERSION} / Knative ${KNATIVE_VERSION}"
  echo "=========================================="
  echo ""

  case "${action}" in
    install)
      check_prerequisites
      install_cert_manager
      install_knative_serving
      install_kserve
      configure_kserve
      verify_installation
      print_summary
      ;;
    verify)
      check_prerequisites
      verify_installation
      ;;
    test)
      check_prerequisites
      deploy_test
      ;;
    uninstall)
      check_prerequisites
      uninstall
      ;;
    *)
      echo "Usage: $0 {install|verify|test|uninstall}"
      exit 1
      ;;
  esac
}

main "$@"
