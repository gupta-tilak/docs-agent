#!/usr/bin/env bash
# ==========================================================================
# port-forward-milvus.sh – Forward Milvus ports to localhost
# ==========================================================================
# Usage:
#   ./scripts/port-forward-milvus.sh            # foreground (Ctrl-C to stop)
#   ./scripts/port-forward-milvus.sh --bg       # background (writes PID file)
#   ./scripts/port-forward-milvus.sh --stop     # stop background forward
# ==========================================================================
set -euo pipefail

NAMESPACE="kubeflow"
HELM_RELEASE="milvus"
GRPC_PORT="${MILVUS_GRPC_PORT:-19530}"
HTTP_PORT="${MILVUS_HTTP_PORT:-9091}"
PID_FILE="/tmp/milvus-port-forward.pid"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Find the Milvus service name
find_service() {
    local svc
    svc=$(kubectl get svc -n "$NAMESPACE" \
          -l "app.kubernetes.io/instance=${HELM_RELEASE},app.kubernetes.io/component=standalone" \
          -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -z "$svc" ]]; then
        svc=$(kubectl get svc -n "$NAMESPACE" \
              -l "app.kubernetes.io/instance=${HELM_RELEASE}" \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    fi

    if [[ -z "$svc" ]]; then
        svc=$(kubectl get svc -n "$NAMESPACE" \
              -l "app=milvus" \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    fi

    if [[ -z "$svc" ]]; then
        error "Could not find Milvus service in namespace '${NAMESPACE}'."
        error "Is Milvus deployed? Run: ./scripts/deploy-milvus.sh"
        exit 1
    fi

    echo "$svc"
}

# Health check after port-forward is up
health_check() {
    info "Checking Milvus health ..."
    sleep 2
    if curl -sf "http://localhost:${HTTP_PORT}/healthz" >/dev/null 2>&1; then
        info "Milvus is healthy (GET /healthz -> OK)"
    else
        warn "Health check returned non-200 (Milvus may still be starting)."
    fi
}

start_foreground() {
    local svc
    svc=$(find_service)
    info "Port-forwarding Milvus service '${svc}' ..."
    info "  gRPC : localhost:${GRPC_PORT} -> ${svc}:19530"
    info "  HTTP : localhost:${HTTP_PORT}  -> ${svc}:9091"
    info "Press Ctrl-C to stop."
    echo ""

    kubectl port-forward "svc/${svc}" \
        "${GRPC_PORT}:19530" "${HTTP_PORT}:9091" \
        -n "$NAMESPACE"
}

start_background() {
    local svc
    svc=$(find_service)

    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "Port-forward already running (PID $(cat "$PID_FILE"))."
        return
    fi

    info "Starting port-forward in background ..."
    kubectl port-forward "svc/${svc}" \
        "${GRPC_PORT}:19530" "${HTTP_PORT}:9091" \
        -n "$NAMESPACE" &>/dev/null &
    echo $! > "$PID_FILE"

    info "Port-forward running (PID $(cat "$PID_FILE"))."
    info "  gRPC : localhost:${GRPC_PORT}"
    info "  HTTP : localhost:${HTTP_PORT}"
    health_check
}

stop_background() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping port-forward (PID ${pid}) ..."
            kill "$pid"
        else
            warn "Process ${pid} is not running."
        fi
        rm -f "$PID_FILE"
    else
        warn "No PID file found. Nothing to stop."
    fi
}

case "${1:-}" in
    --bg|--background)
        start_background
        ;;
    --stop)
        stop_background
        ;;
    *)
        start_foreground
        ;;
esac
