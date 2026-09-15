#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
DEFAULT_REPO="$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)"
DEFAULT_LOG_DIR="/tmp/neuronpedia-qwen-scope"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="5002"
DEFAULT_CUDA_DEVICES="0"
DEFAULT_TOKEN_LIMIT="8192"
DEFAULT_ACTIVATION_BATCH_SIZE="8"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.70"
DEFAULT_TIMEOUT_SECONDS="1800"
DEFAULT_STOP_TIMEOUT_SECONDS="30"
MODEL_ID="Qwen/Qwen3.5-27B"
SAELENS_RELEASE="qwen-scope-3.5-27b-w80k-l50"

usage() {
  cat <<'EOF'
Qwen-Scope Layer Sweep Server

Start one Neuronpedia inference server for one Qwen-Scope SAE layer, wait until
/health is ready, then print a JSON line with the PID, process group, URL,
pidfile, log file, and stop command.

Commands:
  qwen-scope-layer-server start --layer N [options]
  qwen-scope-layer-server stop --pidfile PATH [--timeout-seconds N]
  qwen-scope-layer-server status --pidfile PATH
  qwen-scope-layer-server list-layers

Common start options:
  --layer N                         Layer number, 0..63. Example: 31 -> layer31.
  --repo PATH                       Neuronpedia repo. Default: /home/daniele.cipollone/neuronpedia
  --port PORT                       Bind port. Default: 5002
  --cuda-devices IDS                CUDA_VISIBLE_DEVICES. Default: 0
  --token-limit N                   Inference token limit. Default: 8192
  --activation-batch-size N         Activation batch size. Default: 8
  --vllm-gpu-memory-utilization F   vLLM GPU memory fraction. Default: 0.70
  --timeout-seconds N               Startup readiness timeout. Default: 1800
  --log-dir PATH                    Log/pidfile directory. Default: /tmp/neuronpedia-qwen-scope

Other start options:
  --host HOST                       Bind host. Default: 127.0.0.1
  --extra-arg ARG                   Extra argument passed to start.py. Repeatable.

Stop options:
  --pidfile PATH                    Pidfile produced by start.
  --timeout-seconds N               Graceful stop timeout. Default: 30

Examples:
  Start layer31:
    START_JSON=$(qwen-scope-layer-server start --layer 31 --cuda-devices 0 --port 5002)

  Read fields from the success JSON:
    BASE_URL=$(jq -r ".base_url" <<<"$START_JSON")
    PIDFILE=$(jq -r ".pidfile" <<<"$START_JSON")
    SOURCE=$(jq -r ".source" <<<"$START_JSON")

  Stop that server:
    qwen-scope-layer-server stop --pidfile "$PIDFILE"

  Sweep every layer safely, one server at a time:
    for layer in $(seq 0 63); do
      START_JSON=$(qwen-scope-layer-server start --layer "$layer" --cuda-devices 0 --port 5002)
      BASE_URL=$(jq -r ".base_url" <<<"$START_JSON")
      PIDFILE=$(jq -r ".pidfile" <<<"$START_JSON")
      SOURCE=$(jq -r ".source" <<<"$START_JSON")

      python run_extraction.py --base-url "$BASE_URL" --source "$SOURCE"
      qwen-scope-layer-server stop --pidfile "$PIDFILE"
    done

What start launches:
  model:             Qwen/Qwen3.5-27B
  SAELens release:   qwen-scope-3.5-27b-w80k-l50
  source pattern:    ^layerN$
  SAE sets:          []
  dtype:             model=bfloat16, sae=bfloat16
  backend:           vLLM
  Jacobian lens:     skipped

Activation-source payload tip:
  For positive/negative branches, send activationPositions: [[-1], [-1]].
  Qwen still processes the full context; activationPositions only reduces SAE
  encode work and returned features.

Layers:
  Available layers are layer0 through layer63.
  Run qwen-scope-layer-server list-layers to print one layer per line.

Memory note:
  One Qwen-Scope SAE is about 1.6 GiB at bf16 before overhead. All 64 layers are
  roughly 100 GiB of SAE weights before overhead, plus Qwen 27B and vLLM memory.
  One layer per process is the safe default.

More detail:
  /home/daniele.cipollone/neuronpedia/apps/inference/docs/qwen_scope_layer_sweep.md
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

json_line() {
  local status="$1"
  local message="${2:-}"
  printf '{"status":"%s"' "$(json_escape "$status")"
  if [[ -n "$message" ]]; then
    printf ',"message":"%s"' "$(json_escape "$message")"
  fi
  printf '}\n'
}

list_layers() {
  local layer
  for layer in $(seq 0 63); do
    echo "layer${layer}"
  done
}

validate_layer() {
  local layer="$1"
  [[ "$layer" =~ ^[0-9]+$ ]] || die "--layer must be an integer 0 through 63, got '$layer'"
  (( layer >= 0 && layer <= 63 )) || die "--layer must be between 0 and 63, got '$layer'"
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || die "$name must be an integer >= 1, got '$value'"
}

validate_port() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ && "$value" -ge 1 && "$value" -le 65535 ]] || die "--port must be 1..65535, got '$value'"
}

port_is_open() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

http_health_ok() {
  local base_url="$1"
  python3 - "$base_url" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        if response.status != 200:
            sys.exit(1)
        payload = json.loads(response.read().decode("utf-8"))
except (OSError, ValueError, urllib.error.URLError):
    sys.exit(1)
sys.exit(0 if payload == {"status": "healthy"} else 1)
PY
}

load_pidfile() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || die "pidfile not found: $pidfile"
  # shellcheck disable=SC1090
  source "$pidfile"
  [[ -n "${PID:-}" && -n "${PGID:-}" ]] || die "pidfile is missing PID or PGID: $pidfile"
}

process_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

group_alive() {
  local pgid="$1"
  kill -0 "-$pgid" 2>/dev/null
}

write_pidfile() {
  local pidfile="$1"
  local tmpfile="${pidfile}.tmp"
  cat >"$tmpfile" <<EOF
PID=$PID
PGID=$PGID
LAYER=$LAYER
PORT=$PORT
BASE_URL=$BASE_URL
LOG_FILE=$LOG_FILE
STARTED_AT=$STARTED_AT
EOF
  mv "$tmpfile" "$pidfile"
}

cmd_start() {
  local repo="$DEFAULT_REPO"
  local host="$DEFAULT_HOST"
  local port="$DEFAULT_PORT"
  local cuda_devices="$DEFAULT_CUDA_DEVICES"
  local token_limit="$DEFAULT_TOKEN_LIMIT"
  local activation_batch_size="$DEFAULT_ACTIVATION_BATCH_SIZE"
  local vllm_gpu_memory_utilization="$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION"
  local timeout_seconds="$DEFAULT_TIMEOUT_SECONDS"
  local log_dir="$DEFAULT_LOG_DIR"
  local layer=""
  local extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --layer) layer="${2:-}"; shift 2 ;;
      --repo) repo="${2:-}"; shift 2 ;;
      --host) host="${2:-}"; shift 2 ;;
      --port) port="${2:-}"; shift 2 ;;
      --cuda-devices) cuda_devices="${2:-}"; shift 2 ;;
      --token-limit) token_limit="${2:-}"; shift 2 ;;
      --activation-batch-size) activation_batch_size="${2:-}"; shift 2 ;;
      --vllm-gpu-memory-utilization) vllm_gpu_memory_utilization="${2:-}"; shift 2 ;;
      --timeout-seconds) timeout_seconds="${2:-}"; shift 2 ;;
      --log-dir) log_dir="${2:-}"; shift 2 ;;
      --extra-arg) extra_args+=("${2:-}"); shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown start option: $1" ;;
    esac
  done

  [[ -n "$layer" ]] || die "start requires --layer N"
  validate_layer "$layer"
  validate_port "$port"
  validate_positive_int "--token-limit" "$token_limit"
  validate_positive_int "--activation-batch-size" "$activation_batch_size"
  validate_positive_int "--timeout-seconds" "$timeout_seconds"
  [[ -d "$repo/apps/inference" ]] || die "repo does not look like Neuronpedia: $repo"

  if port_is_open "$host" "$port"; then
    die "port ${host}:${port} is already accepting connections"
  fi

  mkdir -p "$log_dir"
  local source="layer${layer}"
  local pidfile="${log_dir}/${source}.port${port}.pid"
  local log_file="${log_dir}/${source}.port${port}.log"
  local base_url="http://${host}:${port}"

  if [[ -f "$pidfile" ]]; then
    # shellcheck disable=SC1090
    source "$pidfile"
    if [[ -n "${PID:-}" ]] && process_alive "$PID"; then
      die "pidfile already exists for live process $PID: $pidfile"
    fi
    rm -f "$pidfile"
  fi

  (
    cd "$repo/apps/inference"
    export CUDA_VISIBLE_DEVICES="$cuda_devices"
    export SAE_SETS="[]"
    export SAELENS_RELEASE="[\"${SAELENS_RELEASE}\"]"
    exec setsid uv run python start.py \
      --host "$host" \
      --port "$port" \
      --model_id "$MODEL_ID" \
      --model_dtype bfloat16 \
      --sae_dtype bfloat16 \
      --num-gpus 1 \
      --token_limit "$token_limit" \
      --activation_batch_size "$activation_batch_size" \
      --force-vllm \
      --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \
      --saelens_release "$SAELENS_RELEASE" \
      --include_sae "^${source}$" \
      --jlens_skip \
      "${extra_args[@]}"
  ) >"$log_file" 2>&1 &

  PID=$!
  PGID=$PID
  LAYER="$source"
  PORT="$port"
  BASE_URL="$base_url"
  LOG_FILE="$log_file"
  STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  write_pidfile "$pidfile"

  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! process_alive "$PID"; then
      rm -f "$pidfile"
      die "server process exited before readiness; see log: $log_file"
    fi
    if http_health_ok "$base_url"; then
      printf '{"status":"ready","pid":%s,"pgid":%s,"source":"%s","base_url":"%s","pidfile":"%s","log":"%s","stop_command":"%s stop --pidfile %s"}\n' \
        "$PID" \
        "$PGID" \
        "$(json_escape "$source")" \
        "$(json_escape "$base_url")" \
        "$(json_escape "$pidfile")" \
        "$(json_escape "$log_file")" \
        "$(json_escape "$0")" \
        "$(json_escape "$pidfile")"
      return 0
    fi
    sleep 2
  done

  "$0" stop --pidfile "$pidfile" --timeout-seconds 10 >/dev/null || true
  die "timed out after ${timeout_seconds}s waiting for ${base_url}/health; see log: $log_file"
}

cmd_status() {
  local pidfile=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pidfile) pidfile="${2:-}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown status option: $1" ;;
    esac
  done
  [[ -n "$pidfile" ]] || die "status requires --pidfile PATH"
  load_pidfile "$pidfile"

  if process_alive "$PID"; then
    printf '{"status":"running","pid":%s,"pgid":%s,"source":"%s","base_url":"%s","pidfile":"%s","log":"%s"}\n' \
      "$PID" "$PGID" "$(json_escape "$LAYER")" "$(json_escape "$BASE_URL")" "$(json_escape "$pidfile")" "$(json_escape "$LOG_FILE")"
  else
    printf '{"status":"stopped","pid":%s,"pgid":%s,"source":"%s","pidfile":"%s","log":"%s"}\n' \
      "$PID" "$PGID" "$(json_escape "$LAYER")" "$(json_escape "$pidfile")" "$(json_escape "$LOG_FILE")"
    return 1
  fi
}

cmd_stop() {
  local pidfile=""
  local timeout_seconds="$DEFAULT_STOP_TIMEOUT_SECONDS"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pidfile) pidfile="${2:-}"; shift 2 ;;
      --timeout-seconds) timeout_seconds="${2:-}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown stop option: $1" ;;
    esac
  done
  [[ -n "$pidfile" ]] || die "stop requires --pidfile PATH"
  validate_positive_int "--timeout-seconds" "$timeout_seconds"
  load_pidfile "$pidfile"

  if ! process_alive "$PID"; then
    rm -f "$pidfile"
    json_line "stopped" "process was already gone"
    return 0
  fi

  kill -TERM "-$PGID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! group_alive "$PGID" && ! process_alive "$PID"; then
      rm -f "$pidfile"
      json_line "stopped"
      return 0
    fi
    sleep 1
  done

  kill -KILL "-$PGID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
  sleep 1
  rm -f "$pidfile"
  json_line "killed" "process group did not stop after TERM"
}

main() {
  local command="${1:-}"
  [[ -n "$command" ]] || { usage; exit 1; }
  shift || true

  case "$command" in
    start) cmd_start "$@" ;;
    stop) cmd_stop "$@" ;;
    status) cmd_status "$@" ;;
    list-layers) list_layers ;;
    -h|--help|help) usage ;;
    *) die "unknown command: $command" ;;
  esac
}

main "$@"
