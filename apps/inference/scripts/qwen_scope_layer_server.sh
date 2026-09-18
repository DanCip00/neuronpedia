#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
DEFAULT_REPO="$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)"
DEFAULT_LOG_DIR="/tmp/neuronpedia-qwen-scope"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="5002"
DEFAULT_CUDA_DEVICES="auto"
DEFAULT_TOKEN_LIMIT="8192"
DEFAULT_ACTIVATION_BATCH_SIZE="8"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION="0.70"
DEFAULT_TIMEOUT_SECONDS="1800"
DEFAULT_STOP_TIMEOUT_SECONDS="60"
DEFAULT_PROBE_TIMEOUT_SECONDS="300"
MODEL_ID="Qwen/Qwen3.5-27B"
SAELENS_RELEASE="qwen-scope-3.5-27b-w80k-l50"
MAX_LAYER=63

usage() {
  cat <<EOF
Qwen-Scope Layer Sweep Server

Start one Neuronpedia inference server for one Qwen-Scope SAE layer, wait until
the model and SAE are loaded and the SAE answers a probe request, then print a
JSON line with the PID, process group, URL, pidfile, log file, and stop command.

Commands:
  ${SCRIPT_NAME} start --layer N [options]
  ${SCRIPT_NAME} stop --pidfile PATH [--timeout-seconds N]
  ${SCRIPT_NAME} status --pidfile PATH
  ${SCRIPT_NAME} list-layers
  ${SCRIPT_NAME} pick-gpu [--vllm-gpu-memory-utilization F]

Common start options:
  --layer N                         Layer number, 0..${MAX_LAYER}. Example: 31 -> layer31.
  --repo PATH                       Neuronpedia repo. Default: ${DEFAULT_REPO}
  --port PORT                       Bind port. Default: ${DEFAULT_PORT}
  --cuda-devices IDS|auto           CUDA_VISIBLE_DEVICES, or "auto" to pick the emptiest GPU
                                    with room for the requested vLLM memory fraction.
                                    Default: ${DEFAULT_CUDA_DEVICES}
  --token-limit N                   Inference token limit. Default: ${DEFAULT_TOKEN_LIMIT}
  --activation-batch-size N         Activation batch size. Default: ${DEFAULT_ACTIVATION_BATCH_SIZE}
  --vllm-gpu-memory-utilization F   vLLM GPU memory fraction. Default: ${DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}
  --timeout-seconds N               Startup readiness timeout. Default: ${DEFAULT_TIMEOUT_SECONDS}
  --probe-timeout-seconds N         Timeout for the post-startup SAE probe request. Default: ${DEFAULT_PROBE_TIMEOUT_SECONDS}
  --no-probe                        Skip the /v1/activation/source probe after /health is ready.
  --log-dir PATH                    Log/pidfile directory. Default: ${DEFAULT_LOG_DIR}

Other start options:
  --host HOST                       Bind host. Default: ${DEFAULT_HOST}
  --extra-arg ARG                   Extra argument passed to start.py. Repeatable.

Stop options:
  --pidfile PATH                    Pidfile produced by start.
  --timeout-seconds N               Graceful stop timeout before KILL. Default: ${DEFAULT_STOP_TIMEOUT_SECONDS}

Readiness:
  /health answers 500 "Server not initialized" while the model loads and keeps
  answering that, with the error appended, if loading failed. start treats the
  former as "keep waiting" and the latter as fatal: it stops the process and
  exits non-zero instead of waiting out the timeout. Once /health is 200, start
  sends one tiny /v1/activation/source request naming the layer's source, so a
  server that came up without that SAE (wrong pattern, failed download) is
  refused here rather than by the first real client request.

Authentication:
  If NEURONPEDIA_SECRET or SECRET is set in the environment, the probe sends it
  as X-SECRET-KEY. /health never needs it.

Examples:
  Start layer31 on whichever GPU has room:
    START_JSON=\$(${SCRIPT_NAME} start --layer 31 --port 5002)

  Read fields from the success JSON:
    BASE_URL=\$(jq -r ".base_url" <<<"\$START_JSON")
    PIDFILE=\$(jq -r ".pidfile" <<<"\$START_JSON")
    SOURCE=\$(jq -r ".source" <<<"\$START_JSON")

  Stop that server (returns once the process group is gone and the port is free):
    ${SCRIPT_NAME} stop --pidfile "\$PIDFILE"

  Sweep layers, one server at a time:
    for layer in 32 36 40 44 48; do
      START_JSON=\$(${SCRIPT_NAME} start --layer "\$layer" --port 5002)
      BASE_URL=\$(jq -r ".base_url" <<<"\$START_JSON")
      PIDFILE=\$(jq -r ".pidfile" <<<"\$START_JSON")
      SOURCE=\$(jq -r ".source" <<<"\$START_JSON")

      python run_extraction.py --base-url "\$BASE_URL" --source "\$SOURCE"
      ${SCRIPT_NAME} stop --pidfile "\$PIDFILE"
    done

What start launches:
  model:             ${MODEL_ID}
  SAELens release:   ${SAELENS_RELEASE}
  source pattern:    ^layerN\$
  SAE sets:          []
  dtype:             model=bfloat16, sae=bfloat16
  backend:           vLLM
  Jacobian lens:     skipped

Memory note:
  One Qwen-Scope SAE is about 1.6 GiB at bf16 before overhead. All 64 layers are
  roughly 100 GiB of SAE weights before overhead, plus Qwen 27B and vLLM memory.
  One layer per process is the safe default. vLLM refuses to start unless the
  GPU has at least (fraction x total) memory free, which is what "auto" checks.

More detail:
  ${DEFAULT_REPO}/apps/inference/docs/qwen_scope_layer_sweep.md
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
  for layer in $(seq 0 "$MAX_LAYER"); do
    echo "layer${layer}"
  done
}

validate_layer() {
  local layer="$1"
  [[ "$layer" =~ ^[0-9]+$ ]] || die "--layer must be an integer 0 through ${MAX_LAYER}, got '$layer'"
  (( layer >= 0 && layer <= MAX_LAYER )) || die "--layer must be between 0 and ${MAX_LAYER}, got '$layer'"
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

validate_fraction() {
  local name="$1"
  local value="$2"
  python3 - "$name" "$value" <<'PY' || exit 1
import sys

name, raw = sys.argv[1], sys.argv[2]
try:
    value = float(raw)
except ValueError:
    print(f"error: {name} must be a number in (0, 1], got '{raw}'", file=sys.stderr)
    sys.exit(1)
if not 0 < value <= 1:
    print(f"error: {name} must be in (0, 1], got '{raw}'", file=sys.stderr)
    sys.exit(1)
PY
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

# Prints one of: ready | initializing | down | failed:<message>
health_state() {
  local base_url="$1"
  python3 - "$base_url" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        status, body = response.status, response.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    status, body = exc.code, exc.read().decode("utf-8", "replace")
except (OSError, ValueError):
    print("down")
    sys.exit(0)

if status == 200:
    try:
        print("ready" if json.loads(body) == {"status": "healthy"} else "initializing")
    except ValueError:
        print("initializing")
    sys.exit(0)

# CudaHealthMiddleware answers 500 {"error": "Server not initialized.[ Initialization error: ...]"}
# for every path until the model is loaded. Only the variant carrying an error is terminal.
try:
    message = str(json.loads(body).get("error", ""))
except ValueError:
    message = body
marker = "Initialization error:"
if marker in message:
    print("failed:" + message.split(marker, 1)[1].strip().replace("\n", " "))
else:
    print("initializing")
PY
}

# Exit 0 when /v1/activation/source answers for the source; otherwise print the failure and exit 1.
source_probe() {
  local base_url="$1"
  local source="$2"
  local timeout_seconds="$3"
  python3 - "$base_url" "$source" "$MODEL_ID" "$timeout_seconds" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

base_url, source, model_id, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
payload = {
    "model": model_id,
    "source": source,
    "promptTokenIds": [[1, 2, 3]],
    "activationPositions": [[-1]],
    "insertion": {"bos": "never", "eos": "never", "prefixTokenIds": [], "suffixTokenIds": []},
}
headers = {"Content-Type": "application/json"}
secret = os.environ.get("NEURONPEDIA_SECRET") or os.environ.get("SECRET")
if secret:
    headers["X-SECRET-KEY"] = secret
request = urllib.request.Request(
    base_url.rstrip("/") + "/v1/activation/source",
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}")
    sys.exit(1)
except (OSError, ValueError) as exc:
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
results = body.get("results") if isinstance(body, dict) else None
if not isinstance(results, list) or len(results) != 1:
    print(f"unexpected response shape: {json.dumps(body)[:500]}")
    sys.exit(1)
sys.exit(0)
PY
}

# Print the index of the GPU with the most free memory that still has room for the vLLM fraction.
pick_gpu() {
  local utilization="$1"
  command -v nvidia-smi >/dev/null 2>&1 || die "--cuda-devices auto needs nvidia-smi on PATH"
  local table
  table="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)" \
    || die "nvidia-smi failed; pass --cuda-devices explicitly"
  python3 - "$utilization" "$table" <<'PY'
import sys

utilization = float(sys.argv[1])
rows = []
for line in sys.argv[2].strip().splitlines():
    index, used, total = (part.strip() for part in line.split(","))
    rows.append((int(index), float(used), float(total)))
if not rows:
    print("error: nvidia-smi reported no GPUs", file=sys.stderr)
    sys.exit(1)
# vLLM refuses to start unless free memory >= utilization * total; keep 1 GiB of slack for the
# CUDA context and whatever else is about to be allocated on the device.
fits = [row for row in rows if row[2] - row[1] >= utilization * row[2] + 1024]
if not fits:
    usage = ", ".join(f"GPU {index}: {used:.0f}/{total:.0f} MiB used" for index, used, total in rows)
    print(
        f"error: no GPU has {utilization:.2f} x total memory free for vLLM ({usage}); "
        "lower --vllm-gpu-memory-utilization or free a GPU",
        file=sys.stderr,
    )
    sys.exit(1)
print(min(fits, key=lambda row: (row[1], row[0]))[0])
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

process_pgid() {
  local pid="$1"
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
}

write_pidfile() {
  local pidfile="$1"
  local tmpfile="${pidfile}.tmp"
  cat >"$tmpfile" <<EOF
PID=$PID
PGID=$PGID
LAYER=$LAYER
PORT=$PORT
HOST=$HOST
BASE_URL=$BASE_URL
LOG_FILE=$LOG_FILE
CUDA_DEVICES=$CUDA_DEVICES
STARTED_AT=$STARTED_AT
EOF
  mv "$tmpfile" "$pidfile"
}

# Wait until nothing in the group is left and the port has closed, so the next start
# on the same GPU/port does not race the dying vLLM workers for memory or the socket.
wait_for_release() {
  local pgid="$1"
  local pid="$2"
  local host="$3"
  local port="$4"
  local timeout_seconds="$5"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! group_alive "$pgid" && ! process_alive "$pid" && ! port_is_open "$host" "$port"; then
      return 0
    fi
    sleep 1
  done
  return 1
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
  local probe_timeout_seconds="$DEFAULT_PROBE_TIMEOUT_SECONDS"
  local probe=1
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
      --probe-timeout-seconds) probe_timeout_seconds="${2:-}"; shift 2 ;;
      --no-probe) probe=0; shift ;;
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
  validate_positive_int "--probe-timeout-seconds" "$probe_timeout_seconds"
  validate_fraction "--vllm-gpu-memory-utilization" "$vllm_gpu_memory_utilization"
  [[ -d "$repo/apps/inference" ]] || die "repo does not look like Neuronpedia: $repo"

  if port_is_open "$host" "$port"; then
    die "port ${host}:${port} is already accepting connections; stop that server or pass --port"
  fi

  if [[ "$cuda_devices" == "auto" ]]; then
    cuda_devices="$(pick_gpu "$vllm_gpu_memory_utilization")"
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

  # Append rather than truncate: a sweep that restarts the same layer after a failure
  # would otherwise destroy the evidence of what went wrong the first time.
  {
    echo "===== ${SCRIPT_NAME} start ${source} port=${port} cuda=${cuda_devices} at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
  } >>"$log_file"

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
  ) >>"$log_file" 2>&1 &

  PID=$!
  # setsid makes the child a session leader, so its PGID is normally its own PID -- but read it
  # back rather than assume it, since stop kills by group and a wrong group kills nothing.
  PGID=""
  local attempt
  for attempt in $(seq 1 50); do
    PGID="$(process_pgid "$PID")"
    if [[ -n "$PGID" && "$PGID" != "$(process_pgid $$)" ]]; then
      break
    fi
    PGID=""
    sleep 0.1
  done
  if [[ -z "$PGID" ]]; then
    if process_alive "$PID"; then
      PGID="$PID"
    else
      die "server process exited immediately; see log: $log_file"
    fi
  fi
  LAYER="$source"
  PORT="$port"
  HOST="$host"
  BASE_URL="$base_url"
  LOG_FILE="$log_file"
  CUDA_DEVICES="$cuda_devices"
  STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  write_pidfile "$pidfile"

  local deadline=$((SECONDS + timeout_seconds))
  local state=""
  while (( SECONDS < deadline )); do
    if ! process_alive "$PID"; then
      rm -f "$pidfile"
      die "server process exited before readiness; see log: $log_file"
    fi
    state="$(health_state "$base_url")"
    case "$state" in
      ready) break ;;
      failed:*)
        "$0" stop --pidfile "$pidfile" --timeout-seconds 30 >/dev/null || true
        die "server failed to load ${source}: ${state#failed:}; see log: $log_file"
        ;;
    esac
    sleep 2
  done
  if [[ "$state" != "ready" ]]; then
    "$0" stop --pidfile "$pidfile" --timeout-seconds 30 >/dev/null || true
    die "timed out after ${timeout_seconds}s waiting for ${base_url}/health; see log: $log_file"
  fi

  if (( probe )); then
    local probe_error
    if ! probe_error="$(source_probe "$base_url" "$source" "$probe_timeout_seconds")"; then
      "$0" stop --pidfile "$pidfile" --timeout-seconds 30 >/dev/null || true
      die "server is up but ${source} did not answer /v1/activation/source: ${probe_error}; see log: $log_file"
    fi
  fi

  printf '{"status":"ready","pid":%s,"pgid":%s,"source":"%s","source_set":"%s","model":"%s","base_url":"%s","cuda_devices":"%s","pidfile":"%s","log":"%s","probed":%s,"stop_command":"%s stop --pidfile %s"}\n' \
    "$PID" \
    "$PGID" \
    "$(json_escape "$source")" \
    "$(json_escape "$SAELENS_RELEASE")" \
    "$(json_escape "$MODEL_ID")" \
    "$(json_escape "$base_url")" \
    "$(json_escape "$cuda_devices")" \
    "$(json_escape "$pidfile")" \
    "$(json_escape "$log_file")" \
    "$([[ $probe -eq 1 ]] && echo true || echo false)" \
    "$(json_escape "$0")" \
    "$(json_escape "$pidfile")"
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
    local health
    health="$(health_state "$BASE_URL")"
    printf '{"status":"running","health":"%s","pid":%s,"pgid":%s,"source":"%s","base_url":"%s","cuda_devices":"%s","pidfile":"%s","log":"%s"}\n' \
      "$(json_escape "$health")" "$PID" "$PGID" "$(json_escape "$LAYER")" "$(json_escape "$BASE_URL")" \
      "$(json_escape "${CUDA_DEVICES:-}")" "$(json_escape "$pidfile")" "$(json_escape "$LOG_FILE")"
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
  local host="${HOST:-$DEFAULT_HOST}"
  local port="${PORT:-$DEFAULT_PORT}"

  if ! process_alive "$PID" && ! group_alive "$PGID"; then
    rm -f "$pidfile"
    json_line "stopped" "process was already gone"
    return 0
  fi

  kill -TERM "-$PGID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  if wait_for_release "$PGID" "$PID" "$host" "$port" "$timeout_seconds"; then
    rm -f "$pidfile"
    json_line "stopped"
    return 0
  fi

  kill -KILL "-$PGID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
  if wait_for_release "$PGID" "$PID" "$host" "$port" 30; then
    rm -f "$pidfile"
    json_line "killed" "process group did not stop after TERM"
    return 0
  fi
  die "process group ${PGID} is still alive after KILL, or ${host}:${port} is still open; see $LOG_FILE"
}

cmd_pick_gpu() {
  local utilization="$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --vllm-gpu-memory-utilization) utilization="${2:-}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown pick-gpu option: $1" ;;
    esac
  done
  validate_fraction "--vllm-gpu-memory-utilization" "$utilization"
  pick_gpu "$utilization"
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
    pick-gpu) cmd_pick_gpu "$@" ;;
    -h|--help|help) usage ;;
    *) die "unknown command: $command" ;;
  esac
}

main "$@"
