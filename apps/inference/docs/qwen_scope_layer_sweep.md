# Qwen-Scope Layer Sweep Server

Use `qwen-scope-layer-server` to start one Neuronpedia inference server per Qwen-Scope SAE layer.
`start` waits until the model and the SAE are loaded, confirms the SAE answers a request, prints a
JSON success line, and gives you a pidfile; `stop` tears the server down and returns only once the
GPU and port are free for the next layer.

On this machine, `qwen-scope-layer-server` is installed as a symlink in `~/.local/bin`, pointing at:

```text
/home/daniele.cipollone/neuronpedia/apps/inference/scripts/qwen_scope_layer_server.sh
```

If you are on another machine, either create the same symlink or call the script by absolute path.

Available Qwen-Scope layers are `layer0` through `layer63`.

## Start One Layer

```bash
START_JSON=$(qwen-scope-layer-server start --layer 31 --port 5002)
echo "$START_JSON"
```

The success output is one JSON line:

```json
{
  "status": "ready",
  "pid": 12345,
  "pgid": 12345,
  "source": "layer31",
  "source_set": "qwen-scope-3.5-27b-w80k-l50",
  "model": "Qwen/Qwen3.5-27B",
  "base_url": "http://127.0.0.1:5002",
  "cuda_devices": "3",
  "pidfile": "/tmp/neuronpedia-qwen-scope/layer31.port5002.pid",
  "log": "/tmp/neuronpedia-qwen-scope/layer31.port5002.log",
  "probed": true,
  "stop_command": "qwen-scope-layer-server stop --pidfile /tmp/neuronpedia-qwen-scope/layer31.port5002.pid"
}
```

Parse values with `jq`:

```bash
BASE_URL=$(jq -r '.base_url' <<<"$START_JSON")
PIDFILE=$(jq -r '.pidfile' <<<"$START_JSON")
SOURCE=$(jq -r '.source' <<<"$START_JSON")
```

### Which GPU

`--cuda-devices` defaults to `auto`: the script reads `nvidia-smi` and picks the GPU with the most
free memory among those that still have `--vllm-gpu-memory-utilization` x total free (plus 1 GiB of
slack). That is the same condition vLLM itself checks at startup, so on a shared 8-GPU box `auto`
avoids landing on a card another job already fills and failing twenty minutes later with an
out-of-memory error. Pass an explicit index (`--cuda-devices 3`) to override, and
`qwen-scope-layer-server pick-gpu` to see which GPU `auto` would choose right now.

### What "ready" means

`/health` returns 500 `Server not initialized` while the model loads, and keeps returning that,
with the error appended, if loading failed. `start` keeps waiting on the first and exits non-zero
immediately on the second (after stopping the process), rather than waiting out the
`--timeout-seconds` budget. Once `/health` is 200, it sends one tiny `/v1/activation/source`
request naming `layerN`: a server that came up without that SAE (bad pattern, failed download) is
refused here rather than by the first real client request. `--no-probe` skips that step. If the
server needs a shared secret, export `NEURONPEDIA_SECRET` (or `SECRET`) and the probe sends it.

Startup on this machine takes about 90 s when the model weights are in the HF cache.

### Logs

The log file is appended to, not truncated, with a banner per start, so a sweep that restarts the
same layer after a failure keeps the earlier attempt's output.

## Stop The Server

```bash
qwen-scope-layer-server stop --pidfile "$PIDFILE"
```

`stop` sends `TERM` to the server's process group (read back from `ps` after launch, not assumed),
waits until every process in the group is gone **and** the port has closed, and escalates to `KILL`
if that takes longer than `--timeout-seconds` (default 60). It exits non-zero only if the group is
still alive after `KILL`. Typical teardown is about 5 s, after which `nvidia-smi` shows the memory
released.

## Status

```bash
qwen-scope-layer-server status --pidfile "$PIDFILE"
```

Reports `running` (with the current `/health` state: `ready`, `initializing`, `down` or
`failed:<message>`) or `stopped`, exiting 1 in the latter case.

## Test Another Layer

Change only `--layer`:

```bash
qwen-scope-layer-server start --layer 42 --port 5002
```

The script maps `--layer 42` to `source: "layer42"` and starts with:

```bash
--saelens_release qwen-scope-3.5-27b-w80k-l50
--include_sae '^layer42$'
```

It also sets `SAE_SETS='[]'` so the default `res-jb` set is not loaded accidentally. Note that
`start.py` lets environment variables override its arguments, so `--extra-arg` cannot change the
release or SAE set; it is for flags the script does not already set.

## Sweep Loop

A shell loop, one server at a time:

```bash
for layer in 32 36 40 44 48; do
  START_JSON=$(qwen-scope-layer-server start --layer "$layer" --port 5002)
  BASE_URL=$(jq -r '.base_url' <<<"$START_JSON")
  PIDFILE=$(jq -r '.pidfile' <<<"$START_JSON")
  SOURCE=$(jq -r '.source' <<<"$START_JSON")

  echo "Exploring $SOURCE at $BASE_URL"
  # Replace this with your client.
  python run_extraction.py --base-url "$BASE_URL" --source "$SOURCE"

  qwen-scope-layer-server stop --pidfile "$PIDFILE"
done
```

The feature-steering study in `lib-version-adapter` has a pipeline built on this script,
`src.feature_evaluation.run_sweep`, that runs extraction, ranking, study preparation and both
steering stages per layer with resumable per-layer artifacts and a sweep manifest; see
`src/feature_evaluation/README.md` there ("Layer sweep").

## Activation Request Shape

For positive/negative feature extraction, send selective positions:

```json
{
  "model": "Qwen/Qwen3.5-27B",
  "source": "layer31",
  "promptTokenIds": [
    [100, 200, 300, 401],
    [100, 200, 300, 402]
  ],
  "activationPositions": [
    [-1],
    [-1]
  ],
  "insertion": {
    "bos": "never",
    "eos": "never",
    "prefixTokenIds": [],
    "suffixTokenIds": []
  }
}
```

Qwen still processes the complete contextual sequences. `activationPositions` only reduces the SAE
encode work and returned feature set by passing selected hidden states through the SAE.

## Layer List

```bash
qwen-scope-layer-server list-layers
```

prints `layer0` through `layer63`, one per line.

## Memory Note

Each Qwen-Scope SAE layer is roughly `5120 x 81920` encoder plus decoder weights. At bf16 that is
about `1.6 GiB` per layer before module and allocator overhead. All 64 layers are roughly `100 GiB`
of SAE weights before overhead, in addition to Qwen 27B and vLLM memory. One layer per process is the
safe default for sweeps. With `--vllm-gpu-memory-utilization 0.70` on a 141 GiB H200, a server needs
about 100 GiB free on its GPU; two of them do not fit on one card.
