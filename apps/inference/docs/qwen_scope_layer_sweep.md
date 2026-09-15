# Qwen-Scope Layer Sweep Server

Use `qwen-scope-layer-server` to start one Neuronpedia inference server per Qwen-Scope SAE layer.
The command waits for `/health`, prints a JSON success line, and gives you a pidfile you can use to
stop the server when that layer's exploration is done.

On this machine, `qwen-scope-layer-server` is installed as a symlink in `~/.local/bin`, pointing at:

```text
/home/daniele.cipollone/neuronpedia/apps/inference/scripts/qwen_scope_layer_server.sh
```

If you are on another machine, either create the same symlink or call the script by absolute path.

Available Qwen-Scope layers are `layer0` through `layer63`.

## Start One Layer

```bash
START_JSON=$(qwen-scope-layer-server \
  start \
  --repo /home/daniele.cipollone/neuronpedia \
  --layer 31 \
  --cuda-devices 0 \
  --port 5002)

echo "$START_JSON"
```

The success output is one JSON line:

```json
{
  "status": "ready",
  "pid": 12345,
  "pgid": 12345,
  "source": "layer31",
  "base_url": "http://127.0.0.1:5002",
  "pidfile": "/tmp/neuronpedia-qwen-scope/layer31.port5002.pid",
  "log": "/tmp/neuronpedia-qwen-scope/layer31.port5002.log",
  "stop_command": "qwen-scope-layer-server stop --pidfile /tmp/neuronpedia-qwen-scope/layer31.port5002.pid"
}
```

Parse values with `jq`:

```bash
BASE_URL=$(jq -r '.base_url' <<<"$START_JSON")
PIDFILE=$(jq -r '.pidfile' <<<"$START_JSON")
SOURCE=$(jq -r '.source' <<<"$START_JSON")

echo "server $PIDFILE is serving $SOURCE at $BASE_URL"
```

## Stop The Server

```bash
qwen-scope-layer-server \
  stop \
  --pidfile "$PIDFILE"
```

`stop` sends `TERM` to the server process group, waits, and escalates to `KILL` if the process group
does not exit within the stop timeout.

## Test Another Layer

Change only `--layer`:

```bash
qwen-scope-layer-server \
  start \
  --repo /home/daniele.cipollone/neuronpedia \
  --layer 42 \
  --cuda-devices 0 \
  --port 5002
```

The script maps `--layer 42` to `source: "layer42"` and starts with:

```bash
--saelens_release qwen-scope-3.5-27b-w80k-l50
--include_sae '^layer42$'
```

It also sets `SAE_SETS='[]'` so the default `res-jb` set is not loaded accidentally.

## Full Sweep Loop

This example starts one layer, runs your extraction command, then stops that layer before moving to
the next one.

```bash
REPO=/home/daniele.cipollone/neuronpedia

for layer in $(seq 0 63); do
  START_JSON=$(qwen-scope-layer-server start --repo "$REPO" --layer "$layer" --cuda-devices 0 --port 5002)
  BASE_URL=$(jq -r '.base_url' <<<"$START_JSON")
  PIDFILE=$(jq -r '.pidfile' <<<"$START_JSON")
  SOURCE=$(jq -r '.source' <<<"$START_JSON")

  echo "Exploring $SOURCE at $BASE_URL"

  # Replace this with your extraction client.
  python run_extraction.py \
    --base-url "$BASE_URL" \
    --source "$SOURCE"

  qwen-scope-layer-server stop --pidfile "$PIDFILE"
done
```

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

Print the list:

```bash
qwen-scope-layer-server list-layers
```

The complete layer set is:

```text
layer0
layer1
layer2
layer3
layer4
layer5
layer6
layer7
layer8
layer9
layer10
layer11
layer12
layer13
layer14
layer15
layer16
layer17
layer18
layer19
layer20
layer21
layer22
layer23
layer24
layer25
layer26
layer27
layer28
layer29
layer30
layer31
layer32
layer33
layer34
layer35
layer36
layer37
layer38
layer39
layer40
layer41
layer42
layer43
layer44
layer45
layer46
layer47
layer48
layer49
layer50
layer51
layer52
layer53
layer54
layer55
layer56
layer57
layer58
layer59
layer60
layer61
layer62
layer63
```

## Memory Note

Each Qwen-Scope SAE layer is roughly `5120 x 81920` encoder plus decoder weights. At bf16 that is
about `1.6 GiB` per layer before module and allocator overhead. All 64 layers are roughly `100 GiB`
of SAE weights before overhead, in addition to Qwen 27B and vLLM memory. One layer per process is the
safe default for sweeps.
