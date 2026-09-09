#!/usr/bin/env bash
# Serve gpt-oss-120b for the PUCT loop.
#
# Default topology is DP=8 x TP=1: one independent vLLM replica per H100.
# gpt-oss-120b is MXFP4 (~61GB) so a replica fits on one 80GB card, and the
# model is MoE with only ~5.1B active parameters per token -- tensor-parallel
# all-reduce would cost more than it buys at that activation size. Replicas are
# plain separate servers rather than vLLM's built-in DP so a single crashed
# replica doesn't take the fleet down; the client load-balances across ports.
#
#   bash scripts/serve_gpt_oss.sh start     # launch replicas
#   bash scripts/serve_gpt_oss.sh status    # health + KV cache blocks
#   bash scripts/serve_gpt_oss.sh stop
#
# Env:
#   MODEL           default openai/gpt-oss-120b (or a local path)
#   TP              tensor-parallel size per replica   [1]
#   N_REPLICAS      number of replicas                 [8/TP]
#   BASE_PORT       first port                         [8100]
#   MAX_MODEL_LEN   context window                     [32768]
#   GPU_MEM_UTIL    fraction per GPU                   [0.92]
#   ENFORCE_EAGER   set to 1 to skip torch.compile (needs ninja) []
#   CONDA_ENV       env holding vLLM                   [lumen]
#   PYTHON          explicit interpreter (overrides CONDA_ENV)
#
# Both `lumen` (vLLM 0.23.0 / torch 2.11+cu129) and `memtrace-vllm`
# (0.27.1 / torch 2.13+cu130) register GptOssForCausalLM and mxfp4. `lumen`
# is the default because its torch/CUDA stack matches the rest of the project.
set -uo pipefail

MODEL=${MODEL:-openai/gpt-oss-120b}
TP=${TP:-1}
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
N_REPLICAS=${N_REPLICAS:-$((N_GPUS / TP))}
BASE_PORT=${BASE_PORT:-8100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
CONDA_ENV=${CONDA_ENV:-lumen}
LOGDIR=${LOGDIR:-$HOME/gpt_oss_serve}
mkdir -p "$LOGDIR"

ports() { for i in $(seq 0 $((N_REPLICAS - 1))); do echo $((BASE_PORT + i)); done; }

start() {
  # Resolve the interpreter directly. `conda` is a shell function sourced from
  # .bashrc, so `conda activate` silently no-ops under a non-interactive ssh
  # command and the servers end up on the system python without vLLM.
  PY=${PYTHON:-$HOME/.conda/envs/$CONDA_ENV/bin/python}
  if [ ! -x "$PY" ]; then
    echo "ERROR: interpreter not found at $PY (set PYTHON= or CONDA_ENV=)" >&2
    exit 1
  fi
  # torch.compile shells out to `ninja`, which lives in the env's bin dir.
  # Using the interpreter by absolute path skips activation, so put that dir on
  # PATH explicitly or compilation dies with FileNotFoundError: 'ninja'.
  export PATH="$(dirname "$PY"):$PATH"
  v=$("$PY" -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)
  case "$v" in
    *Error*|*error*) echo "ERROR: vLLM not importable in $PY -- $v" >&2; exit 1 ;;
  esac
  echo "vLLM $v | $N_REPLICAS replicas x TP=$TP | ctx=$MAX_MODEL_LEN | $PY"

  # vLLM sizes its allocation against *free* memory at startup, so a replica
  # launched while a previous one is still releasing its weights dies with
  # "Free memory ... is less than desired GPU memory utilization". Wait for the
  # cards to actually drain before claiming them.
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
    [ "$busy" -eq 0 ] && break
    echo "  waiting for $busy GPU(s) to release memory..."
    sleep 10
  done
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
  if [ "$busy" -ne 0 ]; then
    echo "ERROR: $busy GPU(s) still hold memory; refusing to start" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
    exit 1
  fi
  for i in $(seq 0 $((N_REPLICAS - 1))); do
    port=$((BASE_PORT + i))
    gpus=$(seq -s, $((i * TP)) $((i * TP + TP - 1)))
    CUDA_VISIBLE_DEVICES=$gpus nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --served-model-name gpt-oss-120b \
      --port "$port" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      ${ENFORCE_EAGER:+--enforce-eager} \
      --no-enable-log-requests \
      > "$LOGDIR/replica_$port.log" 2>&1 &
    echo "  replica $i -> GPU $gpus, port $port (pid $!)"
  done
  ports > "$LOGDIR/ports"   # watchdog reads the live topology from here
  echo "logs: $LOGDIR/replica_<port>.log"
  echo "wait for readiness with: bash $0 status"
}

status() {
  for p in $(ports); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$p/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
      kv=$(grep -ohE "GPU KV cache size: [0-9,]+ tokens" "$LOGDIR/replica_$p.log" 2>/dev/null | tail -1)
      echo "  port $p: READY   ${kv:-}"
    else
      last=$(tail -1 "$LOGDIR/replica_$p.log" 2>/dev/null | cut -c1-70)
      echo "  port $p: not ready (HTTP ${code:-none})  $last"
    fi
  done
}

stop() {
  # The API server and the EngineCore that actually holds the weights are
  # separate processes; killing only the former leaves orphaned EngineCores
  # pinning ~68GB per card, which makes the next start fail its free-memory
  # check. Kill by the PIDs the driver reports, then sweep by name.
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(whoami)" ] && kill -9 "$p" 2>/dev/null
  done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 10
  left=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
  echo "  仍占用显存的 GPU: $left"
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  ports) ports | paste -sd, - ;;
  *) echo "usage: $0 {start|status|stop|ports}"; exit 1 ;;
esac
