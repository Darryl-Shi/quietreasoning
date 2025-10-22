#!/usr/bin/env bash
# Launches a command across all workers of a TPU VM pod.
# Usage: ./launch_tpu.sh PROJECT TPU_NAME ZONE "command"
# Optionally set REPO_URL, WANDB_API_KEY, and HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars.

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 PROJECT TPU_NAME ZONE \"command\""
  exit 1
fi

PROJECT="$1"
TPU_NAME="$2"
ZONE="$3"
shift 3
CMD="$*"

REPO_URL="${REPO_URL:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"

if [[ -z "${HF_TOKEN}" && -n "${HUGGINGFACE_TOKEN}" ]]; then
  HF_TOKEN="${HUGGINGFACE_TOKEN}"
fi

if [[ -z "${HF_TOKEN}" && -n "${HUGGING_FACE_HUB_TOKEN}" ]]; then
  HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

if [[ -z "${HUGGING_FACE_HUB_TOKEN}" && -n "${HF_TOKEN}" ]]; then
  HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "Discovering TPU workers for ${TPU_NAME} in ${ZONE} (project ${PROJECT})..."
IFS=';' read -r -a WORKER_IPS <<< "$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
  --zone "${ZONE}" \
  --project "${PROJECT}" \
  --format='value(networkEndpoints[].ipAddress)')"

WORKER_COUNT=${#WORKER_IPS[@]}
if [[ ${WORKER_COUNT} -eq 1 && -z "${WORKER_IPS[0]}" ]]; then
  WORKER_COUNT=0
fi

if [[ ${WORKER_COUNT} -eq 0 ]]; then
  echo "No workers found. Check TPU name and zone."
  exit 1
fi

echo "Launching command on workers:"
for ((i=0; i<WORKER_COUNT; ++i)); do
  echo "  worker ${i} (${WORKER_IPS[i]})"
done

for ((i=0; i<WORKER_COUNT; ++i)); do
  echo "Starting worker ${i} (${WORKER_IPS[i]})..."
  REMOTE_B64=$(REPO_URL="${REPO_URL}" WANDB_API_KEY="${WANDB_API_KEY}" HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" CMD_CONTENT="${CMD}" WORKER_INDEX="${i}" PROCESS_COUNT="${WORKER_COUNT}" python3 - <<'PY'
import base64, os, shlex
repo = os.environ.get("REPO_URL", "")
wandb = os.environ.get("WANDB_API_KEY", "")
cmd = os.environ.get("CMD_CONTENT", "")
worker_index = int(os.environ.get("WORKER_INDEX", "0"))
process_count = int(os.environ.get("PROCESS_COUNT", "1"))
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
script_lines = [
    "set -euo pipefail",
    "if command -v sudo >/dev/null 2>&1; then",
    "  sudo pkill -f run.py || true",
    "  sudo rm -f /tmp/libtpu_lockfile || true",
    "  sudo rm -rf /tmp/tpu_logs || true",
    "fi",
    "cd \"$HOME\"",
]
if repo:
    script_lines.append(
        f'if [ -d "$HOME/quietreasoning/.git" ]; then git -C "$HOME/quietreasoning" pull --ff-only; '
        f'else git clone {shlex.quote(repo)} "$HOME/quietreasoning"; fi'
    )
script_lines.extend([
    'cd "$HOME/quietreasoning"',
    'pip install -e .',
    'cd "$HOME"',
])
if wandb:
    script_lines.append(f'export WANDB_API_KEY={shlex.quote(wandb)}')
if hf_token:
    script_lines.append(f'export HF_TOKEN={shlex.quote(hf_token)}')
    script_lines.append(f'export HUGGING_FACE_HUB_TOKEN={shlex.quote(hf_token)}')
script_lines.extend([
    f'export JAX_PROCESS_INDEX={worker_index}',
    f'export JAX_PROCESS_COUNT={process_count}',
])
script_lines.append(cmd)
script = "\n".join(script_lines) + "\n"
print(base64.b64encode(script.encode()).decode())
PY
)

  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" \
    --project "${PROJECT}" \
    --worker="${i}" \
    --command "bash -lc 'echo ${REMOTE_B64} | base64 -d > /tmp/quietreasoning_launch.sh && bash /tmp/quietreasoning_launch.sh'" \
    -- -f
done

echo "Launch submitted to all workers."
