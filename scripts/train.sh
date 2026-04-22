#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

# Always run from repo root so `python run.py` resolves (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# MuJoCo loads a GL backend at import time unless disabled. EGL/OSMesa often break in
# minimal containers (PyOpenGL PLATFORM is None). Training does not need rendering.
# Override if you have a working stack: MUJOCO_GL=egl or MUJOCO_GL=osmesa ./scripts/train.sh
export MUJOCO_GL="${MUJOCO_GL:-disable}"

# Optional conda: skip if not installed (e.g. already inside docker/venv).
# Override env name: CONDA_ENV=jaxgcrl ./scripts/train.sh
CONDA_ENV="${CONDA_ENV:-contrastive_rl}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

# method=crl
# env=ant_u_maze

# Tyro CLI order: run.py [RunConfig flags...] {crl|ppo|sac|td3} [agent flags...]
# Do NOT put "crl" first — it breaks parsing ("Unrecognized options").
for seed in 1 2 3; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 CUDA_VISIBLE_DEVICES=0 python run.py \
    --env "ant_big_maze" \
    --wandb-project-name ECE567 --wandb-group ant_big_maze --exp-name ant_big_maze_hn --num-evals 250 \
    --seed "${seed}" --total-env-steps 50000000 --num-envs 512 --action-repeat 1 \
    --episode-length 1000 --log-wandb \
    "crl" \
    --batch-size 256 --discounting 0.99 \
    --unroll-length 62 --min-replay-size 1000 --max-replay-size 10000 \
    --contrastive-loss-fn fwd_infonce --energy-fn norm --hard-neg-beta 1.0 \
    --train-step-multiplier 1
  done


for seed in 1 2 3; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 CUDA_VISIBLE_DEVICES=0 python run.py \
    --env "humanoid" \
    --wandb-project-name ECE567 --wandb-group humanoid --exp-name  humanoid_hn  --num-evals 250 \
    --seed "${seed}" --total-env-steps 50000000 --num-envs 512 --action-repeat 1 \
    --episode-length 1000 --log-wandb \
    "crl" \
    --batch-size 256 --discounting 0.99 \
    --unroll-length 62 --min-replay-size 1000 --max-replay-size 10000 \
    --contrastive-loss-fn fwd_infonce --energy-fn norm --hard-neg-beta 1.0 \
    --train-step-multiplier 1
  done

for seed in 1 2 3; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 CUDA_VISIBLE_DEVICES=0 python run.py \
    --env "ant" \
    --wandb-project-name ECE567 --wandb-group ant --exp-name ant_hn --num-evals 250 \
    --seed "${seed}" --total-env-steps 50000000 --num-envs 512 --action-repeat 1 \
    --episode-length 1000 --log-wandb \
    "crl" \
    --batch-size 256 --discounting 0.99 \
    --unroll-length 62 --min-replay-size 1000 --max-replay-size 10000 \
    --contrastive-loss-fn fwd_infonce --energy-fn norm --hard-neg-beta 1.0 \
    --train-step-multiplier 1
  done
  
echo "All runs have finished."
shutdown
