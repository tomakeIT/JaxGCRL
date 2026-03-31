#!/bin/bash
set -euo pipefail

# If you pass explicit args, they are forwarded directly to the dispatcher:
#   ./scripts/train.sh --algorithms crl --envs ant --seeds 1 2 3 --dry-run
if [ "$#" -gt 0 ]; then
  python scripts/dispatch_train.py "$@"
  exit 0
fi

# Default matrix requested for benchmark-style experiments:
# algorithms: crl, ppo
# envs: ant, ant_u_maze
# seeds: 1..5
python scripts/dispatch_train.py \
  --algorithms crl \
  --envs ant \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --project-name jaxgcrl \
  --group-prefix benchmark \
  --gpu-ids 0 1 2 \
  --jobs-per-gpu 1

# echo "All dispatch jobs have finished."


# python scripts/dispatch_train.py \
#   --algorithms crl sac \
#   --envs pusher_hard humanoid ant_big_maze ant_ball\
#   --seeds 1 2 3 4 5 \
#   --project-name jaxgcrl \
#   --group-prefix benchmark \
#   --gpu-ids 0 1 2 \
#   --jobs-per-gpu 1

echo "All dispatch jobs have finished."
