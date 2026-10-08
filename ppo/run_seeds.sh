#!/usr/bin/env bash
# Launch the same configuration over several seeds in parallel (one CPU core each).
#   ppo/run_seeds.sh Ant-v4 tanh64 "--hidden 64 64 --activation tanh" 1 2 3 4 5
set -euo pipefail
ENV_ID=$1; EXP=$2; EXTRA=$3; shift 3
PY=${PY:-$(dirname "$0")/../.venv/bin/python}
for SEED in "$@"; do
  OMP_NUM_THREADS=1 "$PY" -u "$(dirname "$0")/train.py" --env-id "$ENV_ID" --exp-name "$EXP" --seed "$SEED" $EXTRA \
    > "runs/${ENV_ID}__${EXP}__${SEED}.log" 2>&1 &
done
wait
echo "all seeds finished"
