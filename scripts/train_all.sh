#!/usr/bin/env bash
# Train each skill alone (baseline), then run federated fusion. 10 GB GPU: keep num_envs modest.
set -euo pipefail
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}
for s in ${SKILLS:-pick_place lift push}; do
  "$ISAACLAB/isaaclab.sh" -p -m skillfusion.rl.train_skill --skill "$s" --num_envs "${NUM_ENVS:-128}" \
    --iterations "${ITERS:-300}" --viz none
done
"$ISAACLAB/isaaclab.sh" -p -m skillfusion.fl.job --skills "${SKILLS// /,}" --rounds "${ROUNDS:-6}" --num_envs "${NUM_ENVS:-128}"
python3 -m skillfusion.fl.fuse_cli
