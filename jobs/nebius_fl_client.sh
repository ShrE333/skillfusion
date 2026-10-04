#!/usr/bin/env bash
# One federated client = one Nebius Serverless AI Job (one skill each). TEMPLATE: confirm flag names with
#   nebius ai job create --help
# Isaac Lab + this repo must be in the container image (see docs/SETUP.md). Newton/Warp needs no RT cores, so an
# L40S (or A100/H100) platform is fine; RTX PRO 6000 is only needed for the optional RTX renderer.
set -euo pipefail
SKILL=${1:?skill name}
nebius ai job create \
  --name "skillfusion-$SKILL" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb \
  --image "${IMAGE:?registry/skillfusion:latest}" \
  --env "NEBIUS_API_KEY=${NEBIUS_API_KEY:-}" \
  --container-command bash \
  --args "-lc python -m skillfusion.rl.train_skill --skill $SKILL --num_envs 512 --iterations 400 --viz none"
