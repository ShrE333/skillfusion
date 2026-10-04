#!/usr/bin/env bash
# Live demo. Forward ports 8080 (3D scene), 8081 (panel), 8082 (push-to-talk) to your laptop:
#   ssh -L 8080:localhost:8080 -L 8081:localhost:8081 -L 8082:localhost:8082 <host>
set -euo pipefail
: "${NEBIUS_API_KEY:?export your Nebius Token Factory key}"
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}
"$ISAACLAB/isaaclab.sh" -p -m skillfusion.app \
  --viz viser --asr --docs "${DOCS_DIR:-docs/isaac}" "$@"
