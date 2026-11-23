#!/bin/bash
# One command: build the inputs from public data, render one window with VC3D, run frozen hecate and S, score both.
#   bash run_window.sh OUT_DIR S_WEIGHTS_PATH
# Env: PYTHON (default python3), DEVICE (default cuda), and one of VC_RENDER_BIN (a local vc_render_tifxyz binary)
# or VC3D_IMAGE (a docker image holding /opt/vc3d/bin/vc_render_tifxyz). Export it; there is no default.
set -euo pipefail
[ -n "${VC_RENDER_BIN:-}${VC3D_IMAGE:-}" ] || { echo "run_window.sh: no VC3D renderer given; nothing was downloaded.
Set and export one of:
  export VC_RENDER_BIN=/path/to/vc_render_tifxyz    your own VC3D build
  export VC3D_IMAGE=<docker image>                  an image that holds /opt/vc3d/bin/vc_render_tifxyz" >&2; exit 2; }
OUT=$(realpath -m "$1"); SW=$(realpath "${2:-${S_WEIGHTS:?give the S checkpoint path}}")
PY=${PYTHON:-python3}; DEV=${DEVICE:-cuda}
D=$(dirname "$(realpath "$0")")
[ ! -e "$OUT/results.json" ] || { echo "OUT_DIR already holds a result: $OUT"; exit 2; }
$PY "$D/onsheet.py" prep   --out "$OUT"
$PY "$D/onsheet.py" fetch  --out "$OUT"
$PY "$D/onsheet.py" render --out "$OUT"
$PY "$D/onsheet.py" infer  --out "$OUT" --model frozen --device "$DEV"
$PY "$D/onsheet.py" infer  --out "$OUT" --model S --s-weights "$SW" --device "$DEV"
$PY "$D/onsheet.py" score  --out "$OUT"
