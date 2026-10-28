set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE=true
export WANDB_MODE=disabled
export COMET_MODE=DISABLED
export PIP_DISABLE_PIP_VERSION_CHECK=1
exec python3 -B -u train.py --launch "$@"
