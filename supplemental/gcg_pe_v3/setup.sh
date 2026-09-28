#!/usr/bin/env bash
# One-shot environment setup. Run once on the server from the pipeline/ dir:
#   bash setup.sh
set -euo pipefail

PY="${PYTHON:-python3}"

echo "=== Creating virtualenv (.venv) ==="
"${PY}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Installing PyTorch (CUDA) ==="
# Adjust the index URL to your CUDA version if needed (cu121 shown).
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121 || \
  pip install torch   # fallback to default wheel

echo "=== Installing project requirements ==="
pip install -r requirements.txt

echo "=== Downloading spaCy model ==="
python -m spacy download en_core_web_sm

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  huggingface-cli login          # needed for Llama-2-7B access"
echo "  python run.py --config configs/smoke.yaml    # verify (minutes)"
echo "  python run.py --config configs/full.yaml     # full run"
