#!/bin/bash
# STEP 2 of the v3 minimum set: memorization verification (experiment E5).
#
# WHY THIS ONE IS WORTH A GPU JOB
# The paper's load-bearing premise is "the content IS memorized". Right now the
# only evidence for it is a soft prompt that recovers 100% of trained targets --
# and also 100% of controls, so by the paper's own argument it carries zero
# attribution information. Meanwhile every fixed prompt, including a true-prefix
# completion, recovers 0%. That combination is not currently explainable, and
# this job resolves it.
#
# It runs NO optimization: only teacher-forced target NLL and greedy completion
# from the true training-template prefix, on checkpoints you already have.
# Roughly 400 forward passes for 100 targets on two checkpoints.
#
# Submit:
#   sbatch scripts/v3_e5_memorization.sbatch
#
# No arguments: the newest fine-tuned checkpoint and individuals.json under
# runs/ are found automatically, and matched controls are generated if none
# exist (entropy known exactly by construction, identifier ranges that cannot
# collide with real SSNs). Override any of them by exporting first:
#   V3_CKPT=... V3_INDIV=... V3_CONTROLS=... sbatch scripts/v3_e5_memorization.sbatch
#
# HOW TO READ THE RESULT
#   true-prefix greedy hit rate HIGH  -> the premise holds; cite it as measured
#                                        and drop the soft-prompt argument.
#   true-prefix greedy hit rate ~0    -> that is itself the finding. Reframe as
#                                        "the audit reports forcing where little
#                                        is verbatim-extractable", and delete the
#                                        "content IS memorized" paragraph in 6.1.
# Either way the circularity goes away.

#SBATCH --job-name=v3-e5
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3_e5_%j.out
#SBATCH --error=logs/v3_e5_%j.out
## #SBATCH --account=YOUR_ALLOCATION

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs/v3

source scripts/v3_locate.sh
CKPT="${V3_CKPT:-}"
INDIV="${V3_INDIV:-}"
CONTROLS="${V3_CONTROLS}"
BASE_MODEL="${V3_BASE}"

v3_report
echo
if [ -z "${CKPT}" ] || [ ! -e "${CKPT}" ]; then
  echo "ERROR: no fine-tuned checkpoint found under runs/*/models/."
  echo "Point at one explicitly:  V3_CKPT=path/to/ckpt sbatch $0"
  exit 2
fi
if [ -z "${INDIV}" ] || [ ! -f "${INDIV}" ]; then
  echo "ERROR: no individuals.json found under runs/."
  echo "Point at one explicitly:  V3_INDIV=path/to/individuals.json sbatch $0"
  exit 2
fi

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date)] ckpt=${CKPT} base=${BASE_MODEL}"
nvidia-smi || true

# ---- controls: generate if not supplied ----
if [ ! -f "${CONTROLS}" ]; then
  echo
  echo "================ generating matched controls ================"
  python -m src.controls -n 200 --seed 99991 \
      --individuals "${INDIV}" \
      --tokenizer "${BASE_MODEL}" --ref-model distilgpt2 \
      --fields ssn email \
      --out "${CONTROLS}"
fi

echo
echo "================ E5: memorization verification ================"
# float32 on purpose: NLL in float16 is too noisy to compare arms.
python experiments/e5_nll_decomposition.py \
    --model "${CKPT}" \
    --base-model "${BASE_MODEL}" \
    --individuals "${INDIV}" \
    --controls "${CONTROLS}" \
    --fields ssn email \
    --dtype float32 \
    --out runs/v3/e5_bits.json

echo
echo "[$(date)] done -> runs/v3/e5_bits.json"
echo "Look at the true-prefix greedy hit rate for the D arm first; it decides"
echo "whether Section 6.1's memorization paragraph stands or gets rewritten."
