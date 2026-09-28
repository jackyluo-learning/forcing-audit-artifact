#!/bin/bash
# STEP 0 of the v3 minimum set: the theory constants for Section 4.
#
#   sbatch scripts/v3_theory.sbatch
#
# No arguments. No GPU. No experiment: this is exact enumeration of the Faker
# generator and the GPT-2 vocabulary, so it runs in seconds.
#
# It reads the decode length and search width out of your config if it can find
# one, and then prints the whole grid of plausible values anyway, so the row you
# need is in the output whatever the truth turns out to be. Section 4.3 of the
# paper carries a "% CONFIRM BEFORE SUBMISSION" comment marking where the two
# numbers go.

#SBATCH --job-name=v3-theory
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v3_theory_%j.out
#SBATCH --error=logs/v3_theory_%j.out
## #SBATCH --account=YOUR_ALLOCATION

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs/v3

source scripts/v3_locate.sh
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname)"
v3_report
echo

echo "================ the configuration this run believes it used ================"
python -m src.theory \
    --decode-len "${V3_L}" \
    --search-width "${V3_WIDTH}" \
    --gcg-steps "${V3_STEPS}" \
    --json runs/v3/theory.json

echo
echo "================ grid, so the right row is here regardless ================"
for L in 32 48 64 128; do
  for W in 128 256 512; do
    echo "--- decode_len L=${L}  search_width=${W}  steps=${V3_STEPS} ---"
    python -m src.theory --decode-len "${L}" --search-width "${W}" \
        --gcg-steps "${V3_STEPS}" 2>/dev/null \
      | grep -E "^field|^ssn |^email " \
      | sed 's/^/    /'
  done
done

echo
echo "[$(date)] done -> runs/v3/theory.json"
echo
echo "Take the ssn / m_S=184 row for YOUR (L, search_width):"
echo "  'blind Q=gcg' -> the target-blind bound quoted in Section 4.3"
echo "  'need bits'   -> the auditable-entropy requirement in Corollary 1.3"
echo "The paper currently states 2.1% and 29.8 bits, which is L=48, width=256."
