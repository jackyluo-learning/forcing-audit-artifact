#!/bin/bash
# STEP 1 of the v3 minimum set: zero-GPU diagnostics on an EXISTING attempt log.
#
# The one that matters is the COPY DIAGNOSTIC. Control success in the sweep is
# 0/150 at k=1,2,3 and only opens at k=4-6, which is exactly the token length of
# the targets themselves. That is the signature of prompt-to-output copying
# rather than of forcing, and a reviewer will ask. This answers it.
#
# Submit:
#   sbatch scripts/v3_diagnostics.sbatch
#
# No arguments: the newest attempt log under runs/ is found automatically.
# To pick a different one, or a single analysis:
#   V3_LOG=runs/x/attempts.jsonl sbatch scripts/v3_diagnostics.sbatch
#   V3_ONLY=copy sbatch scripts/v3_diagnostics.sbatch
#   (V3_ONLY choices: copy,acr,seed,field,feasibility,twobytwo,scoring)
#
# IMPORTANT, READ BEFORE SUBMITTING
# ---------------------------------
# The v3 schema needs 16 columns and your OLD harness log will not have them all
# (it had no control arm at all, so `target_membership` is certainly absent).
# The job runs the validator FIRST and it names every missing column, then stops
# rather than producing half-valid numbers.
#
# Column mapping from the old harness to the v3 schema:
#   model          -> model_name          hit         -> exact_match
#   individual_id  -> person_id           output      -> generation
#   frequency      -> train_frequency     best_prompt -> prompt_text
#   method         -> probe
# Still to be supplied: run_id, exp_id, model_state, target_membership,
#   capacity_k, target_string, scored_value, decode_len_L
#
# If writing that adapter is more than you have time for, skip this job and do
# the two-minute eyeball version on the login node instead:
#
#   python3 -c "
#   import json
#   for l in open('runs/<sweep>/prompts.jsonl'):
#       r=json.loads(l)
#       if r.get('kind')=='gcg' and r.get('success'): print(repr(r['prompt'])[:160])
#   " | head -20
#
# Then look for the target's digits inside the prompt. Present -> copying.
# Absent -> not copying. That is enough to write one honest sentence.

#SBATCH --job-name=v3-diag
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v3_diag_%j.out
#SBATCH --error=logs/v3_diag_%j.out
## #SBATCH --account=YOUR_ALLOCATION

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs/v3

source scripts/v3_locate.sh
LOG="${V3_LOG:-}"
ONLY="${V3_ONLY:-}"

v3_report
echo
if [ -z "${LOG}" ] || [ ! -f "${LOG}" ]; then
  echo "ERROR: no per-attempt log found under runs/."
  echo "Point at one explicitly:  V3_LOG=path/to/attempts.jsonl sbatch $0"
  exit 2
fi

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname) log=${LOG}"
echo
echo "================ schema validation ================"
python -m src.logschema "${LOG}"
VALID=$?
if [ "${VALID}" -ne 0 ]; then
  echo
  echo "Schema validation failed. The columns named above are what the analyses"
  echo "need. Map your old column names (see the header of this script), or run"
  echo "the eyeball copy check on the login node instead."
  exit "${VALID}"
fi

echo
echo "================ diagnostics ================"
if [ -n "${ONLY}" ]; then
  python -m src.diagnostics "${LOG}" --only "${ONLY}" --out runs/v3/diag.json
else
  python -m src.diagnostics "${LOG}" --out runs/v3/diag.json
fi

echo
echo "[$(date)] done -> runs/v3/diag.json"
echo "Read 'non_copy_rate' first: that is the rate that deserves to be called forcing."
