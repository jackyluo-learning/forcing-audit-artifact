#!/bin/bash
# Locate everything the v3 jobs need, so no submit command carries a placeholder.
#
# Sourced by the v3_*.sbatch scripts. Also runnable on its own to see what it
# finds before you submit anything:
#
#   bash scripts/v3_locate.sh
#
# Every value can be overridden by exporting it first, e.g.
#   V3_CKPT=/data/my/ckpt sbatch scripts/v3_e5_memorization.sbatch
#
# Exports: V3_RUN_DIR V3_CKPT V3_BASE V3_INDIV V3_CONTROLS V3_LOG
#          V3_L V3_WIDTH V3_STEPS V3_CONFIG

# Newest match of a glob, or empty. Globs that match nothing expand to
# themselves in bash, so each candidate is existence-checked.
_v3_newest() {
  local best="" t_best=0 c t
  for c in "$@"; do
    [ -e "$c" ] || continue
    t=$(stat -c %Y "$c" 2>/dev/null || stat -f %m "$c" 2>/dev/null || echo 0)
    if [ "$t" -ge "$t_best" ]; then t_best="$t"; best="$c"; fi
  done
  printf '%s' "$best"
}

# --- the run directory --------------------------------------------------- #
: "${V3_RUN_DIR:=$(_v3_newest runs/*/ )}"
V3_RUN_DIR="${V3_RUN_DIR%/}"

# --- fine-tuned checkpoint ------------------------------------------------ #
# A HF checkpoint dir is one containing a config.json.
if [ -z "${V3_CKPT:-}" ]; then
  for c in $(ls -dt runs/*/models/*/ 2>/dev/null); do
    if [ -f "${c}config.json" ]; then V3_CKPT="${c%/}"; break; fi
  done
fi
: "${V3_BASE:=gpt2}"

# --- individuals / controls ---------------------------------------------- #
: "${V3_INDIV:=$(_v3_newest runs/*/data/individuals.json runs/*/individuals.json \
                             data/individuals.json)}"
: "${V3_CONTROLS:=$(_v3_newest runs/*/data/controls.json runs/*/controls.json \
                                runs/v3/controls.json)}"
[ -n "${V3_CONTROLS}" ] || V3_CONTROLS="runs/v3/controls.json"   # will be generated

# --- per-attempt log ----------------------------------------------------- #
: "${V3_LOG:=$(_v3_newest runs/*/attempts_v3.jsonl runs/*/attempts_v3.parquet \
                           runs/*/attempts.jsonl runs/*/attempts.parquet)}"

# --- the run's constants, read out of a config rather than guessed -------- #
: "${V3_CONFIG:=$(_v3_newest runs/*/config.yaml runs/*/config.yml \
                              configs/full_fast.yaml configs/full.yaml)}"
if [ -n "${V3_CONFIG}" ] && [ -f "${V3_CONFIG}" ]; then
  _v3_cfg=$(python3 - "$V3_CONFIG" <<'PY' 2>/dev/null
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
except Exception:
    sys.exit(0)
e = (c.get("extract") or {})
g = (e.get("gcg") or {})
print(e.get("max_new_tokens", ""), g.get("search_width", ""), g.get("num_steps", ""))
PY
)
  set -- ${_v3_cfg:-}
  : "${V3_L:=${1:-}}"
  : "${V3_WIDTH:=${2:-}}"
  : "${V3_STEPS:=${3:-}}"
fi
# Fallbacks only if no config could be read. These are NOT authoritative: the
# theory job prints a grid so the right row is present whatever the truth is.
: "${V3_L:=48}"
: "${V3_WIDTH:=256}"
: "${V3_STEPS:=200}"

export V3_RUN_DIR V3_CKPT V3_BASE V3_INDIV V3_CONTROLS V3_LOG V3_L V3_WIDTH V3_STEPS V3_CONFIG

v3_report() {
  echo "run dir     : ${V3_RUN_DIR:-<none found>}"
  echo "checkpoint  : ${V3_CKPT:-<none found: export V3_CKPT=...>}"
  echo "base model  : ${V3_BASE}"
  echo "individuals : ${V3_INDIV:-<none found: export V3_INDIV=...>}"
  echo "controls    : ${V3_CONTROLS}$([ -f "${V3_CONTROLS}" ] || echo '  (will be generated)')"
  echo "attempt log : ${V3_LOG:-<none found: export V3_LOG=...>}"
  echo "config      : ${V3_CONFIG:-<none found>}"
  echo "  decode_len L = ${V3_L}   search_width = ${V3_WIDTH}   steps = ${V3_STEPS}"
}

# Standalone invocation: just report.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
  v3_report
fi
