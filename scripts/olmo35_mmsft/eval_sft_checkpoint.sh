#!/bin/bash
# Evaluate one Olmo 3.5 multimodal SFT checkpoint: image QA in the SFT chat layout, plus the
# ledger's text cells on its language model. See README.md beside this script.
#
# usage: eval_sft_checkpoint.sh <short-name> <checkpoint step dir> vision   (export + image QA + HF conversion job)
#        eval_sft_checkpoint.sh <short-name> <checkpoint step dir> text     (once the conversion finished)
#
# Env: OLMO_CORE_DIR (an allenai/OLMo-core checkout at the training code, e.g. #942's branch; it exports
# the checkpoint), SCALING_LADDERS_DIR (allenai/scaling-ladders checkout with
# ladders/olmoe3/multimodal/eval_text.py; pushed), LEDGER_DIR (olmo-post-training-ledger checkout),
# OUT_ROOT (default /weka/oe-training-default/$USER_NAME/mmsft-eval), PY (python with OLMo-core's
# dependencies), OLMO_EVAL_PY (python with olmo-eval[beaker]), CELLS (text cells, default the SFT
# selection bands), plus eval_vision.sh's TASKS, LIMIT, GPUS.
set -euo pipefail
[[ $# -eq 3 ]] || { sed -n 2,14p "$0"; exit 2; }
NAME=$1; CKPT=${2%/}; STAGE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
USER_NAME=${USER_NAME:-jasonr}
OUT_ROOT=${OUT_ROOT:-/weka/oe-training-default/$USER_NAME/mmsft-eval}
OUT=$OUT_ROOT/$NAME; mkdir -p "$OUT"
EXPORT=$OUT/olmo-core-export; HF=$OUT/hf-multimodal; TEXT=$OUT/hf-text
PY=${PY:-python}; OLMO_EVAL_PY=${OLMO_EVAL_PY:-python}
case $STAGE in
  vision)
    : "${OLMO_CORE_DIR:?set OLMO_CORE_DIR}" "${SCALING_LADDERS_DIR:?set SCALING_LADDERS_DIR}"
    if [[ ! -f $EXPORT/model.safetensors ]]; then
      (cd "$OLMO_CORE_DIR" && OMP_NUM_THREADS=16 PYTHONPATH=$OLMO_CORE_DIR/src "$PY" \
        src/scripts/export_multimodal_olmoddp.py "$CKPT" "$EXPORT" --verify) 2>&1 | tail -3
    fi
    OLMO_EVAL_PY=$OLMO_EVAL_PY "$HERE/eval_vision.sh" "$NAME" "$EXPORT" olmo35
    # HF conversion (1 GPU, ~10 min) for the text cells; the text stage waits for _CONVERSION_SUCCESS.json.
    (cd "$SCALING_LADDERS_DIR/ladders/olmoe3/multimodal" && WORKSPACE=${WORKSPACE:-ai2/olmo-instruct} \
      "$PY" eval_text.py convert "$NAME" "$CKPT" --output "$HF")
    ;;
  text)
    : "${LEDGER_DIR:?set LEDGER_DIR}"
    [[ -f $HF/_CONVERSION_SUCCESS.json ]] || { echo "conversion not finished: $HF" >&2; exit 1; }
    [[ -d $TEXT ]] || "$PY" "$HERE/mm_hf_to_text_hf.py" "$HF" "$TEXT" --tokenizer olmo35
    "$PY" "$HERE/text_cells.py" "$TEXT" "$NAME" --ledger "$LEDGER_DIR" --user "$USER_NAME" \
      ${CELLS:+--cells $CELLS}
    ;;
  *) echo "stage must be vision or text" >&2; exit 2 ;;
esac
