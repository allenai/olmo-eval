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
EXPORT=$OUT/olmo-core-export
PY=${PY:-python}; OLMO_EVAL_PY=${OLMO_EVAL_PY:-python}
case $STAGE in
  vision)
    : "${OLMO_CORE_DIR:?set OLMO_CORE_DIR}"
    # The provider loads OLMo-core's consolidated export, not the raw OLMoDDP checkpoint.
    if [[ ! -f $EXPORT/model.safetensors ]]; then
      (cd "$OLMO_CORE_DIR" && OMP_NUM_THREADS=32 PYTHONPATH=$OLMO_CORE_DIR/src "$PY" \
        src/scripts/export_multimodal_olmoddp.py "$CKPT" "$EXPORT" --verify) 2>&1 | tail -3
    fi
    OLMO_EVAL_PY=$OLMO_EVAL_PY "$HERE/eval_vision.sh" "$NAME" "$EXPORT" olmo35
    # Text: the multimodal SFT exporter (OLMo-core #942 src/scripts/olmo35_mm_sft_export.sh) writes
    # hf_mm and the stripped hf_text with the promoted olmo35 tokenizer (<think>=100256,
    # </think>=100266) and runs a generation smoke; one GPU job. Launch it with OLMo-core's
    # olmo35_mm_sft_aux_job.py, or a spec like scripts/olmo35_mmsft/README.md shows, writing to $OUT/export.
    echo "now export the text model to $OUT/export (olmo35_mm_sft_export.sh $CKPT $OUT/export), then run: $0 $NAME $CKPT text"
    ;;
  text)
    : "${LEDGER_DIR:?set LEDGER_DIR}"
    TEXT=$OUT/export/hf_text
    [[ -f $TEXT/STRIP_RECEIPT.json ]] || { echo "text export not finished: $TEXT" >&2; exit 1; }
    "$PY" "$HERE/text_cells.py" "$TEXT" "$NAME" --ledger "$LEDGER_DIR" --user "$USER_NAME" \
      ${CELLS:+--cells $CELLS}
    ;;
  *) echo "stage must be vision or text" >&2; exit 2 ;;
esac
