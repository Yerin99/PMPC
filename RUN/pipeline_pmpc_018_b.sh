#!/bin/bash
set -eo pipefail

# =============================================================================
# PMPC 018 Condition B: Strategy via Pd + LoRA Routing (no probe)
# p_sample sweep × (prepare → backbone → grid search → ensemble)
#
# Key differences from DKFPE 018 B:
# - Uses strategy_predictor instead of probe
# - No probe training step (strategy_predictor is trained end-to-end)
# - No probe source variants (enc/dec) — single config
# - PMPC hyperparams: lr=1e-5, epochs=8, max_input=160, max_dec=40
# - PMPC inference: k=30, p=0.9, t=0.7, rp=1.03
#
# Usage:
#   bash RUN/pipeline_pmpc_018_b.sh
#   LORA_LR=5e-5 bash RUN/pipeline_pmpc_018_b.sh
# =============================================================================

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate cuda

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- Configuration ---
P_SAMPLES=(0.0 0.05 0.1 0.15 0.2)
TRAIN_FILE=./_reformat/train.txt
VALID_FILE=./_reformat/valid.txt
TEST_FILE=./_reformat/test.txt

SASREC_SRC_DIR=/home/yerin/baseline/DKFPE/codes_zcj/DATA/sasrec
SASREC_MODEL=./DATA/sasrec/best_model.pt
SASREC_VALID_PROBS=./DATA/sasrec/valid_probs.json
SASREC_TEST_PROBS=./DATA/sasrec/test_probs.json

# PMPC inference kwargs
INFER_KWARGS="--temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 --repetition_penalty 1.03 --no_repeat_ngram_size 0 --max_length 50 --min_length 5"

# LoRA learning rate
LORA_LR=${LORA_LR:-"5e-5"}

# Skip backbone env vars (set to existing dir path to skip training)
SKIP_BACKBONE_00=${SKIP_BACKBONE_00:-""}
SKIP_BACKBONE_005=${SKIP_BACKBONE_005:-""}
SKIP_BACKBONE_01=${SKIP_BACKBONE_01:-""}
SKIP_BACKBONE_015=${SKIP_BACKBONE_015:-""}
SKIP_BACKBONE_02=${SKIP_BACKBONE_02:-""}

# =============================================================================
# Helper functions
# =============================================================================

get_skip_dir() {
    local p="$1"
    case "${p}" in
        0.0)  echo "${SKIP_BACKBONE_00}" ;;
        0.05) echo "${SKIP_BACKBONE_005}" ;;
        0.1)  echo "${SKIP_BACKBONE_01}" ;;
        0.15) echo "${SKIP_BACKBONE_015}" ;;
        0.2)  echo "${SKIP_BACKBONE_02}" ;;
        *)    echo "" ;;
    esac
}

find_best_epoch() {
    local output_dir="$1"
    python3 -c "
import csv
best_epoch, best_ppl = 0, float('inf')
with open('${output_dir}/eval_log.csv') as f:
    for row in csv.DictReader(f):
        ppl = float(row['freq_ppl'])
        if ppl < best_ppl:
            best_ppl = ppl
            best_epoch = int(row['epoch'])
print(best_epoch)
"
}

find_latest_dir() {
    local base_dir="$1"
    ls -dt "${base_dir}"/*/ 2>/dev/null | head -1 | sed 's:/$::'
}

gen_json_dir() {
    local ckpt_dir="$1"
    local ckpt_name="$2"
    local input_name="$3"
    echo "${ckpt_dir}/res_${ckpt_name}_${input_name}_k.30_p.0.9_b.1_t.0.7_lp.1.0_rp.1.03_ng.0"
}

# =============================================================================
# Step 0: SASRec setup + Prepare B condition data
# =============================================================================

echo "================================================================"
echo "  Step 0a: SASRec model + probs setup"
echo "================================================================"

mkdir -p ./DATA/sasrec
for F in best_model.pt valid_probs.json test_probs.json; do
    DEST="./DATA/sasrec/${F}"
    if [ ! -f "${DEST}" ]; then
        SRC="${SASREC_SRC_DIR}/${F}"
        if [ -f "${SRC}" ]; then
            echo "Copying ${F}..."
            cp "${SRC}" "${DEST}"
        else
            # Try subdirectory pattern
            SRC=$(find "${SASREC_SRC_DIR}" -name "${F}" -type f 2>/dev/null | head -1)
            if [ -n "${SRC}" ]; then
                echo "Copying ${SRC}..."
                cp "${SRC}" "${DEST}"
            else
                echo "WARNING: ${F} not found in ${SASREC_SRC_DIR}"
            fi
        fi
    fi
done

echo "SASRec ready: ${SASREC_MODEL}, ${SASREC_VALID_PROBS}, ${SASREC_TEST_PROBS}"
echo "LoRA learning rate: ${LORA_LR}"
echo ""

# Build optional --lora_lr flag
LORA_LR_FLAG=""
if [ -n "${LORA_LR}" ]; then
    LORA_LR_FLAG="--lora_lr ${LORA_LR}"
fi

echo "================================================================"
echo "  Step 0b: Prepare B condition data.pkl (--no_strat_in_seq)"
echo "================================================================"

DATA_DIR="./DATA/pmpc.pmpc_018_b"
if [ ! -f "${DATA_DIR}/data.pkl" ]; then
    python prepare.py \
        --config_name pmpc_018_b \
        --inputter_name pmpc \
        --train_input_file "${TRAIN_FILE}" \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --no_strat_in_seq \
        --single_processing
    echo "B condition data.pkl created at ${DATA_DIR}"
else
    echo "B condition data.pkl already exists at ${DATA_DIR}, skipping"
fi
echo ""

# =============================================================================
# Main loop: p_sample sweep
# =============================================================================

for P in "${P_SAMPLES[@]}"; do
    echo "================================================================"
    echo "  p_sample=${P}: Starting pipeline (PMPC 018 B condition)"
    echo "================================================================"

    SKIP_DIR=$(get_skip_dir "${P}")

    # -----------------------------------------------------------------
    # Step 1: Backbone training (PMPC + LoRA + SS)
    # -----------------------------------------------------------------
    if [ -n "${SKIP_DIR}" ]; then
        echo "Skipping backbone for p=${P} (using ${SKIP_DIR})"
        BACKBONE_DIR="${SKIP_DIR}"
    else
        echo "--- Step 1: Backbone training (p_sample=${P}, B condition) ---"
        python train.py \
            --config_name pmpc_018_b \
            --inputter_name pmpc \
            --seed 42 \
            --max_input_length 160 \
            --max_decoder_input_length 40 \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 1e-5 \
            ${LORA_LR_FLAG} \
            --num_epochs 8 \
            --warmup_steps 100 \
            --fp16 false \
            --eval_input_file "${VALID_FILE}" \
            --pbar true \
            --use_scheduled_sampling true \
            --ss_schedule_type step \
            --ss_warmup_epochs 1 \
            --ss_p_sample "${P}" \
            --ss_p_random 0.0 \
            --no_strat_in_seq

        BACKBONE_DIR=$(find_latest_dir "./DATA/pmpc.pmpc_018_b")
    fi

    echo "Backbone dir: ${BACKBONE_DIR}"
    BEST_EPOCH=$(find_best_epoch "${BACKBONE_DIR}")
    echo "Best epoch: ${BEST_EPOCH} (lowest val PPL)"
    BEST_CKPT="${BACKBONE_DIR}/epoch-${BEST_EPOCH}.bin"

    # -----------------------------------------------------------------
    # Step 2: (No probe training needed — strategy_predictor is end-to-end)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Step 3: strategy_predictor inference on valid + grid search
    # -----------------------------------------------------------------
    echo "--- Step 3a: strategy_predictor inference on valid ---"
    python infer.py \
        --config_name pmpc_018_b \
        --inputter_name pmpc \
        --seed 42 \
        --load_checkpoint "${BEST_CKPT}" \
        --fp16 false \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --infer_batch_size 16 \
        --infer_input_file "${VALID_FILE}" \
        --no_strat_in_seq \
        ${INFER_KWARGS}

    # Locate gen.json
    CKPT_NAME="epoch-${BEST_EPOCH}.bin"
    GEN_DIR=$(gen_json_dir "${BACKBONE_DIR}" "${CKPT_NAME}" "valid")
    STRAT_GEN_JSON="${GEN_DIR}/gen.json"

    if [ ! -f "${STRAT_GEN_JSON}" ]; then
        echo "ERROR: gen.json not found at ${STRAT_GEN_JSON}"
        echo "  Trying to find it..."
        STRAT_GEN_JSON=$(find "${BACKBONE_DIR}" -name "gen.json" -path "*valid*" -type f 2>/dev/null | head -1)
        if [ -z "${STRAT_GEN_JSON}" ]; then
            echo "  Still not found. Skipping grid search for p=${P}"
            continue
        fi
        echo "  Found: ${STRAT_GEN_JSON}"
    fi

    # Grid search on valid set
    echo "--- Step 3b: Grid search ---"
    GRID_OUTPUT="${BACKBONE_DIR}/grid_search.json"
    python -m ensemble.grid_search_018 \
        --probe_probs_file "${STRAT_GEN_JSON}" \
        --sasrec_probs_file "${SASREC_VALID_PROBS}" \
        --output_file "${GRID_OUTPUT}" \
        --step 0.05

    # Extract best alpha
    BEST_ALPHA=$(python3 -c "
import json
with open('${GRID_OUTPUT}') as f:
    data = json.load(f)
print(data['best']['alpha'])
")
    echo "  Best alpha: ${BEST_ALPHA}"

    # -----------------------------------------------------------------
    # Step 4: Ensemble inference on test set
    # -----------------------------------------------------------------
    echo "--- Step 4: Ensemble inference on test (alpha=${BEST_ALPHA}) ---"
    python infer_ensemble_018.py \
        --config_name pmpc_018_b \
        --inputter_name pmpc \
        --seed 42 \
        --load_checkpoint "${BEST_CKPT}" \
        --sasrec_probs_file "${SASREC_TEST_PROBS}" \
        --ensemble_alpha "${BEST_ALPHA}" \
        --fp16 false \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --no_strat_in_seq \
        ${INFER_KWARGS} \
        --infer_batch_size 1 \
        --infer_input_file "${TEST_FILE}"

    # -----------------------------------------------------------------
    # Step 5: strategy_predictor-only inference on test (no ensemble, comparison)
    # -----------------------------------------------------------------
    echo "--- Step 5: strategy_predictor-only inference on test ---"
    python infer.py \
        --config_name pmpc_018_b \
        --inputter_name pmpc \
        --seed 42 \
        --load_checkpoint "${BEST_CKPT}" \
        --fp16 false \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --infer_batch_size 16 \
        --infer_input_file "${TEST_FILE}" \
        --no_strat_in_seq \
        ${INFER_KWARGS}

    echo ""
    echo "================================================================"
    echo "  p_sample=${P}: Pipeline complete (PMPC 018 B condition)"
    echo "================================================================"
    echo ""
done

echo "All p_sample sweeps complete! (PMPC 018 B condition)"
