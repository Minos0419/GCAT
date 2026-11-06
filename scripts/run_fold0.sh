#!/usr/bin/env bash
set -euo pipefail

GCAT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${GCAT_DATA_ROOT:-${GCAT_DIR}/../../data}
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${GCAT_CONFIG:-${GCAT_DIR}/assets/config.json}
DATASET=${1:?Usage: run_fold0.sh DATASET [OUT_DIR]}
TRAINER_OVERRIDE=${TRAINER_OVERRIDE:-}

case "${DATASET}" in
  data1)
    VARIANT=s4_bal_b64_e10
    TRAINER=train_v1.py
    QUERY=${DATA_ROOT}/data1_st.h5ad
    REF=${DATA_ROOT}/data1_sc.h5ad
    MASK=${DATA_ROOT}/data1_5fold_gene.npy
    MODULE=${GCAT_DIR}/assets/regulons/data1_collectri_min5_uncovered_denovo29.npy
    NICHE_BRIDGE=${GCAT_DIR}/assets/regulons/data1_nichenet_lr_program_to_prior_regulon_top5.npz
    EPOCH=10
    BATCH_SIZE=64
    CCC_MIN_PAIRS=32
    VALIDATION_ARGS=()
    ;;
  AD)
    VARIANT=collectri_min5_u5_k60_b96_e10_ps3
    TRAINER=train_v1_parent_compat.py
    QUERY=${DATA_ROOT}/AD_st.h5ad
    REF=${DATA_ROOT}/AD_sc.h5ad
    MASK=${DATA_ROOT}/AD_5fold_gene.npy
    MODULE=${GCAT_DIR}/assets/regulons/AD_collectri_min5_uncovered_denovo45.npy
    NICHE_BRIDGE=${GCAT_DIR}/assets/regulons/AD_nichenet_lr_program_to_prior_regulon_top5.npz
    EPOCH=10
    BATCH_SIZE=96
    CCC_MIN_PAIRS=32
    VALIDATION_ARGS=()
    ;;
  dev_brain)
    VARIANT=collectri_min5_u5_k52_b96_e15_ps3
    TRAINER=train_v1.py
    QUERY=${DATA_ROOT}/dev_brain/st_300_dev_brain_human_x20pct.h5ad
    REF=${DATA_ROOT}/dev_brain/sc_dev_brain_human.h5ad
    MASK=${GCAT_DIR}/configs/masks/dev_brain_fold0.npy
    MODULE=${GCAT_DIR}/assets/regulons/dev_brain_collectri_min5_uncovered_denovo46.npy
    NICHE_BRIDGE=${GCAT_DIR}/assets/regulons/dev_brain_nichenet_lr_program_to_prior_regulon_top5.npz
    EPOCH=${EPOCH_OVERRIDE:-15}
    BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-96}
    CCC_MIN_PAIRS=16
    VALIDATION_ARGS=()
    ;;
  ovarian|ovary)
    VARIANT=collectri_min5_u5_k83_b96_e10_ps3
    TRAINER=train_v1.py
    QUERY=${DATA_ROOT}/ovarian_cancer_shared_st.h5ad
    REF=${DATA_ROOT}/ovarian_cancer_shared_sc.h5ad
    MASK=${DATA_ROOT}/ovarian_cancer_5fold_gene.npy
    MODULE=${GCAT_DIR}/assets/regulons/ovarian_collectri_min5_uncovered_denovo21.npy
    NICHE_BRIDGE=
    EPOCH=10
    BATCH_SIZE=96
    CCC_MIN_PAIRS=32
    VALIDATION_ARGS=()
    ;;
  RCC)
    VARIANT=collectri_min5_u5_k61_b96_e10_ps3
    TRAINER=train_v1.py
    QUERY=${DATA_ROOT}/RCC_st.h5ad
    REF=${DATA_ROOT}/RCC_sc.h5ad
    MASK=${DATA_ROOT}/RCC_5fold_gene.npy
    MODULE=${GCAT_DIR}/assets/regulons/RCC_collectri_min5_uncovered_denovo47.npy
    NICHE_BRIDGE=
    EPOCH=10
    BATCH_SIZE=96
    CCC_MIN_PAIRS=32
    VALIDATION_ARGS=()
    ;;
  heart_pcw13)
    VARIANT=collectri_min5_u5_k47_b96_e10_ps3
    TRAINER=train_v1.py
    QUERY=${DATA_ROOT}/heart_pcw13_st.h5ad
    REF=${DATA_ROOT}/heart_pcw13_sc.h5ad
    MASK=${DATA_ROOT}/heart_pcw13_5fold_gene.npy
    MODULE=${GCAT_DIR}/assets/regulons/heart_pcw13_collectri_min5_uncovered_denovo33.npy
    NICHE_BRIDGE=
    EPOCH=15
    BATCH_SIZE=96
    CCC_MIN_PAIRS=8
    VALIDATION_ARGS=()
    ;;
  liver)
    VARIANT=collectri_min5_u5_k54_b112_e10_ps3
    TRAINER=train_v1_validation.py
    QUERY=${DATA_ROOT}/liver/combined_liver_st_v2.h5ad
    REF=${DATA_ROOT}/liver/combined_liver_sc_v2.h5ad
    MASK=${GCAT_DIR}/configs/masks/liver_fold0.npy
    MODULE=${GCAT_DIR}/assets/regulons/liver_collectri_min5_uncovered_denovo41.npy
    NICHE_BRIDGE=
    EPOCH=10
    BATCH_SIZE=112
    CCC_MIN_PAIRS=8
    VALIDATION_ARGS=(--validation_split_mode batch_last_10pct)
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Expected data1, AD, dev_brain, ovarian, RCC, heart_pcw13, or liver." >&2
    exit 2
    ;;
esac

# Final S4 defaults. The canonical multi-fold entry reads configs/benchmarks.json;
# this compatibility launcher keeps the same four selected dataset settings.
FLOW_CELL_ATTENTION_OVERRIDE=${FLOW_CELL_ATTENTION_OVERRIDE:-1}
if [[ "${DATASET}" == "data1" || "${DATASET}" == "dev_brain" ]]; then
  BALANCE_MODALITIES_OVERRIDE=${BALANCE_MODALITIES_OVERRIDE:-1}
fi
if [[ "${DATASET}" == "heart_pcw13" ]]; then
  DROPOUT_OVERRIDE=${DROPOUT_OVERRIDE:-0.0}
fi

# Keep canonical defaults above while allowing controlled tuning runs for every dataset.
EPOCH=${EPOCH_OVERRIDE:-${EPOCH}}
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-${BATCH_SIZE}}
CCC_MIN_PAIRS=${CCC_MIN_PAIRS_OVERRIDE:-${CCC_MIN_PAIRS}}

if [[ -n "${TRAINER_OVERRIDE}" ]]; then
  TRAINER=${TRAINER_OVERRIDE}
fi
if [[ -n "${MASK_OVERRIDE:-}" ]]; then
  MASK=${MASK_OVERRIDE}
fi
if [[ -n "${MODULE_OVERRIDE:-}" ]]; then
  MODULE=${MODULE_OVERRIDE}
fi
if [[ -n "${QUERY_OVERRIDE:-}" ]]; then
  QUERY=${QUERY_OVERRIDE}
fi
if [[ -n "${REF_OVERRIDE:-}" ]]; then
  REF=${REF_OVERRIDE}
fi

EXTRA_ARGS=()
if [[ "${DETERMINISTIC_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--deterministic)
fi
if [[ "${SKIP_FINAL_EVAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_final_eval)
fi
if [[ "${SAVE_MODEL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_model)
fi
if [[ "${USE_DATA_COVARIATES_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_data_covariates)
fi
if [[ "${USE_BATCH_COVARIATE_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_batch_covariate)
fi
if [[ "${BALANCE_MODALITIES_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--balance_modalities)
fi
if [[ -n "${ST_BLOCK_GENE_MASK_RATE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--st_block_gene_mask_rate "${ST_BLOCK_GENE_MASK_RATE_OVERRIDE}")
fi
if [[ -n "${LAMBDA_MASKED_ZMSE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--lambda_masked_zmse "${LAMBDA_MASKED_ZMSE_OVERRIDE}")
fi
if [[ "${MAX_TRAIN_BATCHES_OVERRIDE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--max_train_batches "${MAX_TRAIN_BATCHES_OVERRIDE}")
fi
if [[ "${MAX_EVAL_BATCHES_OVERRIDE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--max_eval_batches "${MAX_EVAL_BATCHES_OVERRIDE}")
fi
if [[ -n "${REG_ACTIVITY_STD_FLOOR_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--reg_activity_std_floor "${REG_ACTIVITY_STD_FLOOR_OVERRIDE}")
fi
if [[ "${REG_LOSS_STOP_BELOW:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--reg_loss_stop_below "${REG_LOSS_STOP_BELOW}")
  EXTRA_ARGS+=(--reg_loss_check_batches "${REG_LOSS_CHECK_BATCHES:-10}")
fi
if [[ "${LOG_BATCH_EVERY_OVERRIDE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--log_batch_every "${LOG_BATCH_EVERY_OVERRIDE}")
fi
if [[ "${CCC_PROGRAM_HEAD_BIAS_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--ccc_program_head_bias)
fi
if [[ "${FLOW_LINEAR_BOTTLENECK_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--flow_linear_bottleneck)
fi
if [[ -n "${FLOW_GRAPH_MASK_PATH_OVERRIDE:-}" ]]; then
  [[ -f "${FLOW_GRAPH_MASK_PATH_OVERRIDE}" ]] || {
    echo "Missing Stage-3 flow graph mask: ${FLOW_GRAPH_MASK_PATH_OVERRIDE}" >&2
    exit 1
  }
  EXTRA_ARGS+=(--flow_graph_mask_path "${FLOW_GRAPH_MASK_PATH_OVERRIDE}")
fi
if [[ "${FLOW_ORACLE_ST_TRUTH_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--flow_oracle_st_truth)
fi
if [[ "${FLOW_STAGE5_OVERRIDE:-0}" == "1" ]]; then
  # Minimal Stage 5 implies flow attention on the last two layers.
  : "${FLOW_CELL_ATTENTION_OVERRIDE:=1}"
  : "${FLOW_ATTN_LAYERS_OVERRIDE:=last2}"
  export FLOW_CELL_ATTENTION_OVERRIDE FLOW_ATTN_LAYERS_OVERRIDE
  EXTRA_ARGS+=(--flow_stage5)
fi
if [[ "${FLOW_CELL_ATTENTION_OVERRIDE:-0}" == "1" ]]; then
  # Stage 4 flow attention.  Default replaces the scalar CCC bias entirely
  # (trailing --no_ccc_bias wins over --use_ccc_bias).  With
  # FLOW_HYBRID_CCC_LAYERS set, the scalar CCC bias is kept on those layers
  # (must be disjoint from the flow layers).
  EXTRA_ARGS+=(
    --flow_cell_attention
    --flow_attn_layers "${FLOW_ATTN_LAYERS_OVERRIDE:-last}"
    --flow_gate_hidden_dim "${FLOW_GATE_HIDDEN_DIM_OVERRIDE:-64}"
    --flow_gate_init "${FLOW_GATE_INIT_OVERRIDE:-0.0}"
    --flow_attn_mode "${FLOW_ATTN_MODE_OVERRIDE:-gated}"
  )
  if [[ -n "${FLOW_HYBRID_CCC_LAYERS:-}" ]]; then
    EXTRA_ARGS+=(
      --ccc_bias_layers "${FLOW_HYBRID_CCC_LAYERS}"
      --ccc_bias_trainable
    )
  else
    EXTRA_ARGS+=(--no_ccc_bias)
  fi
fi
if [[ "${NO_CCC_BIAS_OVERRIDE:-0}" == "1" ]]; then
  # Clean no-CCC comparator (scalar CCC and flow attention both off).
  EXTRA_ARGS+=(--no_ccc_bias)
fi
if [[ "${DISABLE_CELL_TRANSFORMER_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_cell_transformer)
fi
if [[ "${DISABLE_GENE_MODULE_ATTENTION_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_gene_module_attention)
fi
if [[ "${NO_DIST_BIAS_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_dist_bias)
fi
if [[ "${ST_ALIGNED_SC_PRIOR_OVERRIDE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--st_aligned_sc_prior)
fi
if [[ "${GRN_DECODER_RESIDUAL_OVERRIDE:-0}" == "1" ]]; then
  # Signed GRN asset is derived from the module init path by default.
  GRN_SIGN=${GRN_SIGN_PATH_OVERRIDE:-${MODULE%.npy}_signedGRN.npz}
  [[ -f "${GRN_SIGN}" ]] || {
    echo "Missing signed GRN asset: ${GRN_SIGN}" >&2
    exit 1
  }
  EXTRA_ARGS+=(--grn_decoder_residual --grn_sign_path "${GRN_SIGN}")
  if [[ "${GRN_FROZEN_BASELINE_OVERRIDE:-0}" == "1" ]]; then
    [[ -n "${GRN_BASELINE_CKPT:-}" && -f "${GRN_BASELINE_CKPT}" ]] || {
      echo "Missing baseline checkpoint: ${GRN_BASELINE_CKPT:-<unset>}" >&2
      exit 1
    }
    EXTRA_ARGS+=(--grn_frozen_baseline --grn_baseline_ckpt "${GRN_BASELINE_CKPT}")
  fi
fi
NICHE_STAGE=${NICHE_BRIDGE_STAGE_OVERRIDE:-${NICHE_BRIDGE_STAGE:-}}
if [[ "${NICHE_STAGE}" != "off" && ( "${NICHE_REGULON_BRIDGE_OVERRIDE:-0}" == "1" || -n "${NICHE_STAGE}" ) ]]; then
  NICHE_BRIDGE=${NICHE_REGULON_BRIDGE_PATH_OVERRIDE:-${NICHE_BRIDGE}}
  [[ -n "${NICHE_BRIDGE}" && -f "${NICHE_BRIDGE}" ]] || {
    echo "Missing NicheNet bridge matrix for ${DATASET}: ${NICHE_BRIDGE}" >&2
    exit 1
  }
  EXTRA_ARGS+=(
    --niche_regulon_bridge
    --niche_regulon_bridge_path "${NICHE_BRIDGE}"
    --niche_bridge_mode "${NICHE_BRIDGE_MODE_OVERRIDE:-dense}"
    --niche_bridge_stage "${NICHE_STAGE:-pre_encoder}"
  )
fi

MODEL_ARGS=()
for override in \
  HIDDEN_DIM_OVERRIDE \
  NUM_LAYERS_OVERRIDE \
  LATENT_DIM_OVERRIDE \
  DECODER_HIDDEN_DIM_OVERRIDE \
  DECODER_LAYERS_OVERRIDE \
  DROPOUT_OVERRIDE \
  NUM_HEADS_OVERRIDE \
  FFN_MULT_OVERRIDE; do
  if [[ -n "${!override:-}" ]]; then
    option=${override%_OVERRIDE}
    MODEL_ARGS+=("--${option,,}" "${!override}")
  fi
done
OUT=${2:-${GCAT_DIR}/results/${DATASET}/${VARIANT}/fold0}
PREFIX=${PREFIX_OVERRIDE:-gcat_${DATASET}_fold0}
CCC_CACHE_DIR=${CCC_CACHE_DIR_OVERRIDE:-$(dirname "${OUT}")/ccc_cache}
CCC_CACHE_NAME=${CCC_CACHE_NAME_OVERRIDE:-fold0}
CCC_INTERACTION_CACHE=${CCC_ENSEMBL_INTERACTION_CACHE_OVERRIDE:-${GCAT_DIR}/assets/interactions_human__ensembl_lr.pt}
mkdir -p "${OUT}" "${CCC_CACHE_DIR}"

for path in "${QUERY}" "${REF}" "${MASK}" "${MODULE}" \
  "${CONFIG}" \
  "${CCC_INTERACTION_CACHE}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 1; }
done

# The bundled runtime is first so training cannot fall back to repository-level source.
export PYTHONPATH="${GCAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${GCAT_DIR}/results/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

OUTPUT_ARGS=(--save_speed_metrics --save_per_gene_score)
if [[ "${SAVE_PRED_OVERRIDE:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--save_pred)
fi
if [[ "${SKIP_FINAL_METRICS_OVERRIDE:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--skip_final_metrics)
fi
if [[ "${SAVE_MODULE_ANALYSIS_OVERRIDE:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(
    --gene_module_track_every_epochs 1
    --gene_module_track_min_support 5
    --gene_module_track_inactive_mass_fraction 0.1
    --gene_module_export_stats
  )
fi

TRAIN_ARGS=(
  --query_data "${QUERY}"
  --ref_data "${REF}"
  --mask_gene "${MASK}"
  --mask_fold_index "${MASK_FOLD_INDEX_OVERRIDE:-0}"
  --random_seed "${RANDOM_SEED_OVERRIDE:-11}"
  --device "${DEVICE:-cuda:0}"
  --epoch "${EPOCH}"
  --batch_size "${BATCH_SIZE}"
)
TRAIN_ARGS+=("${VALIDATION_ARGS[@]}")
TRAIN_ARGS+=(
  --lr "${LR_OVERRIDE:-5e-4}"
  --weight_decay "${WEIGHT_DECAY_OVERRIDE:-1e-6}"
  --warmup_epochs "${WARMUP_EPOCHS_OVERRIDE:-5}"
  --config "${CONFIG}"
)
TRAIN_ARGS+=("${MODEL_ARGS[@]}")
TRAIN_ARGS+=(
  --prefix "${PREFIX}"
  --out_dir "${OUT}"
  --gene_token_mode module
  --gene_module_num auto
  --gene_module_init_path "${MODULE}"
  --dynamic_gene_modules
  --prior_strength "${PRIOR_STRENGTH_OVERRIDE:-3.0}"
  --off_prior_init_std "${OFF_PRIOR_INIT_STD_OVERRIDE:-0.02}"
  --assignment_temperature "${ASSIGNMENT_TEMPERATURE_OVERRIDE:-1.0}"
  --dist_metric l2
  --coord_norm_mode pc_l2
  --masa_noise_rate 0.0
  --masa_noise_std 0.0
  --masa_mask_rate "${MASA_MASK_RATE_OVERRIDE:-0.1}"
  --drop_node_rate "${DROP_NODE_RATE_OVERRIDE:-0.25}"
  --pair_loss_weight "${PAIR_LOSS_WEIGHT_OVERRIDE:-10.0}"
  --lambda_reg "${LAMBDA_REG_OVERRIDE:-0.1}"
  --lambda_out "${LAMBDA_OUT_OVERRIDE:-0.0}"
  --lambda_pull "${LAMBDA_PULL_OVERRIDE:-0.0}"
  --pull_label_mode "${PULL_LABEL_MODE_OVERRIDE:-both}"
  --pull_target "${PULL_TARGET_OVERRIDE:-latent}"
  --pull_topk "${PULL_TOPK_OVERRIDE:-10}"
  --pull_niche_k "${PULL_NICHE_K_OVERRIDE:-15}"
  --outgoing_lr_std_floor "${OUTGOING_LR_STD_FLOOR_OVERRIDE:-0.1}"
  --lambda_flow "${LAMBDA_FLOW_OVERRIDE:-0.0}"
  --flow_structure "${FLOW_STRUCTURE_OVERRIDE:-full}"
  --flow_supervision "${FLOW_SUPERVISION_OVERRIDE:-source}"
  --amp_mode "${AMP_MODE_OVERRIDE:-fp16}"
  --use_ccc_bias
  --ccc_bias_layers last
  --ccc_bias_init_scale "${CCC_BIAS_INIT_SCALE_OVERRIDE:-1.0}"
  --ccc_bias_trainable
  --ccc_cache_dir "${CCC_CACHE_DIR}"
  --ccc_cache_name "${CCC_CACHE_NAME}"
  --ccc_ensembl_interaction_cache "${CCC_INTERACTION_CACHE}"
  --ccc_topk 64
  --ccc_block_size 512
  --ccc_min_pairs "${CCC_MIN_PAIRS}"
  --ccc_st_candidate_mode spatial
  --ccc_sc_candidate_mode expression
  --ccc_st_candidate_k 0
  --ccc_sc_candidate_k 0
  --ccc_sc_pca_n_components 10
  --eval_every_epochs 0
)
TRAIN_ARGS+=("${OUTPUT_ARGS[@]}")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

"${PYTHON_BIN}" "${GCAT_DIR}/scripts/${TRAINER}" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${OUT}/${PREFIX}.log"
