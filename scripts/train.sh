#!/bin/bash
# CG-MLLM training. The shape-latent length grows over four stages (512 -> 1024 -> 2048 -> 4096);
# stage N > 1 starts from the latest checkpoint of stage N - 1.
#
#   RECIPE=multitask  STAGE=1 bash scripts/train.sh   # v0.1: Qwen3-VL-2B, image / 3D understanding + image / text-to-3D
#   RECIPE=hy3d_i2obj STAGE=1 bash scripts/train.sh   # 4B i2o: Qwen3-VL-4B, HY3D-Bench image-to-3D
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT set.
set -euo pipefail

RECIPE=${RECIPE:-multitask}
STAGE=${STAGE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
POINTBERT=${POINTBERT:-models/point_bert_v1.1.pt}

OBJ_LENS=(512 1024 2048 4096)
case $RECIPE in
  multitask)
    LLM=${LLM:-Qwen/Qwen3-VL-2B-Instruct}; OBJ_UND=True
    LRS=(1e-4 1e-4 1e-4 1e-4); CE_WEIGHTS=(1.0 1.0 1.0 1.0); ACCUMS=(1 1 1 1)
    MAX_TOKENS=(20480 20480 20480 20480); EXPECTED_TOKENS=(20000 20000 20000 20000) ;;
  hy3d_i2obj)
    LLM=${LLM:-Qwen/Qwen3-VL-4B-Instruct}; OBJ_UND=False
    LRS=(2e-4 2e-4 1e-4 1e-4); CE_WEIGHTS=(0.25 0.25 0.1 0.1); ACCUMS=(1 1 1 2)
    MAX_TOKENS=(20480 20480 20480 25600); EXPECTED_TOKENS=(20000 20000 20000 25000) ;;
  *) echo "Unknown RECIPE=$RECIPE (multitask | hy3d_i2obj)"; exit 1 ;;
esac

i=$((STAGE - 1))
OBJ_VAE_LEN=${OBJ_LENS[$i]}
EXP_NAME=${RECIPE}_$(basename "$LLM")_${OBJ_VAE_LEN}
RESUME_ARGS=()
if [ "$STAGE" -gt 1 ]; then
  PREV_DIR=results/${RECIPE}_$(basename "$LLM")_${OBJ_LENS[$((i - 1))]}/checkpoints
  PREV_CKPT=${PREV_CKPT:-$PREV_DIR/$(ls "$PREV_DIR" | sort -n | tail -1)}
  echo "Stage $STAGE initializes from $PREV_CKPT"
  RESUME_ARGS=(--resume_from "$PREV_CKPT" --resume_model_only True)
fi

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train/train.py \
  --llm_path "$LLM" \
  --dataset_config_file "data/configs/cgmllm_${RECIPE}.yaml" \
  --obj_vae_path tencent/Hunyuan3D-2.1 \
  --obj_vae_len "$OBJ_VAE_LEN" \
  --obj_und "$OBJ_UND" \
  --und_obj_vae_path "$POINTBERT" \
  --lr "${LRS[$i]}" \
  --ce_weight "${CE_WEIGHTS[$i]}" \
  --gradient_accumulation_steps "${ACCUMS[$i]}" \
  --max_num_tokens "${MAX_TOKENS[$i]}" \
  --expected_num_tokens "${EXPECTED_TOKENS[$i]}" \
  --max_num_tokens_per_sample 8192 \
  --prefer_buffer_before 8192 \
  --num_workers 8 \
  --num_replicate "$NNODES" \
  --num_shard "$GPUS_PER_NODE" \
  --auto_resume True \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --results_dir "results/${EXP_NAME}" \
  --checkpoint_dir "results/${EXP_NAME}/checkpoints" \
  --wandb_name "$EXP_NAME"
