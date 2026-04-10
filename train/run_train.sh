#!/bin/bash
set -ex

MIXTURE=molmoweb
WANDB_PROJECT=your_project
WANDB_ENTITY=your_entity

CHECKPOINT_PATH=MolmoWeb-Pretrained-4B

WEBOLMO_DATA_DIR=datasets/

NUM_GPUS=8
NUM_NODES=1
PORT=2041
SEQ_LEN=10240
GLOBAL_BATCH_SIZE=64
DEVICE_BATCH_SIZE=2
DURATION=500
SAVE_INTERVAL=100
# run evaluation after every EVAL_INTERVAL steps (at the end of training by default)
EVAL_INTERVAL=$DURATION

RUN_NAME="train_${MIXTURE}"
SAVE_FOLDER=checkpoints/${RUN_NAME}

NCCL_BLOCKING_WAIT=1 NCCL_TIMEOUT=1800 \
WANDB_ENTITY=${WANDB_ENTITY} WANDB_PROJECT=${WANDB_PROJECT} \
DATA_DIR=${WEBOLMO_DATA_DIR} MOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} \
WEBOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} \
uv run torchrun -m \
  --nproc-per-node ${NUM_GPUS} \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --rdzv_backend=c10d \
  --rdzv_id=606 \
  --rdzv_conf="read_timeout=600" \
  --rdzv_endpoint=localhost:${PORT} \
  launch_scripts.train ${MIXTURE} ${CHECKPOINT_PATH} \
  --save_folder=${SAVE_FOLDER} \
  --run_name=${RUN_NAME} \
  --global_batch_size ${GLOBAL_BATCH_SIZE} \
  --device_batch_size ${DEVICE_BATCH_SIZE} \
  --device_eval_batch_size ${DEVICE_BATCH_SIZE} \
  --device_inf_batch_size ${DEVICE_BATCH_SIZE} \
  --duration ${DURATION} \
  --seq_len ${SEQ_LEN} \
  --save_interval ${SAVE_INTERVAL} \
  --eval_interval ${EVAL_INTERVAL} \
  --inf_eval_interval ${EVAL_INTERVAL}
