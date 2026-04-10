#!/bin/bash
set -ex

CHECKPOINT_PATH=MolmoWeb-4B-Native

MIXTURE=screenspot:test,screenspot_v2:test

WEBOLMO_DATA_DIR=datasets

NUM_GPUS=1
DEVICE_BATCH_SIZE=2

SAVE_FOLDER=results/eval_${CHECKPOINT_PATH}

DATA_DIR=${WEBOLMO_DATA_DIR} MOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} \
WEBOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} \
uv run torchrun -m \
  --nproc-per-node ${NUM_GPUS} \
  launch_scripts.eval ${CHECKPOINT_PATH} ${MIXTURE} \
  --save_dir=${SAVE_FOLDER} \
  --device_batch_size ${DEVICE_BATCH_SIZE} \
  --include_image
