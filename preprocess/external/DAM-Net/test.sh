#!/usr/bin/env bash
dataset_name="data"
model="DAM-Net"

echo Which PYTHON: `which python`
python eyebrow_test.py \
--config=config/${model}.toml \
--checkpoint=$(pwd)/checkpoints/${model}/best_model.pth \
--image-dir=$(pwd)/${dataset_name} \
--output=$(pwd)/${dataset_name}.${model}_predict