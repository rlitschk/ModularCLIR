#!/bin/bash
CUDA_VISIBLE_DEVICES="2"

# Turning off adapters in the adapter_retrieval.py script (by omitting --train_adapter) defaults to training a vanilla variant, i.e. monoBERT

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_DIR=/home/usr/resource/data/msmarco
OUTPUT_DIR=/home/usr/resources/baselines/monoBERT

mkdir -p ./src/
cp -r $PROJECT_HOME/* ./src/

python $PROJECT_HOME/adapter_retrieval.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 512 \
    --train_file $DATA_DIR/train_sbert.jsonl \
    --validation_file $DATA_DIR/dev_sbert.jsonl \
    --output_dir $OUTPUT_DIR/ \
    --cache_dir $OUTPUT_DIR/.cache/ \
    --do_train \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --eval_steps 25000 \
    --save_steps 25000 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --num_train_epochs 1 \
    --warmup_steps 5000 \
    --log_level info
