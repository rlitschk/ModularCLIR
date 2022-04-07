PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# output directory of prepare_data.sh
DATA_DIR=/home/usr/resource/data/msmarco
# ranking adapter (output dir)
RA_DIR=/home/usr/resources/adapter/ir
# language adapter
LA_DIR=/home/usr/resources/mlm/rf_2/en/checkpoint-250000/mlm_adapter

MODEL=bert-base-multilingual-uncased
REDUCTION_FACTOR=1
ACTIVATION=relu

mkdir -p ./src/
cp -r $PROJECT_HOME/* ./src/

python $PROJECT_HOME/src/adapter_retrieval.py \
    --overwrite_output_dir \
    --output_dir $RA_DIR/rf_2_$REDUCTION_FACTOR/ \
    --cache_dir $RA_DIR/rf_2_$REDUCTION_FACTOR/.cache/ \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 512 \
    --train_file $DATA_DIR/train_sbert.jsonl \
    --validation_file $DATA_DIR/dev_sbert.jsonl \
    --do_train \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --eval_steps 25000 \
    --save_steps 25000 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --num_train_epochs 1 \
    --warmup_steps 5000 \
    --log_level info \
    --train_adapter \
    --adapter_config pfeiffer \
    --adapter_non_linearity relu \
    --adapter_reduction_factor $REDUCTION_FACTOR \
    --load_lang_adapter $LA_DIR \
    --language en
