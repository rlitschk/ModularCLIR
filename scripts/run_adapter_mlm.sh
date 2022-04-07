PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# language adapter (output dir)
LA_DIR=/home/usr/resources/adapter/mlm

MODEL=bert-base-multilingual-uncased
LANG=en
REDUCITON_FACTOR=2
ACTIVATION=relu
CONFIG=pfeiffer+inv

mkdir -p ./src
cp -r $PROJECT_HOME/* ./src/

python $PROJECT_HOME/src/adapter_mlm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.$LANG \
    --output_dir $LA_DIR/rf_$REDUCITON_FACTOR/$LANG/ \
    --cache_dir $LA_DIR/rf_$REDUCITON_FACTOR/$LANG/.cache/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_steps 250000 \
    --save_steps 25000 \
    --eval_steps 25000 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_seq_length 256 \
    --validation_split_percentage 1 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --log_level info \
    --evaluation_strategy steps \
    --train_adapter \
    --adapter_config $CONFIG \
    --adapter_non_linearity $ACTIVATION \
    --adapter_reduction_factor $REDUCITON_FACTOR \
    --fp16 \
    --fp16_backend amp
