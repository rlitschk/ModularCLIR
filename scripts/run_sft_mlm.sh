PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# language mask (output dir)
LM_DIR=/home/usr/resources/sft/mlm

MODEL=bert-base-multilingual-uncased
LANG=en

mkdir -p src
cp -r $PROJECT_HOME/* ./src/

# Conversion table for bert-base-multilingual-uncased
# task-adapter <-> task-SFTs
#  +----------+-------------------------------+
#  | #params  | (equivalent) reduction factor |
#  +----------+-------------------------------+
#  | 14174208 |                             1 |
#  |  7091712 |                             2 |
#  |  3550464 |                             4 |
#  |  1779840 |                             8 |
#  |   894528 |                            16 |
#  |   451872 |                            32 |
#  +----------+-------------------------------+

# language SFT: reduction-factor of 2 = 7387776 parameters for mbert-uncased, reduction ratio of 0.0441
RF=7387776
EQ_RF=2

STEPS_MIN=10000
STEPS_MAX=100000

python $PROJECT_HOME/src/sft_mlm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikipedia \
    --dataset_config 20200501.$LANG \
    --output_dir $LM_DIR/RF_$EQ_RF/$LANG/ \
    --cache_dir $LM_DIR/RF_$EQ_RF/$LANG/.cache/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --save_steps 10000000 \
    --ft_params_num $RF \
    --freeze_layer_norm \
    --freeze_decoder \
    --full_l1_reg 0.1 \
    --sparse_l1_reg 0.1 \
    --full_ft_min_steps_per_iteration $STEPS_MIN \
    --sparse_ft_min_steps_per_iteration $STEPS_MIN \
    --full_ft_max_steps_per_iteration $STEPS_MAX \
    --sparse_ft_max_steps_per_iteration $STEPS_MAX \
    --full_ft_max_epochs_per_iteration 100 \
    --sparse_ft_max_epochs_per_iteration 100 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --validation_split_percentage 1 \
    --load_best_model_at_end \
    --save_total_limit 2


# EN, DE, FI as above
# for FI, RU:
# --validation_split_percentage 5 
