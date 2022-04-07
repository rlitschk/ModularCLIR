PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_DIR=/home/usr/resource/data/msmarco
# ranking mask (output dir)
RM_DIR=$(pwd)
# language mask
LM_DIR=/home/usr/resources/sft/mlm/rf_2/en

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

RF_EQ=451872
RF=32

STEPS_MIN=625000
STEPS_MAX=625000

python $PROJECT_HOME/src/sft_retrieval.py  \
  --model_name_or_path bert-base-multilingual-uncased  \
  --max_seq_length 512  \
  --train_file $DATA_DIR/train_sbert.jsonl  \
  --validation_file $DATA_DIR/dev_sbert.jsonl \
  --output_dir $RM_DIR/ \
  --cache_dir $RM_DIR/.cache/ \
  --lang_ft $LM_DIR \
  --do_train  \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir  \
  --learning_rate 2e-5  \
  --save_steps 25000  \
  --eval_steps 25000 \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --num_train_epochs 1 \
  --warmup_steps 5000 \
  --log_level info \
  --freeze_layer_norm  \
  --save_total_limit 2  \
  --ft_params_num $RF_EQ \
  --full_ft_min_steps_per_iteration 187500 \
  --sparse_ft_min_steps_per_iteration $STEPS_MIN \
  --full_ft_max_steps_per_iteration 187500 \
  --sparse_ft_max_steps_per_iteration $STEPS_MAX
