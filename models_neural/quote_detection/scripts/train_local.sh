cache_dir=/Users/alex/.cache/torch/transformers/named-models
project_dir=/Users/alex/Projects/usc-research/controlled-sequence-gen

model_type=roberta
if [[ $model_type == 'gpt2' ]]
then
  pretrained_model="$cache_dir/gpt2-medium-expanded-embeddings"
  frozen_layers="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
else
  pretrained_model="$cache_dir/roberta-base-expanded-embeddings"
  frozen_layers="0 1 2 3 4 5 6 7 8 9"
fi
##

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python $SCRIPT_DIR/../train.py \
        --model_type $model_type \
        --pretrained_model_path $pretrained_model \
        --experiment baseline_non-sequential \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --local \
        --do_eval \
        --train_data_file "$project_dir/data/polnear-training-data-stage-1.csv" \
        --notes "Flat Discriminator with polnear sentence data" \
        --freeze_transformer \
        --sentence_embedding_method 'attention' \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \