python trainer.py \
  --model_name_or_path roberta-base \
  --dataset_name data/news-edits-training-data.jsonl \
  --output_dir /dev/shm/roberta-base__news-edits__source-and-text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_strategy no \
  --num_train_epochs 3 \
  --use_input_ids \
  --use_source_ids


python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-top-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-top__source-and-text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type full-sequence \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_strategy no \
  --num_train_epochs 3 \
  --use_input_ids \
  --use_source_ids \
  --platform gcp