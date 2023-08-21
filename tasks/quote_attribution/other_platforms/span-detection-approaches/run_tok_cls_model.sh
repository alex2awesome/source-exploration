python token_classification_trainer.py \
  --dataset_name data_split_annotated_sources.jsonl \
  --output_dir '/dev/shm/big-bird__token-classification-model' \
  --model_name_or_path google/bigbird-roberta-base \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --do_train \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --num_train_epochs 2 \
  --datum_order shortest-first \
  --freeze_layers 0 1 2 3


python token_classification_trainer.py \
  --dataset_name data_split_annotated_sources_coref_resolved.jsonl \
  --output_dir '/dev/shm/big-bird__token-classification-model__coref_resolved' \
  --model_name_or_path google/bigbird-roberta-base \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --do_train \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --num_train_epochs 2 \
  --datum_order shortest-first \
  --freeze_layers 0 1 2 3



