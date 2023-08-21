python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-top-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-top-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-top__source-and-text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --use_source_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 8 9 \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention \
  --num_contextual_layer 4 \
  --datum_order shortest-first

if false
then
python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-top-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-top-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-top__text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention


python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-high-perc-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-high-perc-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-high-perc__source-and-text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --use_source_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 8 9 \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention \
  --datum_order shortest-first


python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-high-perc-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-high-perc-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-high-perc__text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention


python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-any-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-any-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-any__source-and-text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --use_source_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention
fi

python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name data/ablated-any-training-data.jsonl \
  --gold_label_dataset_name data/annotated__ablated-any-training-data.jsonl \
  --output_dir /dev/shm/big-bird-base__ablated-any__text \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --report_to wandb \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --model_type sentence \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --num_train_epochs 2 \
  --use_input_ids \
  --platform gcp \
  --gradient_accumulation_steps 10 \
  --freeze_layers 0 1 2 3 4 5 6 7 8 9 10 \
  --freeze_embeddings \
  --sent_pooling_method attention \
  --word_pooling_method attention \
  --context_layer attention \
  --num_contextual_layer 4
