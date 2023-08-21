# generic
if false
then
python qa_trainer.py \
  --dataset_name data_split_annotated_sources.jsonl \
  --output_dir '/dev/shm/big-bird__qa-model' \
  --model_name_or_path google/bigbird-roberta-base \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --do_train \
  --do_eval \
  --label_names start_positions end_positions \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --num_train_epochs 2
fi

# coref resolved
if false
then
python qa_trainer.py \
  --dataset_name data_split_annotated_sources_coref_resolved.jsonl \
  --output_dir '/dev/shm/big-bird__qa-model__coref-resolved' \
  --model_name_or_path google/bigbird-roberta-base \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --do_train \
  --do_eval \
  --label_names start_positions end_positions \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --num_train_epochs 2
fi

# salience
if false
then
python qa_trainer.py \
  --dataset_name data_split_annotated_sources.jsonl \
  --output_dir '/dev/shm/big-bird__salience-model__augmented-data' \
  --model_name_or_path google/bigbird-roberta-base \
  --model_type salience \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --do_train \
  --do_eval \
  --label_names start_positions end_positions \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --num_train_epochs 2
fi

# loss window
if false
then
python qa_trainer.py \
  --dataset_name data_split_annotated_sources.jsonl \
  --output_dir '/dev/shm/big-bird__loss-window-2' \
  --model_name_or_path google/bigbird-roberta-base \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --do_train \
  --do_eval \
  --label_names start_positions end_positions \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --loss_window 2 \
  --num_train_epochs 2
fi


if false
then
# big bird
python qa_trainer.py \
  --dataset_name data_split_annotated_sources.jsonl \
  --output_dir '/dev/shm/big-bird__qa-model__roberta-large' \
  --model_name_or_path google/bigbird-roberta-large \
  --per_device_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --evaluation_strategy steps \
  --eval_steps 3000 \
  --do_train \
  --do_eval \
  --label_names start_positions end_positions \
  --overwrite_output_dir \
  --save_strategy epoch \
  --attention_type original_full \
  --platform gcp \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --freeze_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
fi