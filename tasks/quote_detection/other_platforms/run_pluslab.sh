
python trainer.py \
  --model_name_or_path google/bigbird-roberta-base \
  --dataset_name training_data.jsonl \
  --do_train \
  --do_eval \
  --output_dir bigbird-roberta-base__vanilla-model \
  --overwrite_output_dir \
  --report_to wandb