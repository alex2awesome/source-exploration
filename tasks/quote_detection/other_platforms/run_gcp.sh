pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install transformers wandb deepspeed accelerate tensorboard evaluate sentencepiece
pip3 install setuptools==59.5.0


gcp_path=gs://usc-data/source-exploration/tasks/quote_detection
gsutil cp "$gcp_path/training_data.jsonl" .
gsutil cp "$gcp_path/trainer.py" .
gsutil cp "$gcp_path/model.py" .

sudo chmod 777 /dev

export WANDB_API_KEY=d4d7db5cd8fd3c332d55a5c429108d2ec5eb7b8d
wandb init

python trainer.py \
   --model_name_or_path google/bigbird-roberta-base \
   --dataset_name training_data.jsonl \
   --do_train \
   --do_eval \
   --output_dir bigbird-roberta-base__vanilla-model \
   --overwrite_output_dir \
   --report_to wandb \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1