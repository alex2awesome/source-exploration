pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install transformers wandb deepspeed accelerate tensorboard
pip3 install setuptools==59.5.0

#
pip3 install wheel
pip3 install -U 'spacy[cuda-autodetect]'
pip3 install torchvision
pip3 install fastcoref
python -m spacy download en_core_web_sm
# if there are install errors:
pip install trash-cli
SITE_PACKAGES_FOLDER=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
trash $SITE_PACKAGES_FOLDER/typing_extensions-4.4*


gcp_path=gs://usc-data/source-exploration/tasks/quote_attribution/gpt3-finetuning
gsutil cp "$gcp_path/finetune.py" .
gsutil cp "$gcp_path/finetuning_hf_trainer.py" .
gsutil cp "$gcp_path/training_data.jsonl" .
gsutil cp "$gcp_path/ds_config_neo.json" .
gsutil cp "$gcp_path/ds_config_gpt2xl.json" .

gsutil cp "$gcp_path/gpt2-medium-expanded-embeddings.zip" .
unzip gpt2-medium-expanded-embeddings.zip

sudo chmod 777 /dev

python finetune.py \
  --training_data_file training_data.jsonl \
  --pretrained_model_name 'EleutherAI/gpt-j-6B' \
  --platform 'gcp'

# needed setup to do deepspeed
python finetuning_hf_trainer.py \
  --dataset_name training_data.jsonl \
  --model_name_or_path ./gpt2-medium-expanded-embeddings \
  --output_dir output_dir \
  --do_train \
  --do_eval \
  --report_to null \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1


deepspeed --num_gpus=2 finetuning_hf_trainer.py \
  --deepspeed ds_config_neo.json \
  --model_name_or_path EleutherAI/gpt-neo-2.7B \
  --dataset_name short_training_data.jsonl \
  --do_train \
  --do_eval \
  --report_to null \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --fp16 \
  --freeze_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --save_strategy epoch \
  --output_dir output_dir \
  --platform gcp


deepspeed --num_gpus=2 finetuning_hf_trainer.py \
  --deepspeed ds_config_neo.json \
  --model_name_or_path ./gpt2-medium-expanded-embeddings \
  --dataset_name training_data.jsonl \
  --do_train \
  --do_eval \
  --report_to null \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --fp16 \
  --save_strategy epoch \
  --output_dir output_dir \
  --platform gcp