#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=tm-with-labels

cd  /home/rcf-proj/ef/spangher/source-exploration/models/topic_model

python3.7 sampler.py \
  -i input_data_with_labels \
  -o model_with_labels \
  -k 50 \
  -p 8 \
  -t 100 \
  --use-cached