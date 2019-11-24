#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=parser

cd  /home/rcf-proj/ef/spangher/source-exploration/models/topic_model

python3.7 process_data_for_tm.py \
  -i ../../data/news-article-flatlist \
  -o data_with_text \
  --use-labels