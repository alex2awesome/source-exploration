#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=parser

cd  /home/rcf-proj/ef/spangher/source-exploration/models/topic_model

python3.7 parser.py \
  -i ../../data/news-article-flatlist \
  -o input_data_with_labels \
  --use-labels