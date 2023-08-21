python predict.py \
  --model_name_or_path alex2awesome/quote-attribution-qa__big-bird-base \
  --tokenizer_name google/bigbird-roberta-base \
  --outfile scoring.jsonl \
  --dataset_name data_split_annotated_sources.jsonl
