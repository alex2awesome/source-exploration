python score_new_articles.py \
  --detection_model alex2awesome/quote-detection__roberta-base__background-excluded \
  --detection_tokenizer roberta-base \
  --do_detection \
  --detection_outfile data_with_detection.jsonl \
  --attribution_model babbage:ft-isi-nlp-2023-01-12-06-58-08 \
  --attribution_outfile test.jsonl \
  --dataset_name data_to_score.jsonl


python score_new_articles.py \
  --detection_outfile data_with_detection.jsonl \
  --do_attribution \
  --n_docs 10 \
  --start_idx 100 \
  --attribution_model babbage:ft-isi-nlp-2023-01-12-06-58-08 \
  --attribution_outfile test.jsonl \
  --dataset_name data_to_score.jsonl