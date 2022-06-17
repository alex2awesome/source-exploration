DEFAULT_REPO='git+https://bbgithub.dev.bloomberg.com/aspangher/controlled-sequence-gen.git'
DEFAULT_BRANCH='master'
DEFAULT_PACKAGE=$DEFAULT_REPO@$DEFAULT_BRANCH

DEFAULT_JOB_SIZE='Custom'
#DEFAULT_FRAMEWORK='pytorch-1.6-python-3.7'
DEFAULT_FRAMEWORK='python-3.7-rhel-cuda-10.2'
DEFAULT_GIT_IDENTIY='spectro-oauth-aspangher'
DEFAULT_HADOOP_IDENTITY='aspangher-cluster-test'
DEFAULT_BCS_IDENTITY='aspangher-cluster-test'
ENV=bb
## gpus
num_nodes=1
num_gpus_per_node=1
if [[ $num_nodes -gt 1 ]]
then
  APPROACH='distributed-pytorch'
  worker_args="--node-num-gpus $num_gpus_per_node --num-workers $num_nodes --node-num-cores 4 --node-memory 60G"
else
  APPROACH='single'
  worker_args="--node-num-gpus $num_gpus_per_node --node-num-cores 4 --node-memory 60G"
fi

model_type=roberta
if [[ $model_type == 'gpt2' ]]
then
  pretrained_model='gpt2-medium-expanded-embeddings'
  frozen_layers="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
else
  pretrained_model='roberta-base'
  frozen_layers="0 1 2 3 4 5 6 7 8 9"
fi
##

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module discriminator.model_runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/source-finding/detection/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --finetuned_lm_file './trial-Finetuning on 30,000 corpus__epoch=09-perplexity=16.57.ckpt' \
        --experiment contextual_flatmap_wordsplit \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --train_data_file_s3 data/news-discourse-processed.tsv \
        --notes "Flattened sentences, word split, large-corpus Fine-tuned LM" \
        --freeze_transformer \
        --context_layer 'gpt2-sentence' \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --sentence_embedding_method 'attention' \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --max_num_word_positions 2048