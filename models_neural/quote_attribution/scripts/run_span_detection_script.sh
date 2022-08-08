DEFAULT_REPO='git+https://bbgithub.dev.bloomberg.com/aspangher/source-finding.git'
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
  pretrained_model='roberta-base-expanded-embeddings'
  frozen_layers="0 1 2 3 4 5 6 7 8 9"
fi
##

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module models_neural.quote_attribution.train \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_model_path $pretrained_model \
        --experiment roberta_qa \
        --batch_size 1 \
        --num_train_epochs 3 \
        --train_data_file data/our-annotated-data__stage-2.tsv \
        --notes "Stage 2: Quote Attribution. Second run, our dataset only." \
        --freeze_transformer \
        --sentence_embedding_method 'attention' \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --spacy_model_file spacy/en_core_web_lg \


#        --tensorboard-log-dir hdfs:///user/aspangher/source-finding/tensorboard \

#     models_neural/quote_detection/stage_one/model_runner
#        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/source-finding/tensorboard \
#        --pretrained_files_s3 $pretrained_model \
#        --freeze_encoder_layers $frozen_layers \
#        --bidirectional \
#        --use_positional \
#        --use_doc_emb \
#        --doc_embed_arithmetic \
#    ;
#            --train_data_file_s3 data/news-discourse-training-data.csv \
#        --concat_headline \



#for i in version_0  version_1  version_10 version_11 version_12 version_13 version_14 version_15 version_16 version_17 version_18 version_19 version_2  version_20 version_21 version_22 version_23 version_24 version_25 version_26 version_27 version_28 version_29 version_3  version_30 version_31 version_32 version_33 version_34 version_35 version_36 version_37 version_38 version_39 version_4  version_40 version_41 version_42 version_43 version_44 version_45 version_46 version_47 version_48 version_49 version_5 version_6  version_7  version_8  version_9;
#do
#  katie hdfs --identity ai-clf-dob2-gen --namespace s-ai-classification rename /projects/ai_classification/aspangher/controlled-sequence-gen/tensorboard/default/$i /projects/ai_classification/aspangher/controlled-sequence-gen/tensorboard-old/default/$i
#done
