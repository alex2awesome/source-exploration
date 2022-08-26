katie tensorboard run \
  --identities bcs=aspangher-cluster-test \
  --job-name test \
  --tensorflow-framework tensorflow-1.14-python-3.7 \
  --log-dir s3://aspangher/source-exploration/logs \
  --node-size Custom \
  --memory 20G \
  --cores 1

#  --namespace s-ai-classification \