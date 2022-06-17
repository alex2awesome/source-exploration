katie tensorboard run \
  --identities hadoop=aspangher-cluster-test \
  --job-name test \
  --tensorflow-framework tensorflow-1.14-python-3.7 \
  --log-dir hdfs:///user/aspangher/source-finding/tensorboard/default \
  --node-size Custom \
  --memory 20G \
  --cores 1



# 2022-06-17 web ui: https://tbsb6c3153a2b21919c7a5cfe564f75745-aspangher.ds-pw-dev02.bce.dev.bloomberg.com/