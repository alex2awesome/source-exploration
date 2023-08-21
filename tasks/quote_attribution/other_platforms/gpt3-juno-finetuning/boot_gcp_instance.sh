gcloud beta compute instances create instance-1 \
  --project=usc-research \
  --zone=us-central1-c \
  --machine-type=a2-ultragpu-1g \
  --network-interface=network-tier=PREMIUM,subnet=default \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --service-account=520950082549-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
  --accelerator=count=1,type=nvidia-a100-80gb \
  --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20221215-debian-10,mode=rw,size=50,type=projects/usc-research/zones/us-central1-c/diskTypes/pd-balanced \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --reservation-affinity=any \
  --threads-per-core=1 \
  --visible-core-count=1



gcloud compute instances create gpuserver \
   --project usc-research \
   --zone us-west1-b \
   --custom-cpu 12 \
   --custom-memory 78 \
   --maintenance-policy TERMINATE \
   --image-family pytorch-1-7-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --metadata "install-nvidia-driver=True" \
   --accelerator="type=nvidia-tesla-v100,count=1" \
   --preemptible


#   c0-deeplearning-common-cu113-v20221215-debian-10