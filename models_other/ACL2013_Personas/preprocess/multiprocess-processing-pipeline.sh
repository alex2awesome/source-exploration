#!/bin/bash
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH -t 10:00:00 # time required

cd /home/rcf-proj/ef/spangher/source-exploration/models/ACL2013_Personas/preprocess
mkdir -p logs

worker_num=12

## 
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

# launch jobs
for (( i=0; i<$worker_num; i++ ))
do
 node_i=${nodes_array[$i]}
 echo "STARTING WORKER $i at $node_i"
 srun \
 	--nodes=1 \
 	--ntasks=1 \
 	-w $node_i \
 	-e logs/logfile__$i.err \
	 	bash pipeline.sh $i &
 sleep 5
done
wait