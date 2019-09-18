#!/bin/bash

#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -t 10:00:00 # time required, here it is 1 min

nrows=45527
worker_num=12
rows_per_worker="$((nrows / worker_num))"
echo "nrows: $nrows, nworkers: $worker_num, rows per worker: $rows_per_worker"

## 
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

# launch jobs
for ((  i=0; i<$worker_num; i++ ))
do
 node_i=${nodes_array[$i]}
 echo "STARTING WORKER $i at $node_i"
 srun \
 	--nodes=1 \
 	--ntasks=1 \
 	-w $node_i \
 	python3.7 data-processing-runner.py \
 		--start "$((i * rows_per_worker))" \
 		--end "$(((i + 1) * rows_per_worker))" &
 sleep 5
done