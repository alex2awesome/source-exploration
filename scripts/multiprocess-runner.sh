#!/bin/bash
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=5G
#SBATCH -t 10:00:00 # time required, here it is 1 min

cd /home/rcf-proj/ef/spangher/source-exploration/scripts

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
 start_idx="$((i * rows_per_worker))"
 end_idx="$(((i + 1) * rows_per_worker))"
 echo "STARTING WORKER $i at $node_i"
 srun \
 	--nodes=1 \
 	--ntasks=1 \
 	-w $node_i \
 	-o logs/logfile__$start_idx-$end_idx.out \
 	-e logs/logfile__$start_idx-$end_idx.err \
 	python3.7 data-processing-runner.py \
 		--start $start_idx \
 		--end $end_idx &
 sleep 5
done
wait