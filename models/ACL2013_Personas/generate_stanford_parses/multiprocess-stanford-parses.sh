#!/bin/bash
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH -t 10:00:00 # time required

cd /home/rcf-proj/ef/spangher/source-exploration/models/ACL2013_Personas/generate_stanford_parses

nrows=45527
worker_num=12
rows_per_worker="$((nrows / worker_num))"
echo "nrows: $nrows, nworkers: $worker_num, rows per worker: $rows_per_worker"

## 
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

# launch jobs
for (( i=0; i<$worker_num; i++ ))
do
 node_i=${nodes_array[$i]}
 start_idx="$((i * rows_per_worker))"
 end_idx="$(((i + 1) * rows_per_worker))"
 echo "STARTING WORKER $i at $node_i"
 srun \
 	--nodes=1 \
 	--ntasks=1 \
 	-w $node_i \
 	-e logs/logfile__$start_idx-$end_idx.err \
	 	source run_parses.sh -s $start_idx -e $end_idx -b $i
 sleep 5
done
wait