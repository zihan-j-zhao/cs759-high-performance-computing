#!/usr/bin/env bash
#SBATCH --job-name=HW08-task3-ts
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --output=HW08-task3-ts.out
#SBATCH --error=HW08-task3-ts.err

cd $SLURM_SUBMIT_DIR

for i in {1..10}
do
	ts=$((2**$i))
	echo "ts = $ts"
	./task3 1000000 8 $ts
	echo ""
done

