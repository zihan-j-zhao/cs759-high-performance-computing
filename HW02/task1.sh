#!/usr/bin/env bash
#SBATCH --job-name=HW02-task1
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:01:30
#SBATCH --output=HW02-task1.out
#SBATCH --error=HW02-task1.err

cd $SLURM_SUBMIT_DIR

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

for i in {10..30}
do
	n=$[2**$i]
	echo "N = $n"
	./task1 $[2**$i]
	echo ""
done
