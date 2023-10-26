#!/usr/bin/env bash
#SBATCH --job-name=HW07-task1-cub
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=HW07-task1-cub.out
#SBATCH --error=HW07-task1-cub.err

cd $SLURM_SUBMIT_DIR

for i in {10..20}
do
        n=$[2**$i]
        echo "N = $n"
        ./task1_cub $n 
        #compute-sanitizer --tool memcheck ./task1_cub $n 10
        echo ""
done

