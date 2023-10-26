#!/usr/bin/env bash
#SBATCH --job-name=HW07-task1-thrust
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=HW07-task1-thrust.out
#SBATCH --error=HW07-task1-thrust.err

cd $SLURM_SUBMIT_DIR

for i in {10..20}
do
        n=$[2**$i]
        echo "N = $n"
        ./task1_thrust $n 
        #compute-sanitizer --tool memcheck ./task1_thrust $n 10
        echo ""
done

