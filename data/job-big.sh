#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=pdpilot-big-all
#SBATCH --output=results/big/output/%A_%a.out
#SBATCH --error=results/big/output/%A_%a.err
#SBATCH --array=0-36
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /work/vis/users/kerrigan.d/miniforge3/bin/activate
conda activate pdpilot-eval

srun python data_models_plots.py -g big -i $SLURM_ARRAY_TASK_ID -o results/big
