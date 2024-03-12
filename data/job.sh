#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=debug-small
#SBATCH --output=/scratch/kerrigan.d/pdpilot/debug/%A_%a.out
#SBATCH --error=/scratch/kerrigan.d/pdpilot/debug/%A_%a.err
#SBATCH --array=0-1%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /home/kerrigan.d/miniconda3/bin/activate

conda env create -f ../environment.yml
conda activate pdpilot-eval

srun python data_models_plots.py -d -i $SLURM_ARRAY_TASK_ID -o /scratch/kerrigan.d/pdpilot/debug
