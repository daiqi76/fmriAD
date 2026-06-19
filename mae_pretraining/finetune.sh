#!/bin/bash -l


#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=finetune_mae

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

module load python
conda activate mae_venv


python finetune.py --fold 0 --plane axial --seed 42

