#!/bin/bash -l


#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=9:00:00
#SBATCH --job-name=pretrain_and_finetune

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

module load python
conda activate mae_venv

python pretrain.py --mask_ratio 0.75 --plane all --seed 42
python finetune.py --fold 0 --plane all --seed 42

