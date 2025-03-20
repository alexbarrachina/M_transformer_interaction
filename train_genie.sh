#!/bin/bash
#SBATCH -J tgenie
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:quadro:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR_LOGS.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

source activate tgenie
python train.py