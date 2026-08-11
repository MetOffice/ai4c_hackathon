#!/bin/bash -l
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --job-name=era5_autoencoder_training

set -e

# conda activate /gws/ssde/j25a/mmh_storage/ai4c_conda/ai4c_cli_gpu
conda activate /gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_gpu

cd ~/prog/ai4c_hackathon/

python src/ai4c_hack/ERA5_autoencoder.py --config-path notebooks/config.json --model-out-dir  $MOHC_USER/era5_autoencoder --batch-size=8 --num-epochs 20 --learning-rate 0.005



