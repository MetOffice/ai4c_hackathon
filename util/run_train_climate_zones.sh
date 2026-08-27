#!/bin/bash -l
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --job-name=climatezones_train_torch

set -e

export CONDA_ENV=/gws/ssde/j25b/ai4climate/environments/ai4c_cli_gpu
export MLFLOW_DIR=$USER_DIR/mlflow
export EXP_DIR=$USER_DIR/experiments

# uncomment if the mlflow directory doe not exist
# mkdir -p ${MLFLOW_DIR}

export MLFLOW_PORT=4455

export AI4C_REPO=$HOME/prog/ai4c_hackathon
cd ${AI4C_REPO}

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

export CONFIG_PATH=$AI4C_REPO/notebooks/config.json

conda activate ${CONDA_ENV}
# conda activate /gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_gpu
python ${AI4C_REPO}/src/ai4c_hack/ClimateZones_Training_Torch.py  --config $CONFIG_PATH --resolution 1.0 --platform jasmin --learning-rate ${LEARNING_RATE} --batch-size ${BATCH_SIZE} --epochs ${NUM_EPOCHS} --model-out-dir $EXP_DIR
