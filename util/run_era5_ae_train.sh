#!/bin/bash -l
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=08:50:00
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --job-name=ai4c_era5_ae_train

set -e


# user directories
# USER_DIR should be defined through sbatch
if [[ ! -v USER_DIR ]]; then
    echo "Error: USER_DIR is not defined." >&2
    exit 1
fi
export USER_EXP_DIR=$USER_DIR/experiments

#uncomment these lines if you do not have a user experiments directory yet
# mkdir $USER_EXP_DIR

if [[ ! -v AI4CLIMATE_GWS_DIR ]]; then
    echo "Error: AI4CLIMATE_GWS_DIR is not defined." >&2
    exit 1
fi

if [[ ! -v AI4C_HACKATHON_REPO_DIR ]]; then
    echo "Error: AI4C_HACKATHON_REPO_DIR is not defined." >&2
    exit 1
fi

# Use nvidia-smi to check whether GPUs are present and select appropriate environment
if nvidia-smi &> /dev/null; then
    export COMPUTE="gpu"
else
    export COMPUTE="cpu"
fi

echo "Environment configured to use conda envirnment for $COMPUTE"
export CONDA_ENV=${AI4CLIMATE_GWS_DIR}/environments/ai4c_cli_${COMPUTE}

export MLFLOW_DIR=${USER_DIR}/mlflow
export MLFLOW_PORT=4455

cd ${AI4C_HACKATHON_REPO_DIR}

export WEATHERBENCH_NORM_DIR=${AI4CLIMATE_GWS_DIR}/data/weatherbench/mlready/norm/

export CONFIG_PATH=${AI4C_HACKATHON_REPO_DIR}/notebooks/config.json

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

conda activate ${CONDA_ENV}

python src/ai4c_hack/train_era5_autoencoder.py --config-path ${CONFIG_PATH} --model-out-dir ${USER_EXP_DIR} --batch-size=${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --learning-rate ${LEARNING_RATE} --data-dir ${WEATHERBENCH_NORM_DIR}  # --mlflow-url "http://localhost" --mlflow-port ${MLFLOW_PORT}

echo cleaning cache dir $DATA_CACHE_DIR
rm -rf $DATA_CACHE_DIR
