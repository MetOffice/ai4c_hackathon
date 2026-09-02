#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --account=mohc_shared
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name=dscop_era5_ae_data_prep

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

echo "Environment configured to use conda envirnment for cpu"
export CONDA_ENV=${AI4CLIMATE_GWS_DIR}/environments/ai4c_cli_cpu

# currently the output goes to a user directory, you can copy the data to a central location after checking the output.
export DATA_OUTPUT_DIR=${USER_DIR}/data/weatherbench/mlready/

# uncomment tro cretae the output directory
# mkdir -p DATA_OUTPUT_DIR

conda activate ${CONDA_ENV}
cd ~/prog/data_science_cop
cd ml_examples/era5_autoencoder/

python era5_autoencoder_data_prep.py --start-year 1980 --end-year 2016 --data-out-dir  $DATA_OUTPUT_DIR --root-data-dir ${AI4CLIMATE_GWS_DIR}/data/weatherbench/



