#!/bin/bash -l
set -e

# This script need to run from the root directory of the data science cop repository, so update the path below accordingly
export AI4C_HACKATHON_REPO_DIR=~/prog/ai4c_hackathon
cd ${AI4C_HACKATHON_REPO_DIR}

export AI4CLIMATE_GWS_DIR=/gws/ssde/j25b/ai4climate/
export USER_DIR=${AI4CLIMATE_GWS_DIR}/users/$USER

# Set to a suitable directory for writing log files (for which you have write permission).
export USER_LOG_DIR=${USER_DIR}/log/

#uncomment these lines if you do not have a user directory yet
# mkdir $USER_DIR
# mkdir $USER_LOG_DIR

export STD_OUT_PATH=${USER_LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=${USER_LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').err
export JOBNAME=era5_ae_train_$(date '+%Y%m%d%H%M')

echo Writing logs to $STD_OUT_PATH and $STD_ERR_PATH

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH -J ${JOBNAME} --export=ALL,USER_DIR,AI4CLIMATE_GWS_DIR,AI4C_HACKATHON_REPO_DIR util/run_era5_ae_train.sh 



