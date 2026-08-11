#!/bin/bash -l
set -e


# Set to a suitable directory for writing log files (for which you have write permission).
export USER_DIR=/gws/ssde/j25b/ai4climate/users/$USER
export USER_LOG_DIR=${USER_DIR}/log

#uncomment these lines if you do not have a user directory yet
# mkdir $USER_DIR
# mkdir $USER_LOG_DIR

cd ~/prog/ai4c_hackathon/

export STD_OUT_PATH=$USER_LOG_DIR/era5_ae_train_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=$USER_LOG_DIR/era5_ae_train_log_$(date '+%Y%m%d%H%M').err

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH --export USER_DIR util/run_train_climate_zones.sh



