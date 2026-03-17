# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.

# this script shows how to set up a venv for running the tutorial through avenv, for example on the JASMIN notebook server.

# comment/uncomment the env name based on whether you are runing on CPU or GPU
export ENV_NAME=ai4c_hack_nb_cpu
#export ENV_NAME=ai4c_hack_nb_gpu

# you will need to change this variable, as the JASMIN user name is not available as an environment variable through the notebook server
export JASMIN_USER=myusername
export USER_VENV_DIR=/gws/ssde/j25a/mmh_storage/ai4c_user/$JASMIN_USER/

# uncomment if you don't have a venv directory
mkdir $USER_VENV_DIR 

#create the venv
python -m venv $USER_VENV_DIR/${ENV_NAME}

# activate the venv
.  $USER_VENV_DIR/${ENV_NAME}/bin/activate

# the next command needs to run in the root directory of the ai4climate hackathon repo. This will be the directory into which you cloned the repository.
cd ~/ai4c_hackathon
export CACHE_DIR=/work/scratch-nopw2/${JASMIN_USER}/cache
mkdir -p $CACHE_DIR

export TMPDIR=$CACHE_DIR
pip install --cache-dir $CACHE_DIR -r requirements.txt

# once the environment is setup, we need to "install" it so it is usable in a jupyter notebook
python -m ipykernel install --user --name ${ENV_NAME} 

