# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.

# this script shows how to set up a venv for running the tutorial through avenv, for example on the JASMIN notebook server.

# comment/uncomment the env name based on whether you are runing on CPU or GPU
export ENV_NAME=ai4c_hack_nb_cpu
#export ENV_NAME=ai4c_hack_nb_gpu
#export ENV_NAME=pet_dev_nb_gpu_jasmin

# uncomment if you don't have a venv directory
# mkdir ~/venv

#create the venv
python -m venv ~/venv/${ENV_NAME}

# activate the venv
.  ~/venv/${ENV_NAME}/bin/activate

# the next command needs to run in the root directory of the ai4climate hackathon repo. This will be the directory into which you cloned the repository.
cd ~/ai4c_hackathon
pip install -r requirements.txt

# once the environment is setup, we need to "install" it so it is usable in a jupyter notebook
python -m ipykernel install --user --name ${ENV_NAME}

