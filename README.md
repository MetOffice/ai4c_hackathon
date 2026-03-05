# AI4Climate Hackathon
This repository contains the tutorial material for the AI4Climate Hackathon in South Africa in March 2026.

## Getting Started

## Cloning the repository 
Run the the clone command from the command line: `git clone git@github.com:MetOffice/ai4c_hackathon.git`

## Set up environments

There are 2 options for setup, either using conda or venvs. It is important to note that whichever option you are using, you need to run on the platform where the environment will be used to ensure the correct version of the machine learning libraries are installed. In particular if you want to run with a GPU, then you need to have installed 

### Conda
Use the `requirements.yaml` file as follows:
`conda env create --file requirements.yaml`

### venv
User the `requirements.txt` file as follows:
- create vitual env `python -m venv ~/venv/ai4c_hack/`
- activate the venv `. ~/venv/ai4c_hack/bin/activate`
- install the libraries `pip install -r requirements.txt`

Alternatively you can use the helper script `install_venv_notebook.sh` which runs these commands together.

### Notebook access
Once you have created the environment, you may need to run the command to make it available to a Jupyter Lab or Hub server:
`python -m ipykernel install --user --name ai4c_hack`
