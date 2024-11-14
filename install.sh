conda create -n evocell python=3.10 

# Activate the newly created conda environment
source activate evocell || conda activate evocell

# Check if the conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "evocell" ]; then
    echo "Failed to activate the uvicorn conda environment. Exiting."
    exit 1
fi

pip install -r ./evocell/requirements.txt

conda install ipykernel -y
python -m ipykernel install --user --name evocell --display-name "Python ([evocell])"


