@echo off

:: Create the conda environment
conda create -n evocell python=3.10 -y

:: Activate the newly created conda environment
CALL conda activate evocell

:: Check if the conda environment is activated
IF "%CONDA_DEFAULT_ENV%" NEQ "evocell" (
    echo Failed to activate the evocell conda environment. Exiting.
    EXIT /B 1
)

:: Install Python packages from requirements.txt
pip install -r evocell\requirements.txt

:: Install ipykernel
conda install ipykernel -y
python -m ipykernel install --user --name evocell --display-name "Python ([evocell])"

echo Installation completed.

