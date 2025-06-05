#!/bin/bash 

#OAR -q besteffort 
#OAR -p cluster='kinovis'
#OAR -l host=1/gpu=1,walltime=5:00:00
#OAR -O oarlogs/OAR_%jobid%.out
#OAR -E oarlogs/OAR_%jobid%.err 



##OAR -p gpu-16GB AND gpu_compute_capability_major>=5

# display some information about attributed resources
hostname 
nvidia-smi 


# Check if uv is installed on the machine. If not install uv. 
# https://docs.astral.sh/uv/getting-started/installation/
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -Ls https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# sync uv environment
source .venv/bin/activate
uv sync


# run script
# module load conda
# conda activate pytorch_env
python3 -m src.sadm.sadm