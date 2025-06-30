#!/bin/bash 

#OAR -q besteffort 
#OAR -p gpu
##OAR -p cluster='graffiti'
#OAR -l host=1/gpu=1,walltime=5:00:00
#OAR -O oarlogs/OAR_%jobid%.out
#OAR -E oarlogs/OAR_%jobid%.err 



##OAR -p gpu-16GB AND gpu_compute_capability_major>=5

# display some information about attributed resources
hostname 
nvidia-smi 


# # Check if uv is installed on the machine. If not install uv. 
# # https://docs.astral.sh/uv/getting-started/installation/
# if ! command -v uv &> /dev/null; then
#     echo "uv not found. Installing..."
#     curl -Ls https://astral.sh/uv/install.sh | sh
# else
#     echo "uv is already installed."
# fi

# # sync uv environment
source .venv/bin/activate
# uv sync


# run script

# # TRAIN SCRIPT
# python3 -m src.monai.sadm \
#         --job_type="train" \
#         --experiment="sadm_ctx_crssattn_nodrop_ep1000"



# # Train simple DDPM 
# python3 -m src.monai.ddpm \
#         --experiment="ddpm_ep200"

# # Inference
# python3 -m src.monai.diffae_resnet18 \
#         --job_type="inference" \
#         --workdir="workdir/diffae_resnet18_ep1000" \
#         --experiment="diffae_resnet18_ep1000" \
#         --checkpoint="workdir/diffae_resnet18_ep1000/ckpt/best.pth" \
#         --datasplit="train"


# TRAIN DiffAE - Starmen with Pytorch CLI
python3 main_diffae.py --config configs/diffae_repr_learn.yaml fit
