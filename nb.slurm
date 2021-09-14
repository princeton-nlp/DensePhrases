#!/bin/bash
#SBATCH --partition general
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 16G
#SBATCH --gres gpu:1 
#SBATCH --time 1-0:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log


# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i10000-60000 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=cycles
#cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.cs.princeton.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.cs.princeton.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# uncomment the following two lines to use your conda environment called notebook_env
# module load miniconda
# source activate notebook_env

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter notebook --no-browser --port=${port} --ip=${node}
