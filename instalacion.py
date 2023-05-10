Installation

https://pytorch.org/get-started/locally/

select OS: Mac
select Package: conda
select Python 3.7

if Linux or Windows and want GPU support
--> >select Cuda version 10.1

Install Cuda Toolkit
Development environment for creating high performance GPU-accelerated applications
You need an NVIDIA GPU in your machine:

https://developer.nvidia.com/cuda-downloads

Legacy releases
10.1 update 2
select OS (e.g. Windows 10)

Download and install

# Crear ambiente conda y activarlo

conda create -n pytorch python=3.7
conda activate pytorch

# Instalar pytorch

conda install pytorch torchvision -c pytorch
or with GPU
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Verificacion:

import torch
x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()
