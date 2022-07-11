<img align="right" alt="" src="d2torch.gif" height="200px"/>

# d2torch

This small project provides a foundation on which to build object detection models in Diablo II: Resurrected. It uses [PyTorch](https://pytorch.org/) as a backend; the current model is `FasterRCNN` with a `mobilenet_v2` backbone. The animation shown depicts the model performance after 10 epochs of training and ca. 5 minutes of training time (NVIDIA GTX 1080 Ti).

## Setup

d2torch is written in Python. We prefer to use [Anaconda](https://www.anaconda.com/) (specifically miniconda) for managing the build environment. You will also need:
- [Diablo II: Resurrected](https://diablo2.blizzard.com/en-us/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Microsoft Build Tools for Visual Studio (C++ desktop)](https://visualstudio.microsoft.com/downloads/)

From the cloned repository:

    conda env create -f environment.yml
    conda activate d2torch
    python .\src\train.py

With Diablo II:Resurrected running (in windowed mode):

    python .\src\infer.py
