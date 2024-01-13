# Installation

## Step 1: Clone repository

Make sure to have git installed, then run
```shell
git clone https://github.com/DuskNgai/plunet.git
```

## Step 2: Create a virtual environment
Before running any scripts, you should create a virtual Python environment. In these instructions, we use Miniconda for managing your virtual environments, but any alternative like Conda, Mamba, virtualenv, venv, ... should be fine.

If you don't have any, you could install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

Now you can create a new virtual environment using
```shell
conda create --name <env_name> python=3.11
```

In order to use it, you need to activate the environment:
```shell
conda activate <env_name>
```

## Step 3: Install Unet and its dependencies
Move to the folder "unet" (from the cloned repository in Step 1) that contains the "src" folder. Here, run

```shell
cd plunet
pip install .
```

This will install unet and all dependencies required.

## Step 4: Validate installation
As a first check whether the installation was successful, you can run
```shell
plunet
```
This should display the different options you can choose from Unet, like "test" and "train".


# Troubleshooting
Here is a collection of common issues and how to fix them:

- `RuntimeError: The NVIDIA driver on your system is too old (found version 11070). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has  been compiled with your version of the CUDA driver.` 

  The latest Pytorch versions require higher CUDA versions that may not be installed on your system yet. You can either install the new CUDA version or (maybe easier) downgrade Pytorch to a version that is compatible:

  ```shell
  pip uninstall torch torchvision
  pip install torch==2.0.1 torchvision==0.15.2
  ```