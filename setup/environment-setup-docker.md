# Steps to set up an Docker-enabled STaCker instance

* Step 1: Prepare a GPU empowered computing instance (16 GB RAM, 6 GB VRAM/GPU memory recommended). Operating system tested is Ubuntu 18.04.6 LTS; a Linux-based operating system is recommended as well.

* Step 2: Generate RSA key to allow your machine to authenticate with GitHub password-free:

```
ssh-keygen -t rsa -C "your-email@XXX.com"
cat ~/.ssh/id_rsa.pub
```

* Step 3: Copy/paste RSA key above onto GitHub.

* Step 4: Setup mounting directory and pull the repo.

```
mkdir workspace
cd workspace
git clone https://github.com/regeneron-mpds/stacker.git
```

* Step 5: Pull the appropriate Docker image.

```
docker pull nvcr.io/nvidia/tensorflow:22.07-tf2-py3
```

* Step 6: docker run (spin up our fresh docker environment), note the image id needs to match (docker image ls)

```
docker run --gpus all -it \
-p 8888:8888 -p 8787:8787 -p 8786:8786 \
-v ${PWD}:/workspace --shm-size=2g --ulimit memlock=-1 YOUR_DOCKER_IMAGE_ID
```

* Step 7: Install conda on your system and initialize conda.

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3/
$HOME/miniforge3/bin/conda init
```

Make sure to restart your terminal or open a new bash session by typing `bash`.

* Step 8: Install dependencies with conda, assuming conda is installed:

```
conda env create --name <your desired envname> --file=environment.yaml
```
