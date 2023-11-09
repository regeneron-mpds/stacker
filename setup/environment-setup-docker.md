# Steps to set up an Docker-enabled STaCker instance

* Step 1: Prepare GPU empowered computing node (GPU, Memory)
* Step 2: login to instance change user

```
sudo su ubuntu
```

* Step 3: Generate RSA key

```
ssh-keygen -t rsa -C "your-email@XXX.com"
cat ~/.ssh/id_rsa.pub
```

* Step 4: Copypasta rsa key above onto github
* Step 5: Setup mounting directory and pull the repo        
```
mkdir workspace
cd workspace
git clone https://github.com/regeneron-mpds/stacker.git
```

* Step 6: docker pull (Update the image to latest when you pull)

```
docker pull nvcr.io/nvidia/tensorflow:22.07-tf2-py3
```

* Step 7: docker run (spin up our fresh docker environment), note the image id needs to match (docker image ls)
```
docker run --gpus all -it \
-p 8888:8888 -p 8787:8787 -p 8786:8786 \
-v ${PWD}:/workspace --shm-size=2g --ulimit memlock=-1 YOUR_DOCKER_IMAGE_ID
```

* Step 8: Install dependencies with conda, assuming conda is installed:

```
conda env create --name <envname> --file=environments.yaml
```
