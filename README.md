# STaCKer: a deep-learning-enabled spatial transcriptomics Common coordinate builder

Establishing a common coordinate framework (CCF) among multiple spatial transcriptome slices is essential for data comparison and integration yet remains a challenge. Here we present a deep learning algorithm STaCker that unifies the coordinates of transcriptomic slices via an image registration process. STaCker derives a composite image representation by integrating tissue image and gene expressions that are transformed to be resilient to noise and batch effects. Trained exclusively on diverse synthetic data, STaCker overcomes the training data scarcity and is applicable to any tissue type. Its performance on various benchmarking datasets shows a significant increase in spatial concordance in aligned slices, surpassing existing methods. STaCker also successfully harmonizes multiple real spatial transcriptome datasets. These results indicate that STaCker is a valuable computational tool for constructing a CCF with spatial transcriptome data.  

## Repository configuration

Below is a tree of the folders in this repository together with descriptions of the code in each folder.

```
stacker/                         Base folder
├── code/                        Source for all code
│   └── libs/                    Source for stacker library functions
├── testing/                     Jupyter notebook testing examples
├── data/                        Example weights for stacker models
├── LICENSE                      License information
├── README.md                    This file, with setup and example code info
├── docker-config.sh             Docker setup script
└── docker-requirements.txt      Python requirements for Docker instances   
```

## Requirements and dependencies

**Please note the following requirements before proceeding.**

STaCKer was tested on a Dockerized container with the following attributes:

* **Operating system(s):** Ubuntu 20.04.4 LTS.
* **Memory:** An environment with at least 16 GB RAM and 6 GM VRAM (GPU RAM) is recommended, but not required.
   * STaCKer may work on systems without GPUs, but performance may be significantly slower. If using Docker, it is recommended to use a system *with* a GPU.
   * Any GPU used must be CUDA-compatible.
* **Program version(s):** 
   * **Python-related:** Python 3.8.10, Tensorflow 2.9; other Python dependencies are given in `docker-requirements.txt`.
   * **Docker:** Although not required, Docker makes the setup and execution of STaCKer very simple. STaCKer was last tested on Docker version 24.0.6.

Note that STaCKer does work in other environments (for example, other versions of Ubuntu); however, its use is not standardized in this case.

## Installation

These installation instructions will assume that the user is running Ubuntu 20.04.4 LTS and is within the current directory to which they will download and install STaCKer.

In the below code snippets, the `$` character denotes lines where the user *inputs* data to the terminal. `$` characters should not be copied from the below code; they are used purely for demonstration. Lines that do not begin with `$` denote where *output* from the terminal should be received.

### Install and configure Docker

1. Install Docker for Ubuntu Linux following the instructions on [the official installation page.](https://docs.docker.com/desktop/install/linux-install/)

2. Pull the appropriate Docker image from the online repository.

   ```sh
   $ docker pull nvcr.io/nvidia/tensorflow:22.07-tf2-py3
   ```

3. Check the Image ID for the newly pulled image.

   ```sh
   $ docker image ls
   REPOSITORY                  TAG             IMAGE ID       CREATED         SIZE
   nvcr.io/nvidia/tensorflow   22.07-tf2-py3   <IMAGE ID>     15 months ago   12.2GB
   ```

4. Create a new Docker container from the image. The following code does so with a number of extra convenience parameters, and it is recommended to copy/paste the below line:

   ```sh
   $ docker run --gpus all -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
      -v ${PWD}:/workspace --name stacker --shm-size=6g --ulimit memlock=-1 <IMAGE ID>
   ```

   Depending on your system's GPU memory size, `--shm-size` may have to be lowered.

   At this point, you should see some output followed by:

   ```sh
   root@9970cc8e1255:/workspace# _
   ```

   This indicates that the Docker container has been started successfully, and your environment is now ready to download and run STaCKer.

To learn more about Docker, including information on how to reuse your newly created Docker container, see the [Docker 101 tutorial.](https://www.docker.com/101-tutorial/)

### Download and run STaCKer

5. From within the Docker container, download the STaCKer files and enter the resulting directory.

   ```sh
   $ git clone https://github.com/regeneron-mpds/stacker.git
   $ cd stacker
   ```

6. Execute the configuration file to install all required dependencies.

   ```sh
   $ ./docker-config.sh
   ```

   Congratulations! You are ready to use STaCKer within your Docker container. You should notice a new `stacker` folder within your original filesystem; you can copy data into this folder to make it appear within the Docker container.

## Demonstrations

Demonstration of how to apply STaCker on data used in this work are provided under `testing/` folder in each **Python (Jupyter) notebook.** Please follow the comments provided in each notebook for guidance. The expected outputs are noted in the notebooks. The run time is typically in the order of seconds. 

### Running the demonstration notebooks

Notebooks must be run from within the Docker container. Luckily, Docker makes this very simple. All that the user needs to do is start a Jupyter server on a specific port:

```sh
$ jupyter lab --ip=0.0.0.0 --port=<PORT>
```

where `PORT` can be 8888, 8787, or 8786. Multiple ports are provided in case one of the ports is busy on the host machine. Once Jupyter is started, users can navigate to localhost:<PORT> in their browser and navigate to the appropriate Jupyter notebook. Make sure to check the Docker container's command-line output for the Jupyter security token.

## License

Please see LICENSE file.
