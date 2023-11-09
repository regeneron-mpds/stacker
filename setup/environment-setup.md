# Environment setup

This file contains the basic procedure for setting up a suitable environment
and also contains some troubleshooting guidelines if things go wrong.

## Setting up the repository

Conda is preferred for setting up the repository as it can install
non-Python packages needed for running. However, some manual configuration
still may need to be done (see **Troubleshooting**).

To set up an acceptable repository, please import the environment specified
within `environment.yaml` in the root directory. This will install conda-
and pip-related dependencies. After that, you should be good to go.

## Troubleshooting

### Nonexistent packages

**If there are problems installing `voxelmorph`, `pystrum`, or `neurite`,
install the following using pip.** You'll need to execute these one-by-one
or make a `requirements.txt` file.

```
git+https://github.com/adalca/pystrum.git
git+https://github.com/adalca/neurite.git
git+https://github.com/voxelmorph/voxelmorph.git
```

### Errors with `cudnn`, `cudatoolkit`, and `PIL`

If you install the libraries yourself, you also need to make sure your `LD_LIBRARY_PATH`
is configured correctly. (You can configure this to run when your conda environment of
choice is activated; see [activate scripts with conda](https://docs.conda.io/projects/conda-build/en/latest/resources/activate-scripts.html).)
Set this path to

```
LD_LIBRARY_PATH=<PATH TO CONDA LIB>:<PATH TO CONDA CUDATOOLKIT LIB>
```

For example,

```
LD_LIBRARY_PATH=/data/home/peter.lais/.conda/envs/imreg/lib:/data/home/peter.lais/.conda/pkgs/cudatoolkit-11.1.74-h6bb024c_0/lib
```

Then, if you're getting errors about a lack of recognition of `cudnn`, please **manually**
create symlinks that point from conda's `cudatoolkit` folder (example above) to conda's `cudnn` folder:

```
(snippet of output from ls-ing cudatoolkit lib folder, cudnn paths are relative to lib folder)

libcudnn.so -> ../../cudnn-8.1.0.77-h90431f1_0/lib/libcudnn.so.8.1.0
libcudnn.so.8 -> ../../cudnn-8.1.0.77-h90431f1_0/lib/libcudnn.so.8.1.0
```

### Errors with `openexr`

Sometimes, `openexr` will have issues with dependencies if you install
it via conda. If using Ubuntu, you can execute

```
sudo apt install libopenexr-dev
```

to install the necessary C++ dependencies. Please look online if you do
not have apt.

If you like using conda, you can install `openexr` (the C++ dependencies)
and then `openexr-python` (the Python wrapper). Both steps are necessary
for effectiveness.
