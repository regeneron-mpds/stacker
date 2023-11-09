# STaCKer: a deep-learning-enabled Spatial Transcriptomics Common coordinate builder

Establishing a common coordinate framework (CCF) among multiple spatial transcriptome slices is essential for data comparison and integration yet remains a challenge. Here we present a deep learning algorithm STaCker that unifies the coordinates of transcriptomic slices via an image registration process. STaCker derives a composite image representation by integrating tissue image and gene expressions that are transformed to be resilient to noise and batch effects. Trained exclusively on diverse synthetic data, STaCker overcomes the training data scarcity and is applicable to any tissue type. Its performance on various benchmarking datasets shows a significant increase in spatial concordance in aligned slices, surpassing existing methods. STaCker also successfully harmonizes multiple real spatial transcriptome datasets. These results indicate that STaCker is a valuable computational tool for constructing a CCF with spatial transcriptome data.  


Please note that this final version of the repository was tested on TensorFlow
2.6.0 using Python 3.9. It does work for future versions, but certain classes
(such as the Keras `BackupAndRestore` class) have moved locations within the
TensorFlow library (e.g., from the `tf.keras.callbacks.experimental` namespace
to the `tf.keras.callbacks` namespace). Slight modifications to the codebase
may be required for certain versions of TensorFlow. Additionally, more configuration
might be required to get TensorFlow to properly recognize GPUs on your local
machine.

## Repository configuration

```
stacker/
   code/
      libs/
   testing/
   setup/
   data/
   LICENSE
   README.md
   environment.yaml
   requirements.txt
```

## Installation
*  Hardware requirement:
*  Operating system tested: Ubuntu 18.04.6 LTS
*  Software dependencies: python==3.8.10, numpy==1.21.6, pandas==1.3.5, libopenexr-dev==2.3.0-6build1,
   neurite==0.2, voxelmorph==0.2, opensimplex==0.4.2, scikit-image== 0.21.0, antspyx==0.3.3, SimpleITK== 2.2.1,
   tensorflow==2.9.1+nv22.7, tensorflow-graphics==2021.12.3, tensorflow-addons==0.17.0, tensorflow-datasets==3.2.1,
   tensorflow-metadata==1.9.0,scanpy== 1.9.1, matplotlib==3.5.0, anndata=0.8.0, Pillow==9.2.0

For detailed instructions about how to configure your environment, use scripts, and (eventually)
a breakdown of the pipeline and its use/mechanism, see the `setup/` folder. The installation may take a few hours.

## Demo

Demonstration of how to apply STaCker on data used in this work are provided under `testing/` folder in each **Python notebook (not scripts).** 
Please follow the comments provided in each notebook for guidance. The expected outputs are noted 
in the notebooks. The run time is typically in the order of seconds. 

## License

Please see LICENSE file.
