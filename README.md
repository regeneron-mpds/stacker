# STaCKer: a deep-learning-enabled spatial transcriptomics registration framework

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
   .gitignore
   README.md
   environment.yaml
   requirements.txt
```

## Setup

For instructions about how to configure your environment, use scripts, and (eventually)
a breakdown of the pipeline and its use/mechanism, see the `setup/` folder.

## Documentation

Documentation is provided under `testing/` folder in each **Python notebook (not scripts).** 
Please follow the comments provided in each notebook for guidance as to what
the notebook is doing. A brief snippet is provided at the top of each
non-archived notebook to explain its purpose as well.

## License

Please see LICENSE file.
