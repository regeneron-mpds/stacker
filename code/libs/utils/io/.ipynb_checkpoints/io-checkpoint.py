import numpy as np
from os import scandir
import requests
from matplotlib.pyplot import imsave
from pathlib import Path
import os

import tensorflow as tf
import warnings
from ..utils import normalize
from ..constants import EXAMPLE_DESCRIPTION, EXAMPLE_OUT_TYPES, EXAMPLE_OUT_SHAPES

# Legacy inclusions that haven't changed
from ....v1.utils.io import (download_file,
                             yield_moving_fixed_slices,
                             is_scalar,
                             byte_feature,
                             int64_feature,
                             float_feature,
                             create_example,
                             tf_synthetic_record_iterator,
                             serialized_example_to_tensors,
                             ensure_shapes,
                             remove_nans_from_dict)

# hypermorph network and model i/o

def generate_tensorspec(tensor_collection, name='collection'):
    """NEEDS DOCUMENTATION. """
    if '__len__' not in dir(tensor_collection):
        raise ValueError('tensor_collection and its elements must be tf.Tensor objects '
                         'or enumerable collections.')
    return [tf.TensorSpec.from_tensor(t, name='%s_%d' % (name, i))
            if isinstance(t, tf.Tensor) else generate_tensorspec(t, name='%s_%d' % (name, i))
            for i, t in enumerate(tensor_collection)]

def np_save_fp(file, *args, **kwargs):
    """NEEDS DOCUMENTATION. np.save with a returned filepath."""
    np.save(file, *args, **kwargs)
    return file

def pil_save_fp(img, fp, *args, **kwargs):
    """NEEDS DOCUMENTATION. np.save with a returned filepath."""
    img.save(fp, *args, **kwargs)
    return fp

def preproc_fn(input, depth=None):
    """
    A preprocessing function that takes an input dictionary and normalizes/
    preprocesses all the values to align with the specifications called for
    by models.

    Images are normalized to [0, 1] and labels are one-hot encoded using either
    an inferred or specified depth.

    Parameters
    ----------
    input : dict
        Dictionary containing tensors to preprocess.
    depth : int or None
        The depth of the one-hot-encoded labelmaps.

    Returns
    -------
    output : dict
        A dictionary containing the preprocessed versions of the tensors in
        the input dictionary.
    """
    output = {}
    for nm, val in input.items():
        if nm.endswith('_img'): output[nm] = normalize(val,
                                                       min_value=0,
                                                       max_value=1,
                                                       lib=tf.experimental.numpy)
        elif nm.endswith('_lbl'): output[nm] = tf.one_hot(indices=val,
                                                          depth=depth if depth is not None
                                                                else tf.math.reduce_max(val) + 1,
                                                          dtype=tf.float32)
        else: output[nm] = val
    return output

def map2vxm(input, hparam_default=0.0):
    """
    Map inputs from a dictionary to a format compatible with the input of a model:

    ```
    [input, expected_output, sample_weights] where

    input = [moving_img, fixed_img, hparams, moving_lbl]
    expected_output = [fixed_img, 0]
    sample_weights = [1, hparams, 1]
    ```

    If 'hparams' is not included in the dataset, the associated parameter
    will be set to 'hparam_default'. This means the model will behave as
    if any corresponding loss/other coefficients are OMITTED (since hparam_default
    is 0) unless hparam_default is changed. This is for backwards-compatibility
    purposes. Use augment_dataset to set custom hparam values.

    Parameters
    ----------
    input : dict
        The dictionary containing relevant tensor values.
    include_lbl : bool
        Whether to include labels in the returned structure. If not, return 
        a zero tensor for the moving label. Default is True.
    
    Returns
    -------
    tuple
        Tensors arranged in the appropriate format.
    """
    return ((input['moving_img'],
             input['fixed_img'],
             input.get('hparams', tf.constant((hparam_default,))),
             input['moving_lbl']),
            (input['fixed_img'],
             tf.zeros((1,)),
             input['fixed_lbl']),
            (tf.ones((1,)),
             input.get('hparams', tf.constant((hparam_default,))),
             tf.ones((1,)),))

def load_dict_dataset(filepath,
                      example_description=EXAMPLE_DESCRIPTION,
                      out_types=EXAMPLE_OUT_TYPES,
                      out_shapes=EXAMPLE_OUT_SHAPES):
    """
    Load a TFRecord dataset in a format that is compatible with training a model.

    Parameters
    ----------
    filepath : str
        Filepath where the TFRecord is stored.
    depth : int or None
        Depth to use when creating one-hot-encoded labelmaps. Default is None,
        which infers the size from the file (might error for labelmaps of varying
        depth).
    example_description : dict or None
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information. Default value is supplied
        by utils.constants.
    out_types : dict or None
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types. Default value is supplied
        by utils.constants.
    out_shapes : dict or None
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode. Default value is supplied
        by utils.constants.
    
    Returns
    -------
    tf.data.Dataset
        A dataset consisting of returned dictionary elements. No preprocessing is applied.
    """
    # load serialized
    dsser = tf.data.TFRecordDataset(filepath) 

    # deserialize and make types uniform
    ds = dsser.map(lambda x: serialized_example_to_tensors(x,
                                                            example_description,
                                                            out_types,
                                                            out_shapes))
    return ds

def standardize_dtypes(tdict):
    """
    NEED DOCUMENTATION. Standardize datatypes of dataset.
    """
    return {nm: tf.cast(val,
                        dtype=tf.int32 if nm.endswith('_lbl')
                                       else tf.float32)
            for nm, val in tdict.items()}

def load_dataset(filepath,
                 depth=None,
                 example_description=EXAMPLE_DESCRIPTION,
                 out_types=EXAMPLE_OUT_TYPES,
                 out_shapes=EXAMPLE_OUT_SHAPES):
    """
    Load a TFRecord dataset in a format that is compatible with training a model.

    Parameters
    ----------
    filepath : str
        Filepath where the TFRecord is stored.
    depth : int or None
        Depth to use when creating one-hot-encoded labelmaps. Default is None,
        which infers the size from the file (might error for labelmaps of varying
        depth).
    example_description : dict or None
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information. Default value is supplied
        by utils.constants.
    out_types : dict or None
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types. Default value is supplied
        by utils.constants.
    out_shapes : dict or None
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode. Default value is supplied
        by utils.constants.
    
    Returns
    -------
    tf.data.Dataset
        A dataset consisting of elements mapped to Hypermorph networks.
    """
    # Preprocess and return
    ds = load_dict_dataset(filepath=filepath,
                           example_description=example_description,
                           out_types=out_types,
                           out_shapes=out_shapes)
    dspre = ds.map(standardize_dtypes)
    dspre = dspre.map(lambda x: preproc_fn(x, depth=depth))
    dspre = dspre.map(map2vxm)
    return dspre

def augment_hparam_dataset(dataset, hp):
    """
    Augment a Hypermorph-capable dataset by changing the hyperparameter
    values associated with it.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A dataset containing examples in the format output by `map2vxm`.
    hp : numeric or scipy.stats distribution
        A number to use for all hyperparameter values or a random distribution
        that will be sampled to generate new hyperparameter values.
    """
    def gen_fn():
        for datum in dataset:
            input, output, weight = [list(elem) for elem in datum]
            hp_shape = input[2].shape
            curr_vals = hp.rvs(size=hp_shape) if 'rvs' in dir(hp) else hp
            input[2] = tf.ones(input[2].shape, dtype=input[2].dtype) * tf.convert_to_tensor(curr_vals, dtype=input[2].dtype)
            weight[1] = tf.ones(weight[1].shape, dtype=weight[1].dtype) * tf.convert_to_tensor(curr_vals, dtype=weight[1].dtype)
            yield tuple([tuple(elem) for elem in [input, output, weight]])
    return tf.data.Dataset.from_generator(gen_fn, output_signature=dataset.element_spec)
