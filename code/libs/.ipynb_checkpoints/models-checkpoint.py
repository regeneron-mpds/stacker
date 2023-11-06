"""
MODELS

Contains all models and associated model helper methods.

Helper functions will be moved to appropriate locations with the release of v4.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Callable, Collection, List, Optional, Union
import warnings
import tensorflow as tf
import neurite as ne
import numpy as np
import voxelmorph as vxm
from tensorflow.keras import layers as KL, \
                             initializers as KI, \
                             Model

from .utils import (tensors_to_tensorspec,
                    sig2tensors,
                    tensorspec_to_dict,
                    tensors_to_dict,
                    maybe_rename_tensors,
                    maybe_remap_names_tensorspecs,
                    map_dict_keys)

#################
# MODEL CLASSES #
#################

class EnrichedFunctionalModel(Model):
    """
    An extension of the functional keras Model API to allow for the requesting of
    input and output signatures that can be overridden by subclasses.
    
    You should be able to use this method exactly as you would use keras.Model.
    The only difference is that you will now be able to call some additional
    methods from any model constructed using this class.
    """
    
    # get the signature of the method and transform the model's inputs
    def __init__(self,
                 inputs: Collection[tf.Tensor],
                 outputs: Collection[tf.Tensor],
                 dict_inputs: bool = False,
                 dict_outputs: bool = False,
                 fill_missing_dict_inputs: bool = False,
                 outputs_base_name: Optional[str] = None,
                 outputs_name_map: Optional[dict] = None,
                 **kwargs):
        """
        Parameters
        ----------
        inputs : keras.Input or list of keras.Input
            A list of Keras tensors that define the input of the functional model.
        outputs : keras.Input or list of keras.Input
            A list of Keras tensors that are connected to the input tensors by a TensorFlow
            graph.
        dict_inputs : bool, default False
            Whether to accept model inputs in a dictionary form. Good for ease of use.
        dict_outputs : bool, default False
            Whether to output model outputs in a dictionary form. Good for ease of use.
        fill_missing_dict_inputs : bool, default False
            If dict_inputs is True, then this argument determines whether any arguments
            missing in the supplied input dictionary should be auto-filled. This is good
            if you have a model with a large number of inputs that can be omitted
            (filled with zeros), but otherwise it is best practice to leave this False.
        outputs_base_name : str or None
            By default, the output name of tensors (or keys if the output is a dictionary)
            are the name of the 
        """
        
        # Initialize above so properties are accessible
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        
        # Process dict inputs/outputs
        self.dict_inputs = dict_inputs
        self.dict_outputs = dict_outputs
        self.fill_missing = fill_missing_dict_inputs
        self.dict_output_mapping = outputs_name_map
        
        # Process dict outputs
        if outputs_base_name is not None:
            self._output_named_tensor_list = None
            self._output_default_name = outputs_base_name
        else:
            self._output_named_tensor_list = self.outputs
            self._output_default_name = 'tensor'
        
    # Return the input signature with more arguments
    # 'private' helper method to prevent abuse
    def _get_input_signature(self, dict_mode: Optional[bool] = None,
                             custom_inputs: Collection[tf.Tensor] = None):
        inputs = self.inputs if custom_inputs is None else custom_inputs
        dict_mode = dict_mode if dict_mode is not None else self.dict_inputs
        return tensorspec_to_dict(tensorspec_list=tensors_to_tensorspec(tensor_list=inputs)) \
               if dict_mode else tensors_to_tensorspec(tensor_list=inputs)

    def get_input_signature(self):
        """
        Return the input signature for this model. Automatically converts to
        dictionary form if the model takes dictionary inputs.

        Returns
        -------
        list or dict
            The model input signature.
        """
        return self._get_input_signature(dict_mode=None,
                                         custom_inputs=None)
    
    # Set a default name if named tensor list (since base name was specified)
    # Return the output signature with more arguments
    # 'private' helper method to prevent abuse
    def _get_output_signature(self, dict_mode: Optional[bool] = None,
                              custom_outputs: Collection[tf.Tensor] = None):
        outputs = self.outputs if custom_outputs is None else custom_outputs
        dict_mode = dict_mode if dict_mode is not None else self.dict_outputs
        out_tensorspec = tensors_to_tensorspec(
            tensor_list=outputs,
            default_name=(self._output_default_name if self._output_named_tensor_list is None
                          else None))
        if dict_mode:
            out_tensorspec_dict = tensorspec_to_dict(tensorspec_list=out_tensorspec)
            if self.dict_output_mapping is not None:
                out_tensorspec_dict = map_dict_keys(input_dict=out_tensorspec_dict,
                                                    conversion_dict=self.dict_output_mapping)
            return out_tensorspec_dict
        elif self.dict_output_mapping is not None:
            return maybe_remap_names_tensorspecs(tensorspecs=out_tensorspec,
                                                 remap_dict=self.dict_output_mapping)
        else: return out_tensorspec
    
    def get_output_signature(self):
        """
        Return the output signature for this model. Automatically converts to
        dictionary form if the model takes dictionary outputs.

        Returns
        -------
        list or dict
            The model output signature.
        """
        return self._get_output_signature(dict_mode=None,
                                          custom_outputs=None)
        
    def call(self, inputs: Collection[tf.Tensor], *, training: bool, **kwargs) -> Union[List[tf.Tensor], dict]:
        """
        Call the model. See keras.Model.call for details.
        """
        assert not (isinstance(inputs, dict) ^ self.dict_inputs), \
            'Expected a %s input, got %s.' % ('dict' if self.dict_inputs else 'non-dict',
                                              type(inputs).__name__)
        if self.dict_inputs: inputs = sig2tensors(signature=tensors_to_tensorspec(self.inputs),
                                                   tensor_dict=inputs,
                                                   fill_missing=self.fill_missing)
        result = super().call(inputs, **kwargs)
        if self.dict_outputs: 
            result = tensors_to_dict(tensor_list=result,
                                     named_tensor_list=self._output_named_tensor_list,
                                     default_name=self._output_default_name)
            if self.dict_output_mapping is not None:
                result = map_dict_keys(input_dict=result,
                                       conversion_dict=self.dict_output_mapping)
        elif self.dict_output_mapping is not None:
            result = maybe_rename_tensors(tensors=result,
                                          names=self._get_output_signature(dict_mode=False))
        return result

    # def call_seeded(self, inputs, *, training, **kwargs):
    #     raise NotImplementedError('This model has no parameters to seed.')


class SynthMorph(EnrichedFunctionalModel):
  """
  Another iteration of model that draws inspiration from voxelmorph in its construction.
  Siamese for the first portion of it, followed by a traditional UNet.
  """

  class SiameseInput(Model):
    """
    A collection of KL that intakes a list of two images, concatenates them, and converts them into
    an output shape. The output has the same shape as each of the inputs.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'siamese_in',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      # Make the inputs and outputs.
      src_image = KL.Input(shape=input_shape, name='%s_src_image' % name)
      tgt_image = KL.Input(shape=input_shape, name='%s_tgt_image' % name)
      inputs = [src_image, tgt_image]
      num_filters = input_shape[-1]

      # Convolve each layer separately.
      inputs_conv = []
      for input_ in inputs:
        inputs_conv.append(
          KL.Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            name='%s_indep_conv' % input_.name,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
          )(input_)
        )

      # Concatenate the inputs and downconvolve.
      input_stack = KL.Concatenate(
        axis=-1,
        name='%s_cat' % name
      )(inputs_conv)
      outputs = KL.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding='same',
        name='%s_joint_conv' % name,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
      )(input_stack)
    
      # Make the model.
      super().__init__(name=name, inputs=inputs, outputs=outputs)

  class BaseUNet(Model):
    """
    A simple UNet without the final 1x1 conv layer at the end.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'unet',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      input_ = KL.Input(
        shape=input_shape,
        name='%s_input' % name
      )

      # Make generators.
      def create_conv(filter_ct, name):
        return KL.Conv2D(
          filters=filter_ct,
          kernel_size=3,
          padding='same',
          activation=activation,
          name=name,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer
        )
      def create_maxpool(name):
        return KL.MaxPool2D(
          pool_size=2, strides=2,
          name=name
        )
      def create_convtranspose(filter_ct, name):
        return KL.Conv2DTranspose(
          filters=filter_ct,
          kernel_size=2,
          strides=2,
          padding='same',
          name=name,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer
        )
      
      # Make the down stack.
      down_filter_cts = [64, 128, 256, 512, 1024]
      skips = []
      for ct in down_filter_cts:
        x = create_maxpool(
          name='%s_maxpool_down_%d' % (name, ct)
        )(skips[-1]) if skips else input_
        for i in range(2):
          x = create_conv(
            filter_ct=ct,
            name='%s_conv_down_%d_%d' % (name, ct, i)
          )(x)
        skips.append(x)
      
      # Make the up stack.
      x = skips.pop()
      # lo_dim_x = x # cache?
      for ct, skip in reversed(list(zip(down_filter_cts[:-1], skips))):
        x = create_convtranspose(
          filter_ct=ct,
          name='%s_upconv_%d' % (name, ct)
        )(x)
        x = KL.Concatenate(
          axis=-1,
          name='%s_cat_%d' % (name, ct)
        )([x, skip])
        for i in range(2):
          x = create_conv(
            filter_ct=ct,
            name='%s_conv_up_%d_%d' % (name, ct, i)
          )(x)
      
      outputs = x
      super().__init__(inputs=input_, outputs=outputs, name=name)

  class SVFFormer(Model):
    """
    Takes the output of a UNet and turns it into a 2D SVF.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'def_field_former',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      # Make the input and the output shape. Also extract relevant information.
      input_ = KL.Input(
        shape=input_shape,
        name='%s_input' % name
      )

      # Make the 2D deformation map.
      output = KL.Conv2D(
        filters=2,
        kernel_size=2,
        padding='same',
        activation=activation,
        name='%s_conv' % name,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
      )(input_)

      super().__init__(inputs=input_, outputs=output, name=name)

  # Finalize this later.
  def __init__(
    self,
    input_shape: Collection[int],
    name: str = 'sm',
    siamese_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    siamese_bias_initializer: Union[str, Callable] ='zeros',
    siamese_activation: Union[str, Callable] ='relu',
    baseunet_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    baseunet_bias_initializer: Union[str, Callable] ='zeros',
    baseunet_activation: Union[str, Callable] ='relu',
    svfformer_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    svfformer_bias_initializer: Union[str, Callable] ='zeros',
    svfformer_activation: Union[str, Callable, None] =None,
    auxiliary_outputs: Collection[str] = [],
    **kwargs
  ):

    # Form the inputs.
    input_image_src = KL.Input(
      shape=input_shape,
      name='%s_moving_img' % name
    )
    input_image_tgt = KL.Input(
      shape=input_shape,
      name='%s_fixed_img' % name
    )
    inputs = [input_image_src, input_image_tgt]

    # Make the Siamese input layer.
    siamese_output = self.SiameseInput(
      input_shape=input_shape,
      activation=siamese_activation,
      name='%s_siamese_in' % name,
      kernel_initializer=siamese_kernel_initializer,
      bias_initializer=siamese_bias_initializer
    )([input_image_src, input_image_tgt])

    # Run the image through the main model.
    unet_output = self.BaseUNet(
      input_shape=siamese_output.shape[1:],
      activation=baseunet_activation,
      name='%s_unet' % name,
      kernel_initializer=baseunet_kernel_initializer,
      bias_initializer=baseunet_bias_initializer
    )(siamese_output)

    # Turn the output into an SVF.
    svf_output = self.SVFFormer(
      input_shape=unet_output.shape[1:],
      activation=svfformer_activation,
      name='%s_def_field_former' % name,
      kernel_initializer=svfformer_kernel_initializer,
      bias_initializer=svfformer_bias_initializer
    )(unet_output)

    # Integrate via a vxm.layers.VecInt layer.
    def_output = vxm.layers.VecInt(
      indexing='ij',
      method='ss',
      int_steps=7,
      out_time_pt=1,
      name='%s_vec_int' % name
    )(svf_output)
    
    # Determine the inverse via a vxm.layers.VecInt layer.
    inv_def_output = vxm.layers.VecInt(
      indexing='ij',
      method='ss',
      int_steps=7,
      out_time_pt=1,
      name='%s_inv_vec_int' % name
    )(-svf_output)
    
    # Finally, transform the image with a vxm.layers.SpatialTransformer.
    output = vxm.layers.SpatialTransformer(
      interp_method='linear',
      indexing='ij',
      single_transform=False,
      fill_value=None,
      shift_center=True,
      name='%s_spatial_transform' % name
    )([input_image_src, def_output])
    outputs = [output]

    # Make auxiliary outputs available to wrapped classes and such.
    self.references = ne.modelio.LoadableModel.ReferenceContainer()
    self.references.siamese_output = siamese_output
    self.references.unet_output = unet_output
    self.references.svf_output = svf_output
    self.references.def_output = def_output
    self.references.inv_def_output = inv_def_output

    # Include some auxiliary outputs as actual outputs
    for aux_out in auxiliary_outputs:
      outputs.append(getattr(self.references, aux_out))

    super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)


class SynthMorphSemiSupervised(EnrichedFunctionalModel):
  """
  Adds onto SynthMorph by adding the ability to deform models.
  """

  def __init__(
    self,
    img_shape: Collection[int],
    lbl_shape: Collection[int],
    name: str = 'sm_semi',
    input_names: Optional[Collection[str]] = None,
    output_names: Optional[Collection[str]] = None,
    efm_kwargs: dict = {},
    **kwargs
  ):

    # Base model that will give us a segmentation map.
    base_model = SynthMorph(
      input_shape=img_shape,
      name='%s_base' % name,
      **kwargs,
      **efm_kwargs
    )

    # Form the inputs. Draw from the inputs of UNet2D_V4 since they're
    # already taken from there.
    input_label_src = KL.Input(
      shape=lbl_shape,
      name='moving_lbl'
    )
    base_inputs = base_model.inputs
    inputs = base_inputs + [input_label_src]

    # Form the outputs, using the def_output to additionally deform
    # the passed in label image.
    base_outputs = base_model.outputs
    def_output = base_model.references.def_output
    lbl_image_transformed = vxm.layers.SpatialTransformer(
      interp_method='linear',
      indexing='ij',
      single_transform=False,
      fill_value=None,
      shift_center=True,
      name='%s_spatial_transform' % name
    )([input_label_src, def_output])
    outputs = base_outputs + [lbl_image_transformed]

    # Auxiliary outputs. base_model will contain most references already in
    # its own references property (alias provided for convenience)
    self.references = ne.modelio.LoadableModel.ReferenceContainer()
    self.references.base_model = base_model
    self.references.base_model_refs = base_model.references
    
    # Name the inputs and outputs if desired. Makes it easier to map
    # losses and metrics using dicts if you know the names.
    # Otherwise output names are output_1, output_2, ... by default,
    # not sure if similar for inputs.
    # https://datascience.stackexchange.com/questions/74406/custom-output-names-for-keras-model
    # https://stackoverflow.com/questions/54750890/multiple-metrics-to-specific-inputs
    if input_names is not None:
      inputs = {nname: input for nname, input in zip(input_names, inputs)}
    if output_names is not None:
      outputs = {nname: output for nname, output in zip(output_names, outputs)}
   
    # Make a Model that adheres to these specifications.
    super().__init__(inputs=inputs, outputs=outputs, name=name, **efm_kwargs)


class SynthMorphSupervised(EnrichedFunctionalModel):
    """
    Supervised voxelmorh that is trained to align images but whose
    weights are updated based on the loss between corresponding labels. It
    is 'supervised' as it also feeds label images (that directly correspond
    to the input images) directly into the model as input (rather that only
    for loss calculation, as done in the semi-supervised variants of this
    model). 

    An example for two-dimensional inputs:

    * EXPECTED INPUT FORMAT: [src_image, trg_image, src_label, trg_label]
    * EXPECTED INPUT SHAPES: [(B, H, W, C),
                              (B, H, W, C),
                              (B, H, W, LC),
                              (B, H, W, LC)]
      where B is batch size, H is image height, W is image width, C is image
      channels, and LC is number of specified labels.


    The 'simple' part of this supervised model refers to the idea that
    the labels are concatenated along the channel dimension with their respective
    input images and are then mixed down to C number of channels using a 1x1
    convolution.

    Note that the input format for this model differs from that expected for
    the old SynthMorph models.
    """
    def __init__(self,
                 img_shape: Collection[int], # (no channels)
                 lbl_shape: Collection[int], # (no channels)
                 img_feats: int = 1, # channels of source, tgt images
                 lbl_feats: int = 1, # channels of labelmaps
                 name: str = 'sm_super',
                 efm_kwargs: dict = {},
                 **kwargs):
        """ 
        Parameters
        ----------
        inshape : tuple
            Input shape without channel number. This may be up to three
            dimensions. E.g. (192, 192, 192)
        name : str
            Model name - also used as layer name prefix. Default is 'hss-vxm'.
        kwargs
            Passed to VxmDenseSemiSupervisedSeg; see its documentation
            in voxelmorph for more details.
        """

        # get the dimensionality of convolutions
        # commented since SynthMorphSemiSupervised forces 2D images
        #ndims = len(img_shape) # since shapes don't have channels
        #Conv = getattr(KL, 'Conv%dD' % ndims)

        # prepare the base model: make the channel-enabled shapes expected by
        # the base model
        img_shape_ss = tuple(img_shape) + (img_feats,)
        lbl_shape_ss = tuple(lbl_shape) + (lbl_feats,)
        main_net = SynthMorphSemiSupervised(img_shape=img_shape_ss,
                                            lbl_shape=lbl_shape_ss,
                                            name='%s_main' % name,
                                            efm_kwargs={'dict_inputs': False,
                                                        'dict_outputs': False},
                                            **kwargs)
        
        # Make the inputs for the current model
        src_image = KL.Input(shape=img_shape_ss, name='%s_src_image_input' % name)
        trg_image = KL.Input(shape=img_shape_ss, name='%s_trg_image_input' % name)
        src_label = KL.Input(shape=lbl_shape_ss, name='%s_src_label_input' % name)
        trg_label = KL.Input(shape=lbl_shape_ss, name='%s_trg_label_input' % name)
        model_input = [src_image, trg_image, src_label, trg_label]
        
        # Convolve the image/labelmaps down to the number of image channels
        src_input = KL.Conv2D(filters=img_feats,
                         kernel_size=(1,1),
                         padding='same')(KL.Concatenate()([src_image, src_label]))
        trg_input = KL.Conv2D(filters=img_feats,
                         kernel_size=(1,1),
                         padding='same')(KL.Concatenate()([trg_image, trg_label]))
        model_output = main_net([src_input, trg_input, src_label])

        # Initialize the network
        super().__init__(inputs=model_input,
                         outputs=model_output,
                         name=name,
                         **efm_kwargs)
        
        # Add all references. Not very many due to constraints
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references = main_net.references


class HyperSemiSupervisedVxmDense(EnrichedFunctionalModel):
    """
    Deprecated.
    """

    def __init__(self,
                 inshape: Collection[int],
                 nb_labels: int,
                 nb_hyp_params: int = 1,
                 nb_hyp_KL: int = 6,
                 nb_hyp_units: int = 128,
                 name: str = 'hss-vxm',
                 output_refs: Optional[Collection[str]] = None,
                 regularization_parameter: Optional[float] = None,
                 efm_kwargs: dict = {},
                 **kwargs):
        """ 
        Parameters
        ----------
        inshape : tuple
            Input shape without channel number. This may be up to three
            dimensions. E.g. (192, 192, 192)
        nb_hyp_params : int
            Number of input hyperparameters.
        nb_hyp_KL : int
            Number of dense KL in the hypernetwork.
        nb_hyp_units : int
            Number of units in each dense layer of the hypernetwork.
        name : str
            Model name - also used as layer name prefix. Default is 'hss-vxm'.
        kwargs
            Passed to VxmDenseSemiSupervisedSeg; see its documentation
            in voxelmorph for more details.
        """

        # This is deprecated but still included in case it is useful in the future.
        warnings.warn('HyperSemiSupervisedVxmDense is deprecated. Its use is not recommended.')

        # build hypernetwork
        self.hyp_input_name = '%s_hyp_input' % name
        hyp_input = KL.Input(shape=[nb_hyp_params], name=self.hyp_input_name)
        hyp_last = hyp_input
        for n in range(nb_hyp_KL):
            hyp_last = KL.Dense(nb_hyp_units,
                                    activation='relu',
                                    kernel_initializer=KI.RandomNormal(mean=0,
                                                                       stddev=0.055, # this controls inital state
                                                                       seed=None),
                                    bias_initializer='zeros',
                                    name='%s_hyp_dense_%d' % (name, n + 1))(hyp_last)
        # initialize hypernetwork
        hypernet = Model(name='hypernet', inputs=hyp_input, outputs=hyp_last)

        # Initialize the current network with relevant parameters
        base = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=inshape,
                                                      nb_labels=nb_labels,
                                                      hyp_model=hypernet,
                                                      **kwargs)
        inputs = base.inputs
        outputs = base.outputs
        self.references = base.references
        self.references.hyp_model = hypernet
        self.references.hyp_input = hyp_input
        self.references.efm_kwargs = efm_kwargs
        
        self.regularization = regularization_parameter

        # Append any desired variables to the output
        if output_refs is not None:
            outputs.extend([getattr(self.references, nm) for nm in output_refs])
            
        super().__init__(inputs=inputs, outputs=outputs, name=name, **efm_kwargs)
        
        # get hyp input index and name
        self.hyp_input_index = np.where(np.char.endswith(np.array([t.name for t in self.inputs]),
                                                    '_hyp_input'))[0].item()
        self.hyp_input_name = self.inputs[self.hyp_input_index].name
    
    # intercept the hyperparameter input
    def _insert_predefined_hyperparameter(self, inputs: Union[dict, Collection[tf.Tensor]]):
        # Make sure that we're not erroneously calling this function
        assert self.regularization is not None, "'regularization' cannot be None if inserting " \
            "a predefined hyperparameter."
        assert len(inputs) == len(self.inputs) - 1, \
            'Detected %d inputs instead of %d, likely trying to use a custom hyperparameter. ' \
            'Set regularization parameter to None first.' % (len(inputs), len(self.inputs) - 1)
        
        # Get variables
        dict_inputs = self.references.efm_kwargs.get('dict_inputs', False) # using dicts?
        batch_shape = (next(iter(inputs.values())) if dict_inputs else inputs[0]).shape[0] # batch dimension?
        
        # Make a predefined hyperparameter and insert it into the inputs.
        predefined_hyperparameter = tf.zeros(
            shape=(batch_shape,) + self.inputs[self.hyp_input_index].shape[1:]) + self.regularization
        if dict_inputs: inputs[self.hyp_input_name] = predefined_hyperparameter
        else: inputs.insert(self.hyp_input_index, predefined_hyperparameter)
        
        return inputs

    def _get_input_signature(self, **kwargs):
        if self.regularization != None:
            kwargs.pop('custom_inputs', None)
            custom_inputs = [ip for i, ip in enumerate(self.inputs) if i != self.hyp_input_index]
            return super()._get_input_signature(custom_inputs=custom_inputs)
        else: return super()._get_input_signature(**kwargs)
    
    def get_input_signature(self):
       return self._get_input_signature()
    
    # call is omitted, this should be pure function
  
    def call_seeded(self, inputs: Collection[tf.Tensor], training: bool = False, **kwargs):
        if self.regularization is not None: inputs = self._insert_predefined_hyperparameter(inputs)
        #print(inputs)
        return self.call(inputs, training=training, **kwargs)
