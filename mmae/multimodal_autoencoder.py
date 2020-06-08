# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Arno Onken
#
# This file is part of the mmae package.
#
# The mmae package is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# The mmae package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This module implements multimodal autoencoders by means of Bregman divergences.

"""
from __future__ import absolute_import
from __future__ import division

import copy
import numpy as np
import six
try:
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.constraints import max_norm
    import tensorflow.keras.backend as K
except ImportError:
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers import Reshape
    from keras.layers import Activation
    from keras.layers import Dropout
    from keras.layers import concatenate
    from keras.models import Model
    from keras.constraints import max_norm
    import keras.backend as K

from . import bregman_divergences


class MultimodalAutoencoder(Model):
    """
    This class represents a multimodal autoencoder.  By default, the central
    fusion network consists of dense layers with ReLU activations connected
    directly to the unimodal input/output layers.  Unimodal networks for each
    modality can be added by overriding the `_construct_unimodal_encoders` and
    `_construct_unimodal_decoders` methods.

    Parameters
    ----------
    input_shapes : dict or list of lists
        Shape of input elements for each modality.
    hidden_dims : int or list of ints
        Number of elements per layer of the encoder.  The last element of the
        array represents the number of elements of the latent representation.
    output_activations : str, callable, dict or list, optional
        Output activation functions for each modality.  (Default: 'linear')
    dropout_rates : float or list of floats, optional
        Dropout rates for each encoder layer.  Corresponding rates will be
        used for the decoder.  No dropout if set to `None`.  (Default: `None`)
    activity_regularizers : Keras regularizer or list of regularizers, optional
        Activity regularizers for each encoder layer.  Corresponding
        regularizers will be used for the decoder.  No activity regularizers
        if set to `None`.  (Default: `None`)

    Attributes
    ----------
    encoder : Keras model
        Encoder model, transforming data to a latent representation.
    decoder : Keras model
        Decoder model, transforming a latent representation to the original
        data space.

    Methods
    -------
    compile(optimizer, loss, loss_weights, sample_weight_mode, target_tensors)
        Sets the model configuration for training.
    fit(data, batch_size, epochs, verbose, callbacks, validation_split,
        validation_data, shuffle, sample_weight, validation_sample_weight,
        initial_epoch, steps_per_epoch, validation_steps)
        Fits the model to the given training data.
    evaluate_reconstruction(data, batch_size, verbose, sample_weight, steps)
        Evaluates reconstruction loss for the given test data.
    train_on_batch(data, sample_weight)
        Applies a single batch update step using the provided training data.
    test_on_batch(data, sample_weight)
        Evaluates reconstruction loss on a single batch of test data.
    encode(data)
        Encodes the data `data` to the latent representation.
    decode(latent_data)
        Decodes the data `latent_data` from the latent representation.

    """
    def __init__(self, input_shapes, hidden_dims, output_activations='linear',
                 dropout_rates=None, activity_regularizers=None,
                 **unimodal_kwargs):
        # Make sure that dropout_rates and activity_regularizers are lists
        if dropout_rates is None:
            # No dropout if no rates specified
            dropout_rates = [0.0] * (len(hidden_dims)+1)
        else:
            if np.isscalar(dropout_rates):
                dropout_rates = [dropout_rates] * (len(hidden_dims)+1)
        if not isinstance(activity_regularizers, list):
            activity_regularizers = [activity_regularizers] \
                    * (len(hidden_dims)+1)
        if isinstance(input_shapes, dict):
            # Extract modality names and convert input_shapes to list
            modality_names = list(input_shapes.keys())
            input_shapes = [input_shapes[name] for name in modality_names]
        else:
            modality_names = None
        # list of input layers, one layer per modality
        input_layers = self._construct_input_layers(input_shapes,
                                                    modality_names)
        # list of unimodal layers before fusion
        um_layers = self._construct_unimodal_encoders(copy.copy(input_layers),
                                                      modality_names,
                                                      **unimodal_kwargs)
        # list of unimodal shapes right before and after fusion
        fusion_shapes = self._get_output_shapes(um_layers)
        mm_layer = self._construct_multimodal_layer(um_layers)
        encoder = self._construct_fusion_encoder(mm_layer, hidden_dims,
                                                 dropout_rates,
                                                 activity_regularizers)
        # Layer holding the encoded representation
        encoded_layer = Input(shape=(hidden_dims[-1],))
        fusion_decoder, fusion_autoencoder \
                = self._construct_fusion_decoder(encoded_layer, encoder,
                                                 hidden_dims, fusion_shapes,
                                                 dropout_rates,
                                                 activity_regularizers)
        output_decoder, output_autoencoder \
                = self._construct_unimodal_decoders(fusion_decoder,
                                                    fusion_autoencoder,
                                                    input_shapes,
                                                    modality_names,
                                                    **unimodal_kwargs)
        output_encoder = encoder
        output_decoder, output_autoencoder \
                = self._add_output_activations(output_activations,
                                               output_decoder,
                                               output_autoencoder,
                                               modality_names)
        # Initialize models
        super(MultimodalAutoencoder, self).__init__(input_layers,
                                                    output_autoencoder)
        self.encoder = Model(input_layers, output_encoder)
        self.decoder = Model(encoded_layer, output_decoder)

    def compile(self, optimizer='adam', loss='gaussian_divergence',
                loss_weights=None, sample_weight_mode=None,
                target_tensors=None):
        """
        Sets the model configuration for training.

        Parameters
        ----------
        optimizer : str, optional
            Name of optimization algorithm.  (Default: 'adam')
        loss : str, callable, dict or list, optional
            Loss functions, including Bregman divergences, for each modality.
            (Default: 'gaussian_divergence')
        loss_weights : dict or list of floats, optional
            Loss weights for each modality.  `None` corresponds to weight `1.0`
            for each modality.  (Default: `None`)
        sample_weight_mode : str, list or dict, optional
            Sample weight mode for each modality.  Each mode can be `None`
            corresponding to sample-wise weighting or 'temporal' for
            timestep-wise weighting.  (Default: `None`)
        target_tensors : tensor, list of tensors or dict, optional
            Target tensors to be used instead of `data` arguments for training.
            (Default: `None`)

        """
        # Replace Bregman divergence strings with actual functions
        loss = self._replace_bregman_strings(loss)
        # For dicts, rename keys to match output keys
        loss = self._rename_output_keys(loss)
        loss_weights = self._rename_output_keys(loss_weights)
        sample_weight_mode = self._rename_output_keys(sample_weight_mode)
        target_tensors = self._rename_output_keys(target_tensors)
        super(MultimodalAutoencoder,
              self).compile(optimizer=optimizer, loss=loss,
                            loss_weights=loss_weights,
                            sample_weight_mode=sample_weight_mode,
                            target_tensors=target_tensors)

    def fit(self, data=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, sample_weight=None, validation_sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        """
        Fits the model to the given training data based on the reconstruction
        loss.  If provided, also evaluates the reconstruction loss on the
        validation data.

        Parameters
        ----------
        data : array_like, optional
            The training data.  Can be a NumPy array, a list of NumPy arrays
            (one for each modality) or a dict with NumPy arrays as values where
            the keys are the modality names.  (Default: `None`)
        batch_size : int, optional
            The number of samples used to calculate a gradient update.  If
            `None` then the Keras default will be used.  (Default: `None`)
        epochs : int, optional
            Number of epochs to train.  (Default: 1)
        verbose : int, optional
            Verbosity mode.  One of 0, 1 or 2.  (Default: 1)
        callbacks : list of Keras callbacks, optional
            Callbacks to be called during training.  (Default: `None`)
        validation_split : float, optional
            The fraction as a value between 0 and 1 of `data` to be used as
            validation data.  (Default: 0.0)
        validation_data : array_like, optional
            The validation data.  Can be a NumPy array, a list of NumPy arrays
            (one for each modality) or a dict with NumPy arrays as values where
            the keys are the modality names.  (Default: `None`)
        shuffle : bool or str, optional
            If true then the samples are shuffled before each epoch.  If set to
            the string 'batch' then samples are shuffled in chunks of
            `batch_size`.  (Default: `True`)
        sample_weight : array_like, optional
            Array of values for each sample in `data` to weight the loss
            function for the corresponding sample.  (Default: `None`)
        validation_sample_weight : array_like, optional
            Array of values for each sample in `validation_data` to weight the
            loss function for the corresponding sample.  (Default: `None`)
        initial_epoch : int, optional
            The epoch index at which to start training.  (Default: 0)
        steps_per_epoch : int, optional
            The number of batches used per epoch.  (Default: `None`)
        validation_steps : int, optional
            The number of batches used for validation.  (Default: `None`)

        Returns
        -------
        history : Keras history
            Performance values recorded for each training epoch.

        """
        target_data = self._rename_output_keys(data)
        if validation_data is not None:
            validation_target_data = self._rename_output_keys(validation_data)
            if validation_sample_weight is None:
                validation_data = (validation_data, validation_target_data)
            else:
                validation_data = (validation_data, validation_target_data,
                                   validation_sample_weight)
        return super(MultimodalAutoencoder,
                     self).fit(x=data, y=target_data, batch_size=batch_size,
                               epochs=epochs, verbose=verbose,
                               callbacks=callbacks,
                               validation_split=validation_split,
                               validation_data=validation_data,
                               shuffle=shuffle,
                               sample_weight=sample_weight,
                               initial_epoch=initial_epoch,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps)

    def evaluate_reconstruction(self, data=None, batch_size=None, verbose=1,
                 sample_weight=None, steps=None):
        """
        Evaluates reconstruction loss for the given test data.

        Parameters
        ----------
        data : array_like, optional
            The test data to be evaluated.  Can be a NumPy array, a list of
            NumPy arrays (one for each modality) or a dict with NumPy arrays as
            values where the keys are the modality names.  (Default: `None`)
        batch_size : int, optional
            The number of samples used to calculate a gradient update.  If
            `None` then the Keras default will be used.  (Default: `None`)
        verbose : 0 or 1, optional
            Verbosity mode.  (Default: 1)
        sample_weight : array_like, optional
            Array of values for each sample in `data` to weight the loss
            function for the corresponding sample.  (Default: `None`)
        steps : int, optional
            The total number of batches used.  (Default: `None`)

        Returns
        -------
        test_loss : list
            A list of test reconstruction loss values for each modality.

        """
        target_data = self._rename_output_keys(data)
        return super(MultimodalAutoencoder,
                     self).evaluate(x=data, y=target_data,
                                    batch_size=batch_size, verbose=verbose,
                                    sample_weight=sample_weight, steps=steps)

    def train_on_batch(self, data, sample_weight=None):
        """
        Applies a single batch update step using the provided training data.

        Parameters
        ----------
        data : array_like
            The training data.  Can be a NumPy array, a list of NumPy arrays
            (one for each modality) or a dict with NumPy arrays as values where
            the keys are the modality names.
        sample_weight : array_like, optional
            Array of values for each sample in `data` to weight the loss
            function for the corresponding sample.  (Default: `None`)

        Returns
        -------
        training_loss : list
            A list of training reconstruction loss values for each modality.

        """
        target_data = self._rename_output_keys(data)
        return super(MultimodalAutoencoder,
                     self).train_on_batch(x=data, y=target_data,
                                          sample_weight=sample_weight)

    def test_on_batch(self, data, sample_weight=None):
        """
        Evaluates reconstruction loss on a single batch of test data.

        Parameters
        ----------
        data : array_like
            The test data.  Can be a NumPy array, a list of NumPy arrays (one
            for each modality) or a dict with NumPy arrays as values where the
            keys are the modality names.
        sample_weight : array_like, optional
            Array of values for each sample in `data` to weight the loss
            function for the corresponding sample.  (Default: `None`)

        Returns
        -------
        test_loss : list
            A list of test reconstruction loss values for each modality.

        """
        target_data = self._rename_output_keys(data)
        return super(MultimodalAutoencoder,
                     self).test_on_batch(x=data, y=target_data,
                                         sample_weight=sample_weight)

    def encode(self, data, batch_size=None, verbose=0, steps=None):
        """
        Encodes the input data to the latent representation.

        Parameters
        ----------
        data : array_like
            The data to be encoded.  Can be a NumPy array, a list of NumPy
            arrays (one for each modality) or a dict with NumPy arrays as
            values where the keys are the modality names.
        batch_size : int, optional
            The number of samples used to calculate a gradient update.  If
            `None` then the Keras default will be used.  (Default: `None`)
        verbose : 0 or 1, optional
            Verbosity mode.  (Default: 0)
        steps : int, optional
            The total number of batches used.  (Default: `None`)

        Returns
        -------
        latent_data : array_like
            The latent representation of the input data.

        """
        return self.encoder.predict(x=data, batch_size=batch_size,
                                    verbose=verbose, steps=steps)

    def decode(self, latent_data, batch_size=None, verbose=0, steps=None):
        """
        Decodes the input data from the latent representation.

        Parameters
        ----------
        latent_data : array_like
            The data to be decoded.
        batch_size : int, optional
            The number of samples used to calculate a gradient update.  If
            `None` then the Keras default will be used.  (Default: `None`)
        verbose : 0 or 1, optional
            Verbosity mode.  (Default: 0)
        steps : int, optional
            The total number of batches used.  (Default: `None`)

        Returns
        -------
        data : array_like
            The full reconstructed representation of the latent data.  This is
            either a NumPy array or a list of NumPy arrays (one for each
            modality).

        """
        return self.decoder.predict(x=latent_data, batch_size=batch_size,
                                    verbose=verbose, steps=steps)

    @classmethod
    def _replace_bregman_strings(cls, loss):
        """
        Replaces strings referring to Bregman divergences with the actual
        divergence instances.

        Parameters
        ----------
        loss : str, callable, dict or list
            The loss functions.

        Returns
        -------
        loss : str, callable, dict or list
            The loss functions where the strings referring to Bregman
            divergences were replaced.

        """
        if isinstance(loss, six.string_types):
            fun = getattr(bregman_divergences, loss, None)
            if fun is not None:
                loss = fun
        elif isinstance(loss, dict):
            for key, value in loss.items():
                loss[key] = cls._replace_bregman_strings(value)
        elif isinstance(loss, list):
            loss = [cls._replace_bregman_strings(value)
                    for value in loss]
        return loss

    @staticmethod
    def _construct_input_layers(input_shapes, modality_names=None):
        """
        Generates input layers for each modality.

        Parameters
        ----------
        input_shapes : list of lists
            Shape of input elements per modality.
        modality_names : list of strings, optional
            The names of the modalities.  If not `None` then this list must
            have the same length as input_shapes.  (Default: `None`)

        Returns
        -------
        input_layers : list of Keras tensor inputs
            The input layers for each modality.  If modality names are
            specified, then each input layer is named after the modality.

        """
        if modality_names is None:
            input_layers = [Input(shape=shape) for shape in input_shapes]
        else:
            input_layers = [Input(shape=input_shapes[i], name=name)
                            for i, name in enumerate(modality_names)]
        return input_layers

    @staticmethod
    def _construct_unimodal_encoders(input_layers, modality_names, **kwargs):
        """
        Returns the first argument.  Override this method to add unimodal
        encoders between the input layers and the fusion network.

        Parameters
        ----------
        input_layers : dict or list of Keras tensor inputs
            The input layers for each modality.
        modality_names : list of strings
            The names of the modalities or `None`.

        Returns
        -------
        um_layers : dict or list of Keras tensor outputs from layers
            The unimodal encoder networks for each modality.

        """
        um_layers = input_layers
        return um_layers

    @classmethod
    def _construct_multimodal_layer(cls, um_layers):
        """
        Constructs the first multimodal layer by flattening and concatenating
        the unimodal input layers.

        Parameters
        ----------
        um_layers : list of Keras tensor outputs from layers
            The unimodal encoder networks for each modality.

        Returns
        -------
        mm_layer : Keras tensor output from layer
            The multimodel layer.

        """
        fusion_shapes = cls._get_output_shapes(um_layers)
        # Flatten unimodal layers for concatenation
        for i, shape in enumerate(fusion_shapes):
            if len(shape) > 2:
                um_layers[i] = Flatten()(um_layers[i])
        # Multimodal layer
        if len(um_layers) > 1:
            mm_layer = concatenate(um_layers)
        else:
            mm_layer = um_layers[0]
        return mm_layer

    @classmethod
    def _construct_fusion_encoder(cls, mm_layer, hidden_dims, dropout_rates,
                                  activity_regularizers):
        """
        Adds a densely connected fusion network for encoding to the latent
        representation from a concatenated multimodal layer.

        Parameters
        ----------
        mm_layer : Keras tensor output from layer
            The multimodel layer.
        hidden_dims : int or list of ints
            Number of elements per layer of the encoder.  The last element of
            the array represents the number of elements of the latent
            representation.
        dropout_rates : list of floats
            Dropout rates for each encoder layer.
        activity_regularizers : list of Keras regularizers
            Activity regularizers for each encoder layer.

        Returns
        -------
        encoder : Keras tensor output from layer
            The encoder network.

        """
        kernel_constraints = cls._get_kernel_constraints(dropout_rates)
        mm_layer = Dropout(dropout_rates[0])(mm_layer)
        encoder = mm_layer
        for i, dim in enumerate(hidden_dims):
            layer = Dense(dim, activation='relu',
                          kernel_constraint=kernel_constraints[i+1],
                          activity_regularizer=activity_regularizers[i+1])
            encoder = layer(encoder)
            encoder = Dropout(dropout_rates[i+1])(encoder)
        return encoder

    @classmethod
    def _construct_fusion_decoder(cls, encoded_layer, encoder, hidden_dims,
                                  fusion_shapes, dropout_rates,
                                  activity_regularizers):
        """
        Adds a densely connected fusion network for decoding from the latent
        representation.

        Parameters
        ----------
        encoded_layer : Keras tensor input
            The latent input layer.
        encoder : Keras tensor output from layer
            The encoder network.
        hidden_dims : int or list of ints
            Number of elements per layer of the decoder used in reverse order
            from second-last element to first element.
        fusion_shapes : list of lists
            The shapes of the unimodal layers for each modality right before
            fusion.
        dropout_rates : list of floats
            Dropout rates for each decoder layer, used in reverse order from
            second-last element to first element.
        activity_regularizers : list of Keras regularizers
            Activity regularizers for each decoder layer, used in reverse order
            from second-last element to first element.

        Returns
        -------
        fusion_decoder : list of Keras tensor outputs from layers
            The fusion decoder networks starting from the latent layer.
        fusion_autoencoder : list of Keras tensor outputs from layers
            The networks up to and including the fusion decoder.

        """
        kernel_constraints = cls._get_kernel_constraints(dropout_rates)
        decoder = encoded_layer
        autoencoder = encoder
        for i in range(len(hidden_dims)-2, -1, -1):
            layer = Dense(hidden_dims[i], activation='relu',
                          kernel_constraint=kernel_constraints[i],
                          activity_regularizer=activity_regularizers[i])
            dropout = Dropout(dropout_rates[i])
            decoder = dropout(layer(decoder))
            autoencoder = dropout(layer(autoencoder))
        fusion_decoder = [None] * len(fusion_shapes)
        fusion_autoencoder = [None] * len(fusion_shapes)
        for i, shape in enumerate(fusion_shapes):
            # No activation and no dropout in unimodal separation layers
            layer = Dense(np.prod(shape[1:]))
            fusion_decoder[i] = Reshape(shape[1:])(layer(decoder))
            fusion_autoencoder[i] = Reshape(shape[1:])(layer(autoencoder))
        return fusion_decoder, fusion_autoencoder

    @staticmethod
    def _construct_unimodal_decoders(fusion_decoder, fusion_autoencoder,
                                     output_shapes, modality_names, **kwargs):
        """
        Returns the first two arguments.  Override this method to add unimodal
        decoders between the fusion network and the output layers.

        Parameters
        ----------
        fusion_decoder : dict or list of Keras tensor outputs from layers
            The fusion decoder networks starting from the latent layer.
        fusion_autoencoder : dict or list of Keras tensor outputs from layers
            The networks up to and including the fusion decoder.
        output_shapes : dict or list of lists
            Shapes of output elements for each modality.
        modality_names : list of strings
            The names of the modalities or `None`.

        Returns
        -------
        output_decoder : dict or list of Keras tensor outputs from layers
            The output decoder networks starting from the latent layer.
        output_autoencoder : dict or list of Keras tensor outputs from layers
            The networks up to and including the unimodal decoders.

        """
        output_decoder = fusion_decoder
        output_autoencoder = fusion_autoencoder
        return output_decoder, output_autoencoder

    @staticmethod
    def _get_output_shapes(layers):
        """
        Extracts the output shapes of the input.

        Parameters
        ----------
        layers : list of Keras tensor outputs from layers
            The network with the output shapes of interest.

        Returns
        -------
        output_shapes : list of lists
            The output shapes in a list.

        """
        output_shapes = [K.int_shape(layer) for layer in layers]
        return output_shapes

    @classmethod
    def _add_output_activations(cls, output_activations, output_decoder,
                                output_autoencoder, modality_names=None):
        """
        Adds output activations to the provided decoder and autoencoder.

        Parameters
        ----------
        output_activations : str, callable, dict or list
            Output activation functions for each modality.
        output_decoder : list of Keras tensor outputs from layers
            The output decoder networks starting from the latent layer.
        output_autoencoder : list of Keras tensor outputs from layers
            The networks up to and including the unimodal decoders.
        modality_names : list of strings, optional
            The names of the modalities.  (Default: `None`)

        Returns
        -------
        output_decoder : list of Keras tensor outputs from layers
            The output decoder networks starting from the latent layer with
            the added output activations.
        output_autoencoder : list of Keras tensor outputs from layers
            The networks up to and including the unimodal decoders with the
            added output activations.

        """
        if not isinstance(output_activations, dict) \
                and not isinstance(output_activations, list):
            # Single output activation supplied for all outputs
            output_activations = [output_activations] * len(output_decoder)
        if modality_names is None:
            if isinstance(output_activations, dict):
                activations = [Activation(output_activations[key],
                                          name=cls._append_output_name(key))
                               for key in output_activations]
            else:
                activations = [Activation(activation)
                               for activation in output_activations]
        else:
            # Attach modality names to activations
            if isinstance(output_activations, dict):
                activations = [Activation(output_activations[name],
                                          name=cls._append_output_name(name))
                               for name in modality_names]
            else:
                activations = [Activation(output_activations[i],
                                          name=cls._append_output_name(name))
                               for i, name in enumerate(modality_names)]
        for i, activation in enumerate(activations):
            output_decoder[i] = activation(output_decoder[i])
            output_autoencoder[i] = activation(output_autoencoder[i])
        return output_decoder, output_autoencoder

    @staticmethod
    def _get_kernel_constraints(dropout_rates):
        """
        Constructs a kernel constraint for each dropout rate.

        Parameters
        ----------
        dropout_rates : list of floats
            Dropout rates for each encoder layer.

        Returns
        -------
        kernel_constraints : array of Keras constraints
            Constraints for layer kernels corresponding to the given dropout
            rates.

        """
        kernel_constraints = np.empty(len(dropout_rates), dtype=object)
        # If we use dropout then constrain the size of network weights
        kernel_constraints[np.abs(dropout_rates) > 1e-3] = max_norm(3)
        return kernel_constraints

    @classmethod
    def _rename_output_keys(cls, structure):
        """
        Checks whether the argument is a dict and if so, appends output labels
        to its keys.

        Parameters
        ----------
        structure : object
            An input object that is modified if it is a dict.

        Returns
        -------
        output_structure : object
            Either the unmodified argument or a dict with the same values and
            keys with output labels appended.

        """
        if isinstance(structure, dict):
            return {cls._append_output_name(key): value
                    for key, value in structure.items()}
        return structure

    @staticmethod
    def _append_output_name(name):
        """
        Appends an output name label to the argument.

        Parameters
        ----------
        name : str
            An input name.

        Returns
        -------
        output_name : str
            The input name with an output label appended.

        """
        return name + '_reconstruction'
