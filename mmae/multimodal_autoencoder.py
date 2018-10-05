# -*- coding: utf-8 -*-
# Copyright (C) 2018 Arno Onken
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
This module implements a multimodal autoencoder by means of Bregman
divergences.

"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import six
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
    fusion network consists of Dense layers with ReLU activations connected
    directly to the unimodal input/output layers.  Unimodal networks for each
    modality can be added by overriding the `_construct_unimodal_encoders` and
    `_construct_unimodal_decoders` methods.  The autoencoder can be trained by
    means of the `fit` method inherited from Keras `Model`.

    Parameters
    ----------
    input_shapes : sequence of lists
        Shape of input elements per modality.
    hidden_dims : int or sequence of ints
        Number of elements per layer of the encoder.  The last element of the
        array represents the number of elements of the latent representation.
    output_activations : str, callable or array_like, optional
        Output activation functions for each modality.  (Default: 'linear')
    losses : str, callable or array_like, optional
        Loss functions, including Bregman divergences, for each modality.
        (Default: 'gaussian_divergence')
    loss_weights : sequence of floats, optional
        Loss weights for each modality.  `None` corresponds to weight `1.0`
        for each modality.  (Default: `None`)
    optimizer : str, optional
        Name of optimization algorithm.  (Default: 'adam')
    dropout_rates : float or sequence of floats, optional
        Dropout rates for each encoder layer.  Corresponding rates will be
        used for the decoder.  No dropout if set to `None`.
        (Default: `None`)

    Attributes
    ----------
    encoder : Keras model
        Encoder model, transforming data to a latent representation.
    decoder : Keras model
        Decoder model, transforming a latent representation to the original
        data space.

    Methods
    -------
    encode(data)
        Encodes the data `data` to the latent representation.
    decode(latent_data)
        Decodes the data `latent_data` from the latent representation.

    """
    def __init__(self, input_shapes, hidden_dims, output_activations='linear',
                 losses='gaussian_divergence', loss_weights=None,
                 optimizer='adam', dropout_rates=None,
                 **unimodal_kwargs):
        # Replace Bregman divergence strings with actual functions
        losses = self._replace_bregman_strings(losses)
        # No dropout if no rates specified
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(hidden_dims)+1)
        else:
            if np.isscalar(dropout_rates):
                dropout_rates = [dropout_rates] * (len(hidden_dims)+1)
        # List of input layers, one layer per modality
        input_layers = [Input(shape=shape) for shape in input_shapes]
        # List of unimodal layers before fusion
        um_layers = self._construct_unimodal_encoders(list(input_layers),
                                                      **unimodal_kwargs)
        # List of unimodal shapes right before and after fusion
        fusion_shapes = self._get_output_shapes(um_layers)
        # Flatten unimodal layers for concatenation
        for i, shape in enumerate(fusion_shapes):
            if len(shape) > 2:
                um_layers[i] = Flatten()(um_layers[i])
        # Multimodal layer
        if len(um_layers) > 1:
            mm_layer = concatenate(um_layers)
        else:
            mm_layer = um_layers[0]
        encoder = self._construct_fusion_encoder(mm_layer, hidden_dims,
                                                 dropout_rates)
        # Layer holding the encoded representation
        encoded_layer = Input(shape=(hidden_dims[-1],))
        fusion_decoder, fusion_autoencoder \
                = self._construct_fusion_decoder(encoded_layer, encoder,
                                                 hidden_dims, fusion_shapes,
                                                 dropout_rates)
        output_decoder, output_autoencoder \
                = self._construct_unimodal_decoders(fusion_decoder,
                                                    fusion_autoencoder,
                                                    input_shapes,
                                                    **unimodal_kwargs)
        output_encoder = encoder
        # Attach output activations
        if not isinstance(output_activations, (list, tuple, np.ndarray)):
            output_activations = [output_activations] * len(input_shapes)
        output_activations = [Activation(activation)
                              for activation in output_activations]
        for i, activation in enumerate(output_activations):
            output_decoder[i] = activation(output_decoder[i])
            output_autoencoder[i] = activation(output_autoencoder[i])
        # Initialize models
        super(Model, self).__init__(input_layers, output_autoencoder)
        self.encoder = Model(input_layers, output_encoder)
        self.decoder = Model(encoded_layer, output_decoder)
        Model.compile(self, optimizer=optimizer, loss=losses,
                      loss_weights=loss_weights)

    def encode(self, data):
        """
        Encodes the input data to the latent representation.

        Parameters
        ----------
        data : array_like
            The data to be encoded.

        Returns
        -------
        latent_data : array_like
            The latent representation of the input data.

        """
        return self.encoder.predict(data)

    def decode(self, latent_data):
        """
        Decodes the input data from the latent representation.

        Parameters
        ----------
        latent_data : array_like
            The data to be decoded.

        Returns
        -------
        data : array_like
            The full representation of the latent data.

        """
        return self.decoder.predict(latent_data)

    @classmethod
    def _construct_fusion_encoder(cls, mm_layer, hidden_dims, dropout_rates):
        """
        Adds a dense fusion network for encoding to the latent representation
        from a concatenated multimodal layer.

        Parameters
        ----------
        mm_layer : Keras tensor output from layer
            The multimodel input.
        hidden_dims : int or sequence of ints
            Number of elements per layer of the encoder.  The last element of
            the array represents the number of elements of the latent
            representation.
        dropout_rates : sequence of floats
            Dropout rates for each encoder layer.

        Returns
        -------
        encoder : Keras tensor output from layer
            The encoder network.

        """
        kernel_constraints = cls._get_kernel_constraints(dropout_rates)
        mm_layer = Dropout(dropout_rates[0])(mm_layer)
        encoder = mm_layer
        for i, dim in enumerate(hidden_dims):
            encoder = Dense(dim, activation='relu',
                            kernel_constraint=kernel_constraints[i+1])(encoder)
            encoder = Dropout(dropout_rates[i+1])(encoder)
        return encoder

    @classmethod
    def _construct_fusion_decoder(cls, encoded_layer, encoder, hidden_dims,
                                  fusion_shapes, dropout_rates):
        """
        Adds a dense fusion network for decoding from the latent
        representation.

        Parameters
        ----------
        encoded_layer : Keras tensor input
            The latent input layer.
        encoder : Keras tensor output from layer
            The encoder network.
        hidden_dims : int or sequence of ints
            Number of elements per layer of the decoder used in reverse order
            from second-last element to first element.
        fusion_shapes : array_like
            Array holding the shapes of the unimodal layers for each modality
            right before fusion.
        dropout_rates : sequence of floats
            Dropout rates for each decoder layer, used in reverse order from
            second-last element to first element.

        Returns
        -------
        fusion_decoder : sequence of Keras tensor outputs from layers
            The fusion decoder networks starting from the latent layer.
        fusion_autoencoder : sequence of Keras tensor outputs from layers
            The networks up to and including the fusion decoder.

        """
        kernel_constraints = cls._get_kernel_constraints(dropout_rates)
        decoder = encoded_layer
        autoencoder = encoder
        for i in range(len(hidden_dims)-2, -1, -1):
            layer = Dense(hidden_dims[i], activation='relu',
                          kernel_constraint=kernel_constraints[i])
            dropout = Dropout(dropout_rates[i])
            decoder = dropout(layer(decoder))
            autoencoder = dropout(layer(autoencoder))
        # No activation and no dropout in unimodal separation layers
        separation_layers = [Dense(np.prod(shape[1:])) for shape in fusion_shapes]
        fusion_decoder = [layer(decoder) for layer in separation_layers]
        fusion_autoencoder = [layer(autoencoder)
                              for layer in separation_layers]
        for i, shape in enumerate(fusion_shapes):
            fusion_decoder[i] = Reshape(shape[1:])(fusion_decoder[i])
            fusion_autoencoder[i] = Reshape(shape[1:])(fusion_autoencoder[i])
        return fusion_decoder, fusion_autoencoder

    @staticmethod
    def _construct_unimodal_encoders(input_layers, **kwargs):
        """
        Returns the first argument.  Override this method to add unimodal
        encoders between the input layers and the fusion network.

        Parameters
        ----------
        input_layers : sequence of Keras tensor inputs
            The input layer for each modality.

        Returns
        -------
        um_layers : sequence of Keras tensor outputs from layers
            The unimodal encoder networks for each modality.

        """
        um_layers = input_layers
        return um_layers

    @staticmethod
    def _construct_unimodal_decoders(fusion_decoder, fusion_autoencoder,
                                     output_shapes, **kwargs):
        """
        Returns the first two arguments.  Override this method to add unimodal
        decoders between the fusion network and the output layers.

        Parameters
        ----------
        fusion_decoder : sequence of Keras tensor outputs from layers
            The fusion decoder networks starting from the latent layer.
        fusion_autoencoder : sequence of Keras tensor outputs from layers
            The networks up to and including the fusion decoder.
        output_shapes : sequence of lists
            Shape of output elements per modality.

        Returns
        -------
        output_decoder : sequence of Keras tensor outputs from layers
            The output decoder networks starting from the latent layer.
        output_autoencoder : sequence of Keras tensor outputs from layers
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
        layers : sequence of Keras tensor outputs from layers
            The network with the output shapes of interest.

        Returns
        -------
        output_shapes : sequence of lists
            The output shapes.

        """
        output_shapes = [K.int_shape(layer) for layer in layers]
        return output_shapes

    @staticmethod
    def _get_kernel_constraints(dropout_rates):
        """
        Constructs a kernel constraint for each dropout rate.

        Parameters
        ----------
        dropout_rates : sequence of floats
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
    def _replace_bregman_strings(cls, losses):
        """
        Replaces strings referring to Bregman divergences with the actual
        divergence instances.

        Parameters
        ----------
        losses : str, callable or array_like
            The loss functions.

        Returns
        -------
        losses : str, callable or array_like
            The loss function where the strings referring to Bregman
            divergences were replaced.

        """
        if isinstance(losses, six.string_types):
            fun = getattr(bregman_divergences, losses, None)
            if fun is not None:
                losses = fun
        elif isinstance(losses, (list, tuple, np.ndarray)):
            losses = [cls._replace_bregman_strings(loss)
                      for loss in losses]
        return losses
