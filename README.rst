===================================
mmae Package for TensorFlow / Keras
===================================

Package for multimodal autoencoders with Bregman divergences.


Description
-----------

This package contains an implementation of a flexible autoencoder that can
take into account the noise distributions of multiple modalities.  The
autoencoder can be used to find a low-dimensional representation of
multimodal data, taking advantage of the information that one modality
provides about another.

Noise distributions are taken into account by means of Bregman divergences
which correspond to particular exponential families such as Gaussian, Poisson
or gamma distributions.  Each modality can have its own Bregman divergence as
loss function, thereby assuming a particular output noise distribution.

By default, the autoencoder network fusing multiple modalities consists of a
variable number of ReLU layers that are densely connected.  The number of
layers and number of units per layer of the encoder and decoder networks are
symmetric.  Other network architectures can be easily implemented by overriding
convenience methods.


Requirements
------------

The package is compatible with Python 2.7 and 3.x and additionally requires
NumPy, Six and TensorFlow or Keras.  It was tested with Python 2.7.17,
Python 3.8.2, NumPy 1.18.4, Six 1.15.0, TensorFlow 2.2.0 and Keras 2.3.1.


Installation
------------

To install the mmae package with the TensorFlow back-end with GPU support,
run::

    pip install mmae[tensorflow-gpu]

To install the mmae package with the TensorFlow back-end without GPU support,
run::

    pip install mmae[tensorflow]

To install the mmae package with the Keras back-end, run::

    pip install mmae[keras]


Usage
-----

The main class of this package is ``MultimodalAutoencoder`` which is
implemented in the module ``mmae.multimodal_autoencoder``.  This class can be
used to easily construct a multimodal autoencoder for dimensionality reduction.
The main arguments for instantiation are ``input_shapes`` which is a list or
dictionary of shapes for each modality, ``hidden_dims`` which is a list of the
number of units per hidden layer of the encoder, and ``output_activations``
which is a list or dictionary of output activations for each modality.

The last element of ``hidden_dims`` is the dimensionality of the latent space
representation.  The other elements are mirrored for the decoder construction.
For instance, if ``hidden_dims = [128, 64, 8]`` then the encoder will have
hidden layers with 128 and 64 units and produce an 8 dimensional representation
whereas the decoder will take the 8 dimensional representation, feed it into
hidden layers with 64 and 128 units and produce multimodal outputs with shapes
following ``input_shapes``.

The method ``compile`` is used to set the model configuration for training.
The main arguments are ``optimizer`` which is a Keras optimizer and ``loss``
which is a list or dictionary of loss functions for each modality.  In addition
to the standard Keras loss functions, regular Bregman divergences can be used.
Current options are ``gaussian_divergence``, ``gamma_divergence``,
``bernoulli_divergence`` and ``poisson_divergence``, corresponding to Gaussian,
gamma, Bernoulli and Poisson noise models, respectively.  Divergences
corresponding to binomial and negative binomial distributions can be used by
instantiating ``BinomialDivergence`` and ``NegativeBinomialDivergence``,
respectively, and passing the instance in ``loss``.  To implement other
divergences, additional classes can be derived from ``BregmanDivergence``
where the abstract methods ``_phi`` and ``_phi_gradient`` need to be
overridden.  ``BregmanDivergence`` is implemented in the
``mmae.bregman_divergences`` module.

The following code fits a multimodal autoencoder to MNIST, where the images are
treated as one modality and the number label is treated as another modality:

.. code-block:: python

    # Remove 'tensorflow.' from the next line if you use just Keras
    from tensorflow.keras.datasets import mnist
    from mmae.multimodal_autoencoder import MultimodalAutoencoder
    # Load example data
    (x_train, y_train), (x_validation, y_validation) = mnist.load_data()
    # Scale pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    x_validation = x_validation.astype('float32') / 255.0
    y_validation = y_validation.astype('float32') / 255.0
    # Multimodal training data
    data = [x_train, y_train]
    # Multimodal validation data
    validation_data = [x_validation, y_validation]
    # Set network parameters
    input_shapes = [x_train.shape[1:], (1,)]
    # Number of units of each layer of encoder network
    hidden_dims = [128, 64, 8]
    # Output activation functions for each modality
    output_activations = ['sigmoid', 'relu']
    # Name of Keras optimizer
    optimizer = 'adam'
    # Loss functions corresponding to a noise model for each modality
    loss = ['bernoulli_divergence', 'poisson_divergence']
    # Construct autoencoder network
    autoencoder = MultimodalAutoencoder(input_shapes, hidden_dims,
                                        output_activations)
    autoencoder.compile(optimizer, loss)
    # Train model where input and output are the same
    autoencoder.fit(data, epochs=100, batch_size=256,
                    validation_data=validation_data)

To obtain a latent representation of the training data:

.. code-block:: python

    latent_data = autoencoder.encode(data)

To decode the latent representation:

.. code-block:: python

    reconstructed_data = autoencoder.decode(latent_data)

Encoding and decoding can also be merged into the following single statement:

.. code-block:: python

    reconstructed_data = autoencoder.predict(data)

By default, the different modalities are fed directly into a densely connected
fusion network.  In order to pre- and post-process each modality, for instance
using a convolutional neural network for the image data, the
``MultimodalAutoencoder`` methods ``_construct_unimodal_encoders`` and
``_construct_unimodal_decoders`` can be overridden.  These methods add networks
between the input and the fusion encoder and between the fusion decoder and the
output, respectively.


Source code
-----------

The source code of the mmae package is hosted on
`GitHub
<https://github.com/asnelt/mmae/>`_.


License
-------

Copyright (C) 2018-2020 Arno Onken

This file is part of the mmae package.

The mmae package is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your option) any
later version.

The mmae package is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, see <http://www.gnu.org/licenses/>.
