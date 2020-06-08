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
This module implements common Bregman divergences.

"""
from __future__ import division

import abc
try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import LossFunctionWrapper
    from tensorflow.keras.utils.losses_utils import Reduction
except ImportError:
    from keras import backend as K
    from keras.losses import LossFunctionWrapper
    from keras.utils.losses_utils import Reduction


class BregmanDivergence(LossFunctionWrapper):
    """
    This abstract class represents a Bregman divergence.  Override the abstract
    methods `_phi` and `_phi_gradient` to implement a specific Bregman
    divergence.

    Parameters
    ----------
    reduction : keras.utils.losses_utils.Reduction
        The type of loss reduction.
    name : str
        The name of the divergence.

    """
    def __init__(self, reduction=Reduction.SUM_OVER_BATCH_SIZE, name=None):

        def bregman_function(x, y):
            """
            This function implements the generic Bregman divergence.

            """
            return K.mean(self._phi(x) - self._phi(y)
                          - (x - y) * self._phi_gradient(y), axis=-1)

        super(BregmanDivergence, self).__init__(bregman_function, name=name,
                                                reduction=reduction)
        self.__name__ = name

    @abc.abstractmethod
    def _phi(self, z):
        """
        This is the phi function of the Bregman divergence.

        """
        pass

    @abc.abstractmethod
    def _phi_gradient(self, z):
        """
        This is the gradient of the phi function of the Bregman divergence.

        """
        pass


class GaussianDivergence(BregmanDivergence):
    """
    This class represents the squared Euclidean distance corresponding to a
    Gaussian noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'gaussian_divergence')

    """
    def __init__(self, name='gaussian_divergence'):
        super(GaussianDivergence, self).__init__(name=name)

    def _phi(self, z):
        return K.square(z) / 2.0

    def _phi_gradient(self, z):
        return z

gaussian_divergence = GaussianDivergence()


class GammaDivergence(BregmanDivergence):
    """
    This class represents the Itakura-Saito distance corresponding to a Gamma
    noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'gamma_divergence')

    """
    def __init__(self, name='gamma_divergence'):
        super(GammaDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return -K.log(z)

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return -1.0 / z

gamma_divergence = GammaDivergence()


class BernoulliDivergence(BregmanDivergence):
    """
    This class represents the logistic loss function corresponding to a
    Bernoulli noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'bernoulli_divergence')

    """
    def __init__(self, name='bernoulli_divergence'):
        super(BernoulliDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return z * K.log(z) + (1 - z) * K.log(1 - z)

    def _phi_gradient(self, z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return K.log(z) - K.log(1 - z)

bernoulli_divergence = BernoulliDivergence()


class PoissonDivergence(BregmanDivergence):
    """
    This class represents the generalized Kullback-Leibler divergence
    corresponding to a Poisson noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'poisson_divergence')

    """
    def __init__(self, name='poisson_divergence'):
        super(PoissonDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return z * K.log(z) - z

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return K.log(z)

poisson_divergence = PoissonDivergence()


class BinomialDivergence(BregmanDivergence):
    """
    This class represents the loss function corresponding to a binomial noise
    model.

    Parameters
    ----------
    n : int
        The number of trials in the binomial noise model.  The number must be
        positive.
    name : str, optional
        The name of the divergence.  (Default: 'binomial_divergence')

    Attributes
    ----------
    n : int
        The number of trials in the binomial noise model.

    """
    def __init__(self, n, name='binomial_divergence'):
        self.n = n
        super(BinomialDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.clip(z, K.epsilon(), self.n - K.epsilon())
        return z * K.log(z) + (self.n - z) * K.log(self.n - z)

    def _phi_gradient(self, z):
        z = K.clip(z, K.epsilon(), self.n - K.epsilon())
        return K.log(z) - K.log(self.n - z)


class NegativeBinomialDivergence(BregmanDivergence):
    """
    This class represents the loss function corresponding to a negative
    binomial noise model.

    Parameters
    ----------
    r : int
        The number of failures in the negative binomial noise model.  The
        number must be positive.
    name : str, optional
        The name of the divergence.  (Default: 'negative_binomial_divergence')

    Attributes
    ----------
    r : int
        The number of failures in the negative binomial noise model.

    """
    def __init__(self, r, name='negative_binomial_divergence'):
        self.r = r
        super(NegativeBinomialDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return z * K.log(z) - (self.r + z) * K.log(self.r + z)

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return K.log(z) - K.log(self.r + z)
