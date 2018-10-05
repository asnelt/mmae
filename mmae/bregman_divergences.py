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
This module implements common Bregman divergences.

"""
from __future__ import division

import sys
import abc
from keras import backend as K


# Ensure abstract base class compatibility
if sys.version_info[0] == 3 and sys.version_info[1] >= 4 \
        or sys.version_info[0] > 3:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class BregmanDivergence(ABC):
    """
    This abstract class represents a Bregman divergence.  Override the abstract
    methods `_phi` and `_phi_gradient` to implement a specific Bregman
    divergence.

    """

    def __call__(self, x, y):
        """
        This method implements the equation of the Bregman divergence.

        """
        return K.mean(self._phi(x) - self._phi(y)
                      - (x - y) * self._phi_gradient(y), axis=-1)

    @staticmethod
    @abc.abstractmethod
    def _phi(z):
        """
        This is the phi function of the Bregman divergence.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _phi_gradient(z):
        """
        This is the gradient of the phi function of the Bregman divergence.

        """
        pass


class GaussianDivergence(BregmanDivergence):
    """
    This class represents the squared Euclidean distance corresponding to a
    Gaussian noise model.

    """

    @staticmethod
    def _phi(z):
        return K.square(z) / 2.0

    @staticmethod
    def _phi_gradient(z):
        return z

gaussian_divergence = GaussianDivergence()


class GammaDivergence(BregmanDivergence):
    """
    This class represents the Itakura-Saito distance corresponding to a Gamma
    noise model.

    """

    @staticmethod
    def _phi(z):
        z = K.maximum(z, K.epsilon())
        return -K.log(z)

    @staticmethod
    def _phi_gradient(z):
        z = K.maximum(z, K.epsilon())
        return -1.0 / z

gamma_divergence = GammaDivergence()


class BernoulliDivergence(BregmanDivergence):
    """
    This class represents the logistic loss function corresponding to a
    Bernoulli noise model.

    """

    @staticmethod
    def _phi(z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return z * K.log(z) + (1 - z) * K.log(1 - z)

    @staticmethod
    def _phi_gradient(z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return K.log(z) - K.log(1 - z)

bernoulli_divergence = BernoulliDivergence()


class PoissonDivergence(BregmanDivergence):
    """
    This class represents the generalized Kullback-Leibler divergence
    corresponding to a Poisson noise model.

    """

    @staticmethod
    def _phi(z):
        z = K.maximum(z, K.epsilon())
        return z * K.log(z) - z

    @staticmethod
    def _phi_gradient(z):
        z = K.maximum(z, K.epsilon())
        return K.log(z)

poisson_divergence = PoissonDivergence()
