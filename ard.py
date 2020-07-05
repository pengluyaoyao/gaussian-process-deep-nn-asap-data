import pandas as pd
import os
import sys
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import itertools
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import nngp1

tf.disable_v2_behavior()

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

class InputTransformedKernel(tfk.PositiveSemidefiniteKernel):

    def __init__(self, kernel, transformation, name='InputTransformedKernel'):
        self._kernel = kernel
        self._transformation = transformation
        super().__init__(
            feature_ndims=kernel.feature_ndims,
            dtype=kernel.dtype,
            name=name)

    def apply(self, x1, x2):
        return self._kernel.apply(
            self._transformation(x1),
            self._transformation(x2))

    def matrix(self, x1, x2):
        return self._kernel.matrix(
            self._transformation(x1),
            self._transformation(x2))

    @property
    def batch_shape(self):
        return self._kernel.batch_shape

    def batch_shape_tensor(self):
        return self._kernel.batch_shape_tensor

class InputScaledKernel(InputTransformedKernel):

    def __init__(self, kernel, length_scales, name):
      length_scales = tf.expand_dims(
          length_scales,
          axis=-(kernel.feature_ndims + 1))
      def _transformation(x):
        return x / length_scales
      super().__init__(
          kernel=kernel,
          transformation=_transformation,
          name=name)