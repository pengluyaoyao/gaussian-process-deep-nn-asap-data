from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import tensorflow.compat.v1 as tf
import interp1
tf.disable_v2_behavior()

class NNGPKernel(object):

    def __init__(self,
               depth=1,
               nonlin_fn=None,
               weight_var=1.,
               bias_var=1.,
               use_fixed_point_norm=False,
               use_precomputed_grid=True,
               grid_path=None,
               var_aa_grid=None,
               corr_ab_grid=None,
                qaa_grid =None,
                 qab_grid=None,
               sess=None):
        self.depth = depth
        self.weight_var = weight_var
        self.bias_var = bias_var
        self.use_fixed_point_norm = use_fixed_point_norm
        self.sess = sess
        #if FLAGS.use_precomputed_grid and (grid_path is None):
        #    raise ValueError("grid_path must be specified to use precomputed grid.")
        self.use_precomputed_grid=use_precomputed_grid
        (self.var_aa_grid, self.corr_ab_grid, self.qaa_grid, self.qab_grid) = (var_aa_grid, corr_ab_grid, qaa_grid, qab_grid)
        self.nonlin_fn = nonlin_fn
    def k_diag(self, input_x, input_wx, return_full=True):

        if self.use_fixed_point_norm:
            current_qaa = self.var_fixed_point
        else:
            current_qaa = self.weight_var * tf.convert_to_tensor([1.], dtype=tf.float64) + self.bias_var
        self.layer_qaa_dict = {0: current_qaa}
        for l in range(self.depth):
            with tf.name_scope("layer_%d" % l):
                        samp_qaa = interp1.interp_lin(self.var_aa_grid, self.qaa_grid, current_qaa)
                        samp_qaa = self.weight_var * samp_qaa + self.bias_var
                        self.layer_qaa_dict[l + 1] = samp_qaa
                        current_qaa = samp_qaa
        if return_full:
            qaa = current_qaa #110
        else:
            qaa = current_qaa[0]
        return qaa

    def k_full(self, input1, w_input1, input2=None, w_input2=None): #k_type: data_data, data_test, test_test
        input1 = self._input_layer_normalization(input1) #by row

        if input2 is None:
            input2 = input1
            w_input2 = w_input1
        else:
            input2 = self._input_layer_normalization(input2)
            #w_input2 = self._input_layer_normalization(w_input2)

        with tf.name_scope("k_full"):
            cov_init = tf.matmul(input1, input2, transpose_b=True) / input1.shape[1].value #100,100; den is a tensor
            #w_cov = tf.matmul(w_input1, w_input2, transpose_b=True)/ w_input1.shape[1].value

            self.k_diag(input1, w_input1) #110 vars
            q_aa_init = self.layer_qaa_dict[0] #L个

            q_ab = cov_init
            q_ab = self.weight_var * q_ab + self.bias_var #100, 100
            #corr = q_ab / q_aa_init[0]
            corr = tf.cast(q_ab, tf.float32) / tf.cast(q_aa_init[0], tf.float32) #100, 100
            
            if 32 > 1:
                batch_size, batch_count = self._get_batch_size_and_count(input1, input2)
                
                with tf.name_scope("q_ab"):
                    q_ab_all = []
                    for b_x in range(batch_count):
                        #tf.logging.info('computing kernel for batch:{}'.format(b_x))
                        with tf.name_scope("batch_%d" % b_x):
                            corr_flat_batch = corr[batch_size * b_x : batch_size * (b_x + 1), :]#batchsize, 100
                            corr_flat_batch = tf.reshape(corr_flat_batch, [-1]) #batchsize*100
                            #w_cov_flat_batch = tf.reshape(w_cov[batch_size * b_x : batch_size * (b_x + 1), :], [-1])
                            for l in range(self.depth):
                                with tf.name_scope("layer_%d" % l):
                                    q_aa = self.layer_qaa_dict[l]
                                    q_ab = interp1.interp_lin_2d(x=self.var_aa_grid,
                                                              y=self.corr_ab_grid,
                                                              z=self.qab_grid,
                                                              xp=q_aa,
                                                              yp=tf.cast(corr_flat_batch, tf.float64)) #10000

                                    q_ab = self.weight_var * q_ab + self.bias_var
                                    corr_flat_batch = q_ab / self.layer_qaa_dict[l + 1][0]

                            q_ab_all.append(q_ab)

                    q_ab_all = tf.parallel_stack(q_ab_all)
            else:
                with tf.name_scope("q_ab"):
                    corr_flat = tf.reshape(corr, [-1]) #10000
                    w_cov_flat = tf.reshape(w_cov, [-1])
                    for l in rangenp.reshape(scores_tr,[-1,1])(self.depth):
                        with tf.name_scope("layer_%d" % l):
                            q_aa = self.layer_qaa_dict[l]
                            q_ab = interp1.interp_lin_2d(x=self.var_aa_grid,
                                                      y=self.corr_ab_grid,
                                                      z=self.qab_grid,
                                                      xp=q_aa,
                                                      yp=tf.cast(corr_flat, tf.float64))
                            if l == self.depth-1:
                                q_ab = self.weight_var * q_ab + self.bias_var #+ self.weight_var * w_cov_flat)
                            else:
                                q_ab = self.weight_var * q_ab + self.bias_var
                                corr_flat = q_ab / self.layer_qaa_dict[l+1][0]
                        q_ab_all = q_ab

        return tf.reshape(q_ab_all, cov_init.shape, "qab")

    def _input_layer_normalization(self, x):
        with tf.name_scope("input_layer_normalization"):
          # Layer norm, fix to unit variance
            eps = 1e-15
            mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
            x_normalized = (x - mean) / tf.sqrt(var + eps)
            if self.use_fixed_point_norm:
                x_normalized *= tf.sqrt(
                    (self.var_fixed_point[0] - self.bias_var) / self.weight_var)
            return x_normalized

    def _get_batch_size_and_count(self, input1, input2):
        input1_size = input1.shape[0].value
        input2_size = input2.shape[0].value
        
            
        batch_size = min(np.iinfo(np.int32).max //
                         (2048 * input2_size), input1_size)
        while input1_size % batch_size != 0:
              batch_size -= 1

        batch_count = input1_size // batch_size
        return batch_size, batch_count


def _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn):
    log_weights_ab_unnorm = -(z1**2 + z2**2 - 2 * z1 * z2 * corr_ab) / (
      2 * var_aa[idx] * (1 - corr_ab**2))
    log_weights_ab = log_weights_ab_unnorm - tf.reduce_logsumexp(log_weights_ab_unnorm, axis=[0, 1], keepdims=True)
    weights_ab = tf.exp(log_weights_ab)

    qab_slice = tf.reduce_sum(nonlin_fn(z1) * nonlin_fn(z2) * weights_ab, axis=[0, 1])
    qab_slice = tf.Print(qab_slice, [idx], "Generating slice: ")
    return qab_slice


def _compute_qmap_grid(nonlin_fn,
                   n_gauss,
                   n_var,
                   n_corr,
                   log_spacing=False,
                   min_var=1e-8,
                   max_var=100.,
                   max_corr=0.99999,
                   max_gauss=10.):

    if n_gauss % 2 != 1:
        raise ValueError("n_gauss=%d should be an odd integer" % n_gauss)

    with tf.name_scope("compute_qmap_grid"):
        min_var = tf.convert_to_tensor(min_var, dtype=tf.float64)
        max_var = tf.convert_to_tensor(max_var, dtype=tf.float64)
        max_corr = tf.convert_to_tensor(max_corr, dtype=tf.float64)
        max_gauss = tf.convert_to_tensor(max_gauss, dtype=tf.float64)

    # Evaluation points for numerical integration over a Gaussian.
    z1 = tf.reshape(tf.linspace(-max_gauss, max_gauss, n_gauss), (-1, 1, 1))
    z2 = tf.transpose(z1, perm=[1, 0, 2])

    if log_spacing:
        var_aa = tf.exp(tf.linspace(tf.log(min_var), tf.log(max_var), n_var))
    else:
      # Evaluation points for pre-activations variance and correlation
        var_aa = tf.linspace(min_var, max_var, n_var)
    corr_ab = tf.reshape(tf.linspace(-max_corr, max_corr, n_corr), (1, 1, -1))

    # compute q_aa
    log_weights_aa_unnorm = -0.5 * (z1**2 / tf.reshape(var_aa, [1, 1, -1]))
    log_weights_aa = log_weights_aa_unnorm - tf.reduce_logsumexp(
        log_weights_aa_unnorm, axis=[0, 1], keepdims=True)
    weights_aa = tf.exp(log_weights_aa)
    qaa = tf.reduce_sum(nonlin_fn(z1)**2 * weights_aa, axis=[0, 1])

# compute q_ab
# weights to reweight uniform samples by, for q_ab.
# (weights are probability of z1, z2 under Gaussian
#  w/ variance var_aa and covariance var_aa*corr_ab)
# weights_ab will have shape [n_g, n_g, n_v, n_c]
    def fill_qab_slice(idx):
        return _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn)

    qab = tf.map_fn(
        fill_qab_slice,
        tf.range(n_var),
        dtype=tf.float64,
        parallel_iterations=multiprocessing.cpu_count())

    var_grid_pts = tf.reshape(var_aa, [-1])
    corr_grid_pts = tf.reshape(corr_ab, [-1])

    return var_grid_pts, corr_grid_pts, qaa, qab