from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
# tf.executing_eagerly()

def interp_lin(x, y, xp, log_spacing=False):
    with tf.name_scope('interp_lin'):
        if log_spacing:
            x = tf.log(x)
            xp = tf.log(xp)

        spacing = x[1] - x[0]
        grid = (xp - x[0]) / spacing
        ind1 = tf.cast(grid, tf.int32)
        ind2 = ind1 + 1
        max_ind = x.shape[0].value
        # set top and bottom indices identical if extending past end of range
        ind2 = tf.minimum(max_ind - 1, ind2)

        weight1 = tf.abs(xp - tf.gather(x, ind1)) / spacing
        weight2 = tf.abs(xp - tf.gather(x, ind2)) / spacing
        if log_spacing:
            weight1 = tf.exp(weight1)
            weight2 = tf.exp(weight2)

        weight1 = 1. - tf.reshape(weight1, [-1] + [1] * (len(y.shape) - 1))
        weight2 = 1. - tf.reshape(weight2, [-1] + [1] * (len(y.shape) - 1))

        weight_sum = weight1 + weight2
        weight1 /= weight_sum
        weight2 /= weight_sum

        y1 = tf.gather(y, ind1)
        y2 = tf.gather(y, ind2)
        yp = y1 * weight1 + y2 * weight2
        return yp


def _get_interp_idxs_weights_2d(x, xp, y, yp, x_log_spacing=False):
      with tf.name_scope('get_interp_idxs_weights_2d'):
        if x_log_spacing:
            x = tf.log(x)
            xp = tf.log(xp)

        with tf.control_dependencies([yp]):
            xp = tf.tile(xp, tf.shape(yp))#yp.shape)
        xyp = tf.expand_dims(tf.parallel_stack([xp, yp]), 1)
        xy0 = tf.reshape(tf.parallel_stack([x[0], y[0]]), [2, 1, 1])
        xy1 = tf.reshape(tf.parallel_stack([x[1], y[1]]), [2, 1, 1])

        spacing = xy1 - xy0
        ind_grid = (xyp - xy0) / spacing
        ind = tf.cast(ind_grid, tf.int32) + [[[0], [1]]]

        max_ind = [[[x.shape[0].value - 1]], [[y.shape[0].value - 1]]]
        #max_ind = [[[x.shape[0] - 1]], [[y.shape[0] - 1]]]
        ind = tf.minimum(ind, max_ind)
        ind_float = tf.cast(ind, tf.float64)

        xy_grid = ind_float * spacing + xy0

        weight = tf.abs(xyp - xy_grid) / spacing
        if x_log_spacing:
            weight = tf.parallel_stack([tf.exp(weight[0]), weight[1]])
        weight = 1. - weight

        weight_sum = tf.reduce_sum(weight, axis=1, keepdims=True)
        weight /= weight_sum

        return ind, weight


def interp_lin_2d(x, y, z, xp, yp, x_log_spacing=False):
    with tf.name_scope('interp_lin_2d'):
        ind, weight = _get_interp_idxs_weights_2d(x, xp, y, yp, x_log_spacing) #(2,2,10000),(2,2,10000)
        zp_accum = 0.

        for ind_x, weight_x in [(ind[0, 0], weight[0, 0]), #ind_x: (x1, x2), ind_y: (y1, y2)
                                (ind[0, 1], weight[0, 1])]:
            for ind_y, weight_y in [(ind[1, 0], weight[1, 0]),
                                  (ind[1, 1], weight[1, 1])]:
                zp = tf.gather_nd(z, tf.stack([ind_x, ind_y], axis=1)) #stack along dim1: (x1, y1), (x2, y2)
                zp_accum += zp * weight_x * weight_y #(z1, z2)*10000-> ()

    return zp_accum