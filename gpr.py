from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("print_kernel", False, "Option to print out kernel")


class GaussianProcessRegression(object):
    
    def __init__(self, k_data_data, k_data_test, output_y):
        with tf.name_scope("init"):
            self.output_y = output_y
            self.k_data_data=k_data_data
            self.k_data_test=k_data_test
            #self.k_test_test=k_test_test
            self.num_train, self.output_dim = self.output_y.shape

            self.stability_eps = tf.identity(tf.placeholder(tf.float64))
            self.current_stability_eps = 1e-10

            self.y_pl = tf.placeholder(tf.float64, [1291, 1], name="y_train")
            self.K_data_data_pl = tf.placeholder(tf.float64, [1291, 1291], name="K_data_data")
            
            
            self.l_np = None
            self.v_np = None
            self.k_np = None     

    def _build_cholesky(self):
        self.K_data_data_reg = self.K_data_data_pl + tf.eye(self.output_y.shape[0], dtype=tf.float64) * self.stability_eps
        self.l = tf.linalg.cholesky(self.K_data_data_reg)
        self.v = tf.linalg.triangular_solve(self.l, self.y_pl)

    def predict(self, k_test_test, sess, get_var=False):
        
        self.k_test_test = k_test_test
        
        if self.l_np is None:
            self._build_cholesky()
            start_time = time.time()

            while self.current_stability_eps < 10:
                try:
                    start_time = time.time()
                    self.l_np, self.v_np = sess.run([self.l, self.v], feed_dict={self.y_pl: self.output_y,
                                                                                 self.K_data_data_pl: self.k_data_data,
                                                                                 self.stability_eps: self.current_stability_eps})
                    tf.logging.info("Computed L_DD in %.3f secs"% (time.time() - start_time))
                    break
                except tf.errors.InvalidArgumentError:
                    if self.current_stability_eps<1:
                        self.current_stability_eps *= 10
                    else:
                        self.current_stability_eps += 1
                    tf.logging.info("Cholesky decomposition failed, trying larger epsilon"
                                      ": {}".format(self.current_stability_eps))
        if self.current_stability_eps > 8:
            raise ArithmeticError("Could not compute Cholesky decomposition.")
                       
        self.K_data_test_pl = tf.placeholder(tf.float64, [1291, 327], name="K_data_test")
        self.K_test_test_pl = tf.placeholder(tf.float64, [327, 327], name="K_test_test")
               
        a = tf.matrix_triangular_solve(self.l, self.K_data_test_pl)
        fmean = tf.matmul(a, self.v, transpose_a=True)

        fvar = tf.diag_part(self.K_test_test_pl) - tf.reduce_sum(tf.square(a), 0)
        fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, self.output_y.shape[1]])

        self.fmean = fmean
        self.fvar = fvar    
            

        start_time = time.time()
        mean_pred, var_pred = sess.run([self.fmean, self.fvar], feed_dict = {self.K_data_test_pl: self.k_data_test,
                                                                             #self.K_data_data_pl: self.k_data_data,
                                                                             self.l: self.l_np,
                                                                             self.v: self.v_np,
                                                                             self.K_test_test_pl: self.k_test_test})
        tf.logging.info("Did regression in %.3f secs"% (time.time() - start_time))

        return mean_pred, var_pred 