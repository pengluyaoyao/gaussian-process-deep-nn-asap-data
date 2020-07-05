#!/usr/bin/env python

import pandas as pd
import os
import sys
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import itertools
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import nngp1
import gpr
import ard
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

class args:
    project_dir = os.getcwd()
    data_dir = os.getcwd()+'/data'
    model_type = 'bert'
    model_name_or_path = 'bert-base-cased' #can be path of fined tuned model    
    task_name = 'asap'
    output_dir = 'results/'+task_name
    item_id = 1

args=args

asap_ranges = {
    1: (2,12),
    2: (1,6),
    3: (0,3),
    4: (0,3),
    5: (0,4),
    6: (0,4),
    7: (0,30),
    8: (0,60)
}

wide_features = ['unique_word_count', 'sentence_length_entropy', 'sentence_length_words_mean', 
                 'sentence_count', 'type_token_ratio', 'word_lengths_percentile_75', 'syllable_ratio', 
                 'sentence_diversity', 'unique_spelling_error_ratio']

df_tr = pd.read_csv(args.data_dir+'/essay_asap{}_tr.csv'.format(args.item_id))
X_w_train = df_tr[wide_features]
labels_tr = (df_tr['dimscore1']-asap_ranges[args.item_id][0])/(asap_ranges[args.item_id][1]-asap_ranges[args.item_id][0])
n_train = X_w_train.shape[0]

df_dev = pd.read_csv(args.data_dir+'/essay_asap{}_te.csv'.format(args.item_id))
X_w_dev = df_dev[wide_features]
labels_dev = df_dev['dimscore1']
n_dev = X_w_dev.shape[0]
n_wide_features = len(wide_features)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_w_std = scaler.fit_transform(np.concatenate([X_w_train, X_w_dev], axis=0))
X_w_train_std = X_w_std[0:n_train, :]
X_w_dev_std = X_w_std[n_train:, :]

X = X_w_train_std
X_dev = X_w_dev_std

conf_priors = np.array([[2.]*n_wide_features, [0.75]*n_wide_features])

amplitude = (np.finfo(np.float64).tiny +
           tf.nn.softplus(
               tf.Variable(10.*tf.ones(1, dtype=tf.float64)),
               name='amplitude'))

length_scale = (np.finfo(np.float64).tiny +
              tf.nn.softplus(
                  tf.Variable([30.]*n_wide_features, dtype=np.float64),
                  name='length_scale'))

rv_scale = tfd.Gamma(
  concentration=conf_priors[0],
  rate=conf_priors[1],
  name='length_scale_prior')

kernel = ard.InputScaledKernel(tfk.ExponentiatedQuadratic(amplitude),
                         length_scale,
                         name='ARD_kernel')

gp = tfd.Independent(
  tfd.GaussianProcess(kernel=kernel, index_points=X),
  reinterpreted_batch_ndims=1)

# Joint log prob of length_scale params and data
log_likelihood = (gp.log_prob(tf.constant(labels_tr.tolist(), dtype=tf.float64)) +
                tf.reduce_sum(rv_scale.log_prob(length_scale)))

# Optimization target (neg log likelihood) and optimizer. Use a fairly
# generous learning rate (adaptive optimizer will adapt :))
loss = -log_likelihood
opt = tf.train.AdamOptimizer(.05).minimize(loss)

num_iters = 400
losses_ = np.zeros(num_iters, np.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        _, losses_[i] = sess.run([opt, loss])
#         if losses_[i] < 1e-10:
#             break
    
    length_scale_opt = length_scale.eval()
    amplitude_opt = amplitude.eval()
    
    k_w_data_data = sess.run(kernel.matrix(X, X))
    k_w_data_test = sess.run(kernel.matrix(X, X_dev))
    k_w_test_test = sess.run(kernel.matrix(X_dev, X_dev))

plt.plot(losses_)
plt.savefig('loss_fn.png')
    
X_d_train = np.load(args.data_dir+'/X_d_train_{}.npy'.format(args.item_id))
X_d_dev = np.load(args.data_dir+'/X_d_test_{}.npy'.format(args.item_id))  

config = {'depth': 3, 'nonfn': 'relu', 'bv': 6, 'wv': 5}

var_aa_grid = np.load(args.project_dir+'/grid_data/var_aa_grid_{}.npy'.format(config['nonfn']))
corr_ab_grid = np.load(args.project_dir+'/grid_data/corr_ab_grid_{}.npy'.format(config['nonfn']))
qaa_grid = np.load(args.project_dir+'/grid_data/qaa_grid_{}.npy'.format(config['nonfn']))
qab_grid = np.load(args.project_dir+'/grid_data/qab_grid_{}.npy'.format(config['nonfn']))

var_aa_grid = tf.convert_to_tensor(var_aa_grid)
corr_ab_grid = tf.convert_to_tensor(corr_ab_grid)
qaa_grid = tf.convert_to_tensor(qaa_grid)
qab_grid = tf.convert_to_tensor(qab_grid)

with tf.Session() as sess:
    nngp_kernel = nngp1.NNGPKernel(
        depth=config['depth'],
        weight_var=config['wv'],
        bias_var=config['bv'],
        nonlin_fn=config['nonfn'],
        grid_path=args.project_dir+'/grid_data/',
        var_aa_grid=var_aa_grid,
    corr_ab_grid=corr_ab_grid,
    qaa_grid=qaa_grid,
    qab_grid=qab_grid,
        use_fixed_point_norm=False)
    
    X_d_train_pl = tf.placeholder(dtype=tf.float64, shape=[X_d_train.shape[0], X_d_train.shape[1]])
    X_d_dev_pl = tf.placeholder(dtype=tf.float64, shape=[X_d_dev.shape[0], X_d_dev.shape[1]])
    #X_w_train_pl = tf.placeholder(dtype=tf.float64, shape=[X_w_train.shape[0], X_w_train.shape[1]])
    #X_w_dev_pl = tf.placeholder(dtype=tf.float64, shape=[X_w_dev.shape[0], X_w_dev.shape[1]])

    kdiag = nngp_kernel.k_diag(input_x=X_d_train_pl, input_wx=None)
    k_data_data = nngp_kernel.k_full(input1=X_d_train_pl, w_input1=None)
    k_data_test = tf.identity(nngp_kernel.k_full(input1=X_d_train_pl, input2=X_d_dev_pl, w_input1=None, w_input2=None))
    k_test_test = nngp_kernel.k_full(input1=X_d_dev_pl, w_input1=None)

    feed_dict={X_d_train_pl: X_d_train, X_d_dev_pl:X_d_dev}
    kdiag = sess.run(kdiag, feed_dict={X_d_train_pl: X_d_train})
    k_data_data, k_data_test, k_test_test = sess.run([k_data_data, k_data_test, k_test_test], feed_dict=feed_dict)

K_data_data = k_data_data+k_w_data_data
K_data_test = k_data_test+k_w_data_test
K_test_test = k_test_test+k_w_test_test

from gpr import GaussianProcessRegression

with tf.Session() as sess:
    model = GaussianProcessRegression(K_data_data[0], K_data_test[0], np.reshape(labels_tr.tolist(), (-1,1)))
    gp_pred, var_pred =  model.predict(K_test_test[0], sess)
    
from sklearn.metrics import cohen_kappa_score

gp_pred_int = np.rint(gp_pred*10+2)
qwk = cohen_kappa_score(np.array(labels_dev, dtype=float), gp_pred_int, weights = 'quadratic')
print(qwk)    
