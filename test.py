import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
print(tf.__version__)


chkp.print_tensors_in_checkpoint_file('output/cs_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_noedge_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_input417_l2_deconvk4_nnupsample_noscaling_wt0.2_expwt0.0_dm_xz_2loss_l1wt0.01_depth4pose_depth1normal_eval_edgepretrained_cs/model.latest', tensor_name='', all_tensors=True)
# import numpy as np

# key_num = 5
# key_dim = 2

# with tf.device("/gpu:0"):
#     keys = tf.tile(tf.expand_dims(tf.range(key_num, dtype=tf.int32), 1), [1, key_dim])

#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     print(sess.run(keys))

# import tensorflow as tf

# print tf.get_variable_scope()

# with tf.variable_scope('foo'):
#     with tf.variable_scope(tf.get_variable_scope()) as vs:
#         print vs
#         v = tf.get_variable('v', [1])
#         print v.name

#         with tf.variable_scope(vs, reuse=True):
#             v = tf.get_variable('v', [1])
#             print v.name

# with tf.variable_scope('foo', reuse=True):
#     v = tf.get_variable('v', [1])
#     print v.name



