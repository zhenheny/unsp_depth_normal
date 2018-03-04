from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01
MIN_EDGE = 0.0001

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    tgt_image = tf.image.resize_bilinear(tgt_image, [128, 416])
    src_image_stack = tf.image.resize_bilinear(src_image_stack, [128, 416])
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    batch_norm_params = {'is_training': is_training}
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # normalizer_fn = None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=1, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                        normalizer_fn=None, activation_fn=None)
                    
                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points

def disp_net(tgt_image, is_training=True, do_edge=False):
    batch_norm_params = {'is_training': is_training, 'decay':0.999}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    tgt_image = tf.image.resize_bilinear(tgt_image, [127, 415]) 
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # normalizer_fn = None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=1, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6'    )

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4 = tf.image.resize_bilinear(disp4, [H//8, W//8])
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3 = tf.image.resize_bilinear(disp3, [H//4, W//4])
            cnv1b_shape = cnv1b.get_shape().as_list()
            disp3_up = tf.image.resize_bilinear(disp3, [cnv1b_shape[1], cnv1b_shape[2]])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            upcnv2 = tf.image.resize_bilinear(upcnv2, [cnv1b_shape[1], cnv1b_shape[2]])
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2 = tf.image.resize_bilinear(disp2, [H//2, W//2])
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            disp2_up = tf.image.resize_bilinear(disp2_up, [upcnv1.get_shape().as_list()[1], upcnv1.get_shape().as_list()[2]])
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            disp1 = tf.image.resize_bilinear(disp1, [H, W])

            # Edge mask layers
            if do_edge:
                with tf.variable_scope('edge'):
                    upcnv7_e = slim.conv2d_transpose(cnv7b, 512, [4, 4], stride=2, scope='upcnv7')
                    # There might be dimension mismatch due to uneven down/up-sampling
                    upcnv7_e = resize_like(upcnv7_e, cnv6b)
                    i7_in_e  = tf.concat([upcnv7_e, cnv6b], axis=3)
                    icnv7_e  = slim.conv2d(i7_in_e, 512, [3, 3], stride=1, scope='icnv7')

                    upcnv6_e = slim.conv2d_transpose(icnv7_e, 512, [4, 4], stride=2, scope='upcnv6')
                    upcnv6_e = resize_like(upcnv6_e, cnv5b)
                    i6_in_e  = tf.concat([upcnv6_e, cnv5b], axis=3)
                    icnv6_e  = slim.conv2d(i6_in_e, 512, [3, 3], stride=1, scope='icnv6'    )

                    upcnv5_e = slim.conv2d_transpose(icnv6_e, 256, [4, 4], stride=2, scope='upcnv5')
                    upcnv5_e = resize_like(upcnv5_e, cnv4b)
                    i5_in_e  = tf.concat([upcnv5_e, cnv4b], axis=3)
                    icnv5_e  = slim.conv2d(i5_in_e, 256, [3, 3], stride=1, scope='icnv5')

                    upcnv4_e = slim.conv2d_transpose(icnv5_e, 128, [4, 4], stride=2, scope='upcnv4')
                    i4_in_e  = tf.concat([upcnv4_e, cnv3b], axis=3)
                    icnv4_e  = slim.conv2d(i4_in_e, 128, [3, 3], stride=1, scope='icnv4')
                    edge4  = slim.conv2d(icnv4_e, 1,   [3, 3], stride=1, 
                        activation_fn=tf.sigmoid, normalizer_fn=None, scope='edge4') + MIN_EDGE
                    edge4 = tf.image.resize_nearest_neighbor(edge4, [H//8,W//8])
                    # edge4_up = tf.image.resize_bilinear(edge4, [np.int(H/4), np.int(W/4)])
                    edge4_up = tf.image.resize_nearest_neighbor(edge4, [np.int(H/4), np.int(W/4)])

                    upcnv3_e = slim.conv2d_transpose(icnv4_e, 64,  [4, 4], stride=2, scope='upcnv3')
                    i3_in_e  = tf.concat([upcnv3_e, cnv2b, edge4_up], axis=3)
                    # i3_in_e  = tf.concat([upcnv3_e, cnv2b], axis=3)
                    icnv3_e  = slim.conv2d(i3_in_e, 64,  [3, 3], stride=1, scope='icnv3')
                    edge3  = slim.conv2d(icnv3_e, 1,   [3, 3], stride=1, 
                        activation_fn=tf.sigmoid, normalizer_fn=None, scope='edge3') + MIN_EDGE
                    edge3 = tf.image.resize_nearest_neighbor(edge3, [H//4,W//4])
                    # edge3_up = tf.image.resize_bilinear(edge3, [np.int(H/2), np.int(W/2)])
                    edge3_up = tf.image.resize_nearest_neighbor(edge3, [np.int(H/2), np.int(W/2)])
                    edge3_up = tf.image.resize_nearest_neighbor(edge3_up, [cnv1b_shape[1], cnv1b_shape[2]])
                    upcnv2_e = slim.conv2d_transpose(icnv3_e, 32,  [4, 4], stride=2, scope='upcnv2')
                    upcnv2_e = tf.image.resize_nearest_neighbor(upcnv2_e, [cnv1b_shape[1], cnv1b_shape[2]])
                    i2_in_e  = tf.concat([upcnv2_e, cnv1b, edge3_up], axis=3)
                    # i2_in_e  = tf.concat([upcnv2_e, cnv1b], axis=3)
                    icnv2_e  = slim.conv2d(i2_in_e, 32,  [3, 3], stride=1, scope='icnv2')
                    edge2  = slim.conv2d(icnv2_e, 1,   [3, 3], stride=1, 
                        activation_fn=tf.sigmoid, normalizer_fn=None, scope='edge2') + MIN_EDGE
                    edge2 = tf.image.resize_nearest_neighbor(edge2, [H//2,W//2])
                    # edge2_up = tf.image.resize_bilinear(edge2, [H, W])
                    edge2_up = tf.image.resize_nearest_neighbor(edge2, [H, W])

                    upcnv1_e = slim.conv2d_transpose(icnv2_e, 16,  [4, 4], stride=2, scope='upcnv1')
                    edge2_up = tf.image.resize_nearest_neighbor(edge2, [upcnv1_e.get_shape().as_list()[1], upcnv1_e.get_shape().as_list()[2]])
                    i1_in_e  = tf.concat([upcnv1_e, edge2_up], axis=3)
                    # i1_in_e  = tf.concat([upcnv1_e], axis=3)
                    icnv1_e  = slim.conv2d(i1_in_e, 16,  [3, 3], stride=1, scope='icnv1')
                    edge1  = slim.conv2d(icnv1_e, 1,   [3, 3], stride=1,
                        activation_fn=tf.sigmoid, normalizer_fn=None, scope='edge1') + MIN_EDGE
                    edge1 = tf.image.resize_nearest_neighbor(edge1, [H,W])

                    # down-scale the edges at lower scale from highest resolution edge results
                    # edge2 = slim.max_pool2d(edge1, 2)
                    # edge3 = slim.max_pool2d(edge2, 2)
                    # edge4 = slim.max_pool2d(edge3, 2)
            else:
                edge1 = None
                edge2 = None
                edge3 = None
                edge4 = None
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], [edge1, edge2, edge3, edge4], end_points
