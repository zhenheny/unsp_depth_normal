from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import transformer as trans
import numpy as np
import pdb

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

def leaky_relu(_x, alpha=0.1):
    pos = tf.nn.relu(_x)
    neg = alpha * (_x - abs(_x)) * 0.5

    return pos + neg


def feature_pyramid(image, reuse):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"],
                      reuse=reuse):
                      #outputs_collections=end_points_collection):

      cnv1 = slim.conv2d(image, 16, [3, 3], stride=1, scope="cnv1")
      cnv2 = slim.conv2d(cnv1, 16, [3, 3], stride=1, scope="cnv2")
      cnv3 = slim.conv2d(cnv2, 32, [3, 3], stride=2, scope="cnv3")
      cnv4 = slim.conv2d(cnv3, 32, [3, 3], stride=1, scope="cnv4")
      cnv5 = slim.conv2d(cnv4, 64, [3, 3], stride=2, scope="cnv5")
      cnv6 = slim.conv2d(cnv5, 64, [3, 3], stride=1, scope="cnv6")
      cnv7 = slim.conv2d(cnv6, 96, [3, 3], stride=2, scope="cnv7")
      cnv8 = slim.conv2d(cnv7, 96, [3, 3], stride=1, scope="cnv8")
      cnv9 = slim.conv2d(cnv8, 128, [3, 3], stride=2, scope="cnv9")
      cnv10 = slim.conv2d(cnv9, 128, [3, 3], stride=1, scope="cnv10")
      cnv11 = slim.conv2d(cnv10, 192, [3, 3], stride=2, scope="cnv11")
      cnv12 = slim.conv2d(cnv11, 192, [3, 3], stride=1, scope="cnv12")

      return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12


def cost_volumn(feature1, feature2, d=4):
    """define the correlation between two feature map
    """
    H, W = map(int, feature1.get_shape()[1:3])
    feature2 = tf.pad(feature2, [[0,0], [d,d], [d,d],[0,0]], "SYMMETRIC")
    cv = []
    for i in range(2*d+1):
        for j in range(2*d+1):
            cv.append(tf.reduce_mean(
                feature1*feature2[:, i:(i+H), j:(j+W), :],
                axis=3, keep_dims=True))

    return tf.concat(cv, axis=3)


def motion_decoder_dc(inputs, level, out_channel=2):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu):
      cnv1 = slim.conv2d(inputs, 128, [3, 3], stride=1,
              scope="cnv1_fd_"+str(level))
      cnv2 = slim.conv2d(cnv1, 128, [3, 3], stride=1,
              scope="cnv2_fd_"+str(level))
      cnv3 = slim.conv2d(tf.concat([cnv1, cnv2], axis=3),
              96, [3, 3], stride=1, scope="cnv3_fd_"+str(level))
      cnv4 = slim.conv2d(tf.concat([cnv2, cnv3], axis=3),
              64, [3, 3], stride=1, scope="cnv4_fd_"+str(level))
      cnv5 = slim.conv2d(tf.concat([cnv3, cnv4], axis=3),
              32, [3, 3], stride=1, scope="cnv5_fd_"+str(level))
      flow = slim.conv2d(tf.concat([cnv4, cnv5], axis=3),
              out_channel, [3, 3], stride=1,
              scope="cnv6_fd_"+str(level), activation_fn=None)

      return flow, cnv5


def context_net(inputs, out_channel=2):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu):
                      #outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inputs, 128, [3, 3], rate=1, scope="cnv1_cn")
      cnv2 = slim.conv2d(cnv1, 128, [3, 3], rate=2, scope="cnv2_cn")
      cnv3 = slim.conv2d(cnv2, 128, [3, 3], rate=4, scope="cnv3_cn")
      cnv4 = slim.conv2d(cnv3, 96, [3, 3], rate=8, scope="cnv4_cn")
      cnv5 = slim.conv2d(cnv4, 64, [3, 3], rate=16, scope="cnv5_cn")
      cnv6 = slim.conv2d(cnv5, 32, [3, 3], rate=1, scope="cnv6_cn")

      flow = slim.conv2d(cnv6, out_channel, [3, 3], rate=1, scope="cnv7_cn", activation_fn=None)
      return flow

def pose_exp_net(tgt_image,
                 src_image_seq,
                 tgt_depth=None,
                 src_depth_seq=None,
                 do_exp=True,
                 do_dm=True,
                 is_training=True,
                 reuse=False,
                 in_size=[128, 416]):

    with_depth = (tgt_depth is not None) and (src_depth_seq is not None)

    # reorgnize images to pairs
    tgt_image = tf.image.resize_bilinear(tgt_image, in_size)
    if with_depth:
        tgt_depth = tf.image.resize_bilinear(tgt_depth, in_size)

    input_image_pairs = []
    for i in range(len(src_image_seq)):
        src_image = tf.image.resize_bilinear(src_image_seq[i], in_size)
        if with_depth:
            src_depth = tf.image.resize_bilinear(src_depth_seq[i], in_size)
            input_image_pairs.append(
                    tf.concat([tgt_image, src_image, tgt_depth, src_depth], axis=3))
        else:
            input_image_pairs.append(tf.concat([tgt_image, src_image], axis=3))

    inputs = tf.concat(input_image_pairs, axis=0)
    batch_norm_params = {'is_training': is_training}
    num_source = len(src_image_seq)

    with tf.variable_scope('motion_net', reuse=reuse) as sc:
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
                pose_pred = slim.conv2d(cnv7, 6, [1, 1], scope='pred',
                    stride=1, normalizer_fn=None, activation_fn=None)


                pose_avg = tf.reduce_mean(pose_pred, [1, 2])

                print('pose_shape {}, {}'.format(pose_avg.get_shape().as_list(), num_source))
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_avg = tf.expand_dims(pose_avg, axis=1)
                pose_avg = tf.split(pose_avg, num_source, axis=0)
                pose_final = 0.01 * tf.concat(pose_avg, axis=1)

                print('pose_final_shape {}'.format(pose_final.get_shape().as_list()))

            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, 2, [3, 3], stride=1, scope='mask4',
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, 2, [3, 3], stride=1, scope='mask3',
                        normalizer_fn=None, activation_fn=None)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, 2, [5, 5], stride=1, scope='mask2',
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, 2, [7, 7], stride=1, scope='mask1',
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None

            ## Dense motion specific layers
            if do_dm:
                with tf.variable_scope('dm'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    upcnv4 = tf.concat([upcnv4, cnv3], axis=3)
                    dm4 = slim.conv2d(upcnv4,  3, [3, 3], stride=1, scope='dm4',
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    cnv3_shape = upcnv3.get_shape().as_list()
                    dm4_up = tf.image.resize_nearest_neighbor(dm4, [cnv3_shape[1], cnv3_shape[2]])
                    upcnv3 = tf.concat([upcnv3, cnv2, dm4_up], axis=3)
                    dm3 = slim.conv2d(upcnv3, 3, [3, 3], stride=1, scope='dm3',
                        normalizer_fn=None, activation_fn=None)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    cnv2_shape = upcnv2.get_shape().as_list()
                    dm3_up = tf.image.resize_nearest_neighbor(dm3, [cnv2_shape[1], cnv2_shape[2]])
                    upcnv2 = tf.concat([upcnv2, cnv1, dm3_up], axis=3)
                    dm2 = slim.conv2d(upcnv2, 3, [3, 3], stride=1, scope='dm2',
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    cnv1_shape = upcnv1.get_shape().as_list()
                    dm2_up = tf.image.resize_nearest_neighbor(
                            dm2, [cnv1_shape[1], cnv1_shape[2]])
                    upcnv1 = tf.concat([upcnv1, dm2_up], axis=3)
                    dm1 = slim.conv2d(upcnv1, 3, [7, 7], stride=1, scope='dm1',
                        normalizer_fn=None, activation_fn=None)
            else:
                dm1 = None
                dm2 = None
                dm3 = None
                dm4 = None

            end_points = utils.convert_collection_to_dict(end_points_collection)

            # reorgnize back to original
            masks = [mask1, mask2, mask3, mask4]
            dms = [dm1, dm2, dm3, dm4]

            if do_exp:
                for i, mask in enumerate(masks):
                    src_masks = tf.split(mask, num_source, axis=0)
                    masks[i] = tf.concat(src_masks, axis=3)
            if do_dm:
                for i, dm in enumerate(dms):
                    src_dms = tf.split(dm, num_source, axis=0)
                    dms[i] = tf.concat(src_dms, axis=3)

            return pose_final, masks, dms, end_points



def dense_motion_u_net(tgt_image,
                       src_image_seq,
                       tgt_depth,
                       src_depth_seq,
                       is_training=True,
                       in_size=[128, 416],
                       reuse=False):

    with_depth = (tgt_depth is not None) and (src_depth_seq is not None)

    # reorgnize images to pairs
    tgt_image = tf.image.resize_bilinear(tgt_image, in_size)
    tgt_depth = tf.image.resize_bilinear(tgt_depth, in_size)

    input_image_pairs = []
    for i in range(len(src_image_seq)):
        src_image = tf.image.resize_bilinear(src_image_seq[i], in_size)
        if with_depth:
            src_depth = tf.image.resize_bilinear(src_depth_seq[i], in_size)
            input_image_pairs.append(
                    tf.concat([tgt_image, src_image, tgt_depth, src_depth], axis=3))
        else:
            input_image_pairs.append(tf.concat([tgt_image, src_image], axis=3))

    inputs = tf.concat(input_image_pairs, axis=0)
    batch_norm_params = {'is_training': is_training}
    num_source = len(src_image_seq)

    with tf.variable_scope('dense_motion_u_net', reuse=reuse) as sc:
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

            ## Dense motion specific layers
            with tf.variable_scope('dm'):
                upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                upcnv4 = tf.concat([upcnv4, cnv3], axis=3)
                dm4 = slim.conv2d(upcnv4,  3, [3, 3], stride=1, scope='dm4',
                    normalizer_fn=None, activation_fn=None)

                upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                cnv3_shape = upcnv3.get_shape().as_list()
                dm4_up = tf.image.resize_nearest_neighbor(dm4, [cnv3_shape[1], cnv3_shape[2]])
                upcnv3 = tf.concat([upcnv3, cnv2, dm4_up], axis=3)
                dm3 = slim.conv2d(upcnv3, 3, [3, 3], stride=1, scope='dm3',
                    normalizer_fn=None, activation_fn=None)

                upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                cnv2_shape = upcnv2.get_shape().as_list()
                dm3_up = tf.image.resize_nearest_neighbor(dm3, [cnv2_shape[1], cnv2_shape[2]])
                upcnv2 = tf.concat([upcnv2, cnv1, dm3_up], axis=3)
                dm2 = slim.conv2d(upcnv2, 3, [3, 3], stride=1, scope='dm2',
                    normalizer_fn=None, activation_fn=None)

                upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                cnv1_shape = upcnv1.get_shape().as_list()
                dm2_up = tf.image.resize_nearest_neighbor(
                        dm2, [cnv1_shape[1], cnv1_shape[2]])
                upcnv1 = tf.concat([upcnv1, dm2_up], axis=3)
                dm1 = slim.conv2d(upcnv1, 3, [7, 7], stride=1, scope='dm1',
                    normalizer_fn=None, activation_fn=None)

            end_points = utils.convert_collection_to_dict(end_points_collection)

            # reorgnize back to original
            dms = [dm1, dm2, dm3, dm4]
            for i, dm in enumerate(dms):
                src_dms = tf.split(dm, num_source, axis=0)
                dms[i] = tf.concat(src_dms, axis=3)

            return dms, end_points


def dense_motion_pwc_net(tgt_image,
                         src_image_seq,
                         tgt_depth,
                         src_depth_seq,
                         with_depth=True,
                         is_training=True,
                         in_size=[128, 416],
                         reuse=False):

    H, W = map(int, tgt_image.get_shape()[1:3])

    # reorgnize images to pairs
    tgt_image = tf.image.resize_bilinear(tgt_image, in_size)
    tgt_depth = tf.image.resize_bilinear(tgt_depth, in_size)

    input_image_pairs = []
    for i in range(len(src_image_seq)):
        src_image = tf.image.resize_bilinear(src_image_seq[i], in_size)
        src_depth = tf.image.resize_bilinear(src_depth_seq[i], in_size)
        input_image_pairs.append(
                tf.concat([tgt_image, tgt_depth, src_image, src_depth], axis=3))

    inputs = tf.concat(input_image_pairs, axis=0)
    num_source = len(src_image_seq)
    channel = 3

    with tf.variable_scope('dense_motion_pwc_net', reuse=reuse):
        inputs1, inputs2 = tf.split(inputs, 2, axis=3)
        pyramid1 = feature_pyramid(inputs1, reuse=reuse)
        pyramid2 = feature_pyramid(inputs2, reuse=True)

        pyramid_num = len(pyramid1)
        flows = []
        # pyramid2_warp = None
        flow2next = None

        for i in range(pyramid_num - 1, -1, -1)[:-1]:
            cv = cost_volumn(pyramid1[i], pyramid2[i], d=4)
            if i < pyramid_num - 1:
                cv = tf.concat([cv, pyramid1[i], flow2next], axis=3)

            flow, feat = motion_decoder_dc(cv, level=i+1,
                    out_channel=channel)

            # flow = flow + flow2next if flow2next is not None else flow
            if i == 1:
                flow = context_net(tf.concat([flow, feat], axis=3), channel) + flow
                flows.append(flow)
                break

            flows.append(flow)
            # resize for next level
            curr_image_sz = [int(H/(2**i)), int((W/(2**i)))]
            flow2next = tf.image.resize_bilinear(flow, curr_image_sz) * 2.0
            # pyramid2_warp = trans.transformer(pyramid2[i-1], flow2next, [H/(2**i), W/(2**i)])

        dms = flows[::-1]
        # reorgnize back to original
        # dms = []
        # depth_tgt = tf.slice(inputs1, [0, 0, 0, ], [-1, -1, -1, -1])
        # depth_src = tf.slice(inputs2, [0, 0, 0, ], [-1, -1, -1, -1])

        # for i, flow in enumerate(flows):
        #     sz = [H/(2**i), (W/(2**i))]
        #     curr_tgt_depth = tf.image.resize_bilinear(tgt_depth,sz)
        #     curr_src_depth = tf.image.resize_bilinear(tgt_depth,sz)
        #     dm = depth2_dense_motion(flow, curr_tgt_depth, depth2)
        #     dms.append(dm)
        # dms = dms[::-1]

        for i, dm in enumerate(dms[:4]):
            src_dms = tf.split(dm, num_source, axis=0)
            dms[i] = tf.concat(src_dms, axis=3)
            h, w = map(int, dms[i].get_shape()[1:3])
            dms[i] = tf.image.resize_bilinear(dms[i], [4 * h, 4 * w])

        return dms[:4]


def disp_net(tgt_image, is_training=True, do_edge=False, reuse=False):
    batch_norm_params = {'is_training': is_training, 'decay':0.999}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    tgt_image = tf.image.resize_bilinear(tgt_image, [127, 415])

    with tf.variable_scope('depth_net', reuse=reuse) as sc:
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

