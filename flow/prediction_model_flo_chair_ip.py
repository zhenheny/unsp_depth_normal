from __future__ import division
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""
"""use directly from flow data"""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.ops import init_ops
from optical_flow_warp import transformer
from optical_flow_warp_old import transformer_old

#from correlation import correlation
def blur(image):
  batch_size, img_height, img_width, color_channels = map(int, image.get_shape()[0:4])
  kernel = np.array([1., 2., 1., 2., 4., 2., 1., 2., 1.], dtype=np.float32) / 16.0
  kernel = kernel.reshape((3, 3, 1, 1))
  kernel = tf.constant(kernel, shape=(3, 3, 1, 1), 
                       name='gaussian_kernel', verify_shape=True)

  blur_image = tf.nn.depthwise_conv2d(tf.pad(image, [[0,0], [1,1], [1,1],[0,0]], "SYMMETRIC"), tf.tile(kernel, [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'VALID')
  return blur_image

def down_sample(image, to_blur=True):
  batch_size, img_height, img_width, color_channels = map(int, image.get_shape()[0:4])
  if to_blur:
    image = blur(image)
  return tf.image.resize_bicubic(image, [int(img_height/2), int(img_width/2)])

def get_pyrimad(image):
  image2 = down_sample(down_sample(image))
  image3 = down_sample(image2)
  image4 = down_sample(image3)
  image5 = down_sample(image4)
  image6 = down_sample(image5)

  return image2, image3, image4, image5, image6

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_bilinear(inputs, [rH.value, rW.value])


def leaky_relu(_x, alpha=0.1):
  pos = tf.nn.relu(_x)
  neg = alpha * (_x - abs(_x)) * 0.5

  return pos + neg

def autoencoder(image, reuse_scope=False, trainable=True):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      reuse=reuse_scope,
                      trainable=trainable,
                      variables_collections=["ae"]):
    cnv1  = slim.conv2d(image, 16,  [3, 3], stride=2, scope='cnv1_ae')
    cnv2  = slim.conv2d(cnv1, 32,  [3, 3], stride=2, scope='cnv2_ae')
    cnv3  = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3_ae')
    cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4_ae')
    cnv5  = slim.conv2d(cnv4, 128, [3, 3], stride=2, scope='cnv5_ae')
    cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6_ae')

    deconv5 = slim.conv2d_transpose(cnv6, 128, [4, 4], stride=2, scope='deconv5_ae')
    deconv4 = slim.conv2d_transpose(deconv5, 128, [4, 4], stride=2, scope='deconv4_ae')
    deconv3 = slim.conv2d_transpose(deconv4, 64, [4, 4], stride=2, scope='deconv3_ae')
    deconv2 = slim.conv2d_transpose(deconv3, 32, [4, 4], stride=2, scope='deconv2_ae')
    deconv1 = slim.conv2d_transpose(deconv2, 16, [4, 4], stride=2, scope='deconv1_ae')
    recon   = slim.conv2d_transpose(deconv1, 3, [4, 4], stride=2, scope='recon_ae', activation_fn=tf.nn.sigmoid)

    return recon, [cnv2, cnv3, cnv4, cnv5, cnv6]

def decoder(feature, reuse_scope=True, trainable=True, level=None):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      reuse=reuse_scope,
                      trainable=trainable,
                      variables_collections=["ae"]):
    if level == 5:
      feature = slim.conv2d(feature, 256, [3, 3], stride=2, scope='cnv6_ae')
    deconv5 = slim.conv2d_transpose(feature, 128, [4, 4], stride=2, scope='deconv5_ae')
    deconv4 = slim.conv2d_transpose(deconv5, 128, [4, 4], stride=2, scope='deconv4_ae')
    deconv3 = slim.conv2d_transpose(deconv4, 64, [4, 4], stride=2, scope='deconv3_ae')
    deconv2 = slim.conv2d_transpose(deconv3, 32, [4, 4], stride=2, scope='deconv2_ae')
    deconv1 = slim.conv2d_transpose(deconv2, 16, [4, 4], stride=2, scope='deconv1_ae')
    recon   = slim.conv2d_transpose(deconv1, 3, [4, 4], stride=2, scope='recon_ae', activation_fn=tf.nn.sigmoid)

    return recon
  
def sub_model(inputs, level):
#   batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
#   inputs = tf.concat([image1, image1_warp, flo], axis=3)
#   inputs.set_shape([batch_size, H, W, 8])
  #############################
  scale = 1
  
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      #weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=tf.nn.relu):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(inputs, 32*scale,  [7, 7], stride=1, scope='cnv1_'+str(level))
      cnv2  = slim.conv2d(cnv1, 64*scale,  [7, 7], stride=1, scope='cnv2_'+str(level))
      cnv3  = slim.conv2d(cnv2, 32*scale, [7, 7], stride=1, scope='cnv3_'+str(level))
      cnv4 = slim.conv2d(cnv3,  16*scale, [7, 7], stride=1, scope='cnv4_'+str(level))
      cnv5  = slim.conv2d(cnv4, 2*scale, [7, 7], stride=1, scope='cnv5_'+str(level))
      
      return cnv5

def construct_model_simple(image1, image2, image1_pyrimad, image2_pyrimad, is_training=True):
  """Build convolutional lstm video predictor using STP, CDNA, or DNA.

  Args:
    images: tensor of ground truth image sequences
    actions: tensor of action sequences
    states: tensor of ground truth state sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    use_state: True to include state and action in prediction
    num_masks: the number of different pixel motion predictions (and
               the number of masks for each of those predictions)
    stp: True to use Spatial Transformer Predictor (STP)
    cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
    dna: True to use Dynamic Neural Advection (DNA)
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames
    gen_states: predicted future states

  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """

  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
  image1 = image1
  image2 = image2
  rand = tf.random_uniform([batch_size, H, W, 1])
  image1 = tf.where(tf.tile(rand < 0.2, [1,1,1,color_channels]), tf.zeros_like(image1), image1)
  mask = tf.where(rand < 0.2, tf.ones_like(rand), tf.zeros_like(rand))
  
  images = tf.concat([image1, image2, mask], axis=3)
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  #############################
  
  batch_norm_params = {'is_training': is_training}

  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"]):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(images, 64,  [7, 7], stride=2, scope='cnv1')
      cnv2  = slim.conv2d(cnv1, 128,  [5, 5], stride=2, scope='cnv2')
      cnv3  = slim.conv2d(cnv2, 256, [5, 5], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3,  256, [3, 3], stride=1, scope='cnv3b')
      cnv4  = slim.conv2d(cnv3b, 512, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4,  512, [3, 3], stride=1, scope='cnv4b')
      cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
      cnv6  = slim.conv2d(cnv5b, 1024, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6,  1024, [3, 3], stride=1, scope='cnv6b')
      
      flow6  =  slim.conv2d(cnv6b, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow6')
      #image1_6p, effect_mask6 = transformer(image2_6, 20*flow6/64.0, [H/64, W/64], image1_6)
      image1_6p = transformer_old(image2_6, 20*flow6/64.0, [H/64, W/64])
      
      deconv5 = slim.conv2d_transpose(cnv6b, 512, [4, 4], stride=2, scope='deconv5', weights_regularizer=None)
      flow6to5 = tf.image.resize_bilinear(flow6, [H/(2**5), (W/(2**5))])
      
      concat5 = tf.concat([cnv5b, deconv5, flow6to5], axis=3)
      flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5') + flow6to5
      #image1_5p, effect_mask5 = transformer(image2_5, 20*flow5/32.0, [H/32, W/32], image1_5)
      image1_5p = transformer_old(image2_5, 20*flow5/32.0, [H/32, W/32])
      
      deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], stride=2, scope='deconv4', weights_regularizer=None)
      flow5to4 = tf.image.resize_bilinear(flow5, [H/(2**4), (W/(2**4))])

      concat4 = tf.concat([cnv4b, deconv4, flow5to4], axis=3)
      flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4') + flow5to4
      #image1_4p, effect_mask4 = transformer(image2_4, 20*flow4/16.0, [H/16, W/16], image1_4)
      image1_4p = transformer_old(image2_4, 20*flow4/16.0, [H/16, W/16])
      
      deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], stride=2, scope='deconv3', weights_regularizer=None)
      flow4to3 = tf.image.resize_bilinear(flow4, [H/(2**3), (W/(2**3))])
      
      concat3 = tf.concat([cnv3b, deconv3, flow4to3], axis=3)
      flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3') + flow4to3
      #image1_3p, effect_mask3 = transformer(image2_3, 20*flow3/8.0, [H/8, W/8], image1_3)
      image1_3p = transformer_old(image2_3, 20*flow3/8.0, [H/8, W/8])

      deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], stride=2, scope='deconv2', weights_regularizer=None)
      flow3to2 = tf.image.resize_bilinear(flow3, [H/(2**2), (W/(2**2))])
      
      concat2 = tf.concat([cnv2, deconv2, flow3to2], axis=3)
      flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2') + flow3to2
      
      #image1_2p, effect_mask2 = transformer(image2_2, 20*flow2/4.0, [H/4, W/4], image1_2)
      image1_2p = transformer_old(image2_2, 20*flow2/4.0, [H/4, W/4])
      
      return flow2, flow3, flow4, flow5, flow6, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]

def construct_model(image1, image2, image1_pyrimad, image2_pyrimad, is_training=True):
  """Build convolutional lstm video predictor using STP, CDNA, or DNA.

  Args:
    images: tensor of ground truth image sequences
    actions: tensor of action sequences
    states: tensor of ground truth state sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    use_state: True to include state and action in prediction
    num_masks: the number of different pixel motion predictions (and
               the number of masks for each of those predictions)
    stp: True to use Spatial Transformer Predictor (STP)
    cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
    dna: True to use Dynamic Neural Advection (DNA)
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames
    gen_states: predicted future states

  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """

  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
  images = tf.concat([image1, image2], axis=3)
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  #############################
  
  batch_norm_params = {'is_training': is_training}

  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"]):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(images, 64,  [7, 7], stride=2, scope='cnv1')
      cnv2  = slim.conv2d(cnv1, 128,  [5, 5], stride=2, scope='cnv2')
      cnv3  = slim.conv2d(cnv2, 256, [5, 5], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3,  256, [3, 3], stride=1, scope='cnv3b')
      cnv4  = slim.conv2d(cnv3b, 512, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4,  512, [3, 3], stride=1, scope='cnv4b')
      cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
      cnv6  = slim.conv2d(cnv5b, 1024, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6,  1024, [3, 3], stride=1, scope='cnv6b')
      
      flow6  =  slim.conv2d(cnv6b, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow6')
      image1_6p, effect_mask6 = transformer(image2_6, 20*flow6/64.0, [H/64, W/64], image1_6)
      #image1_6p = transformer_old(image2_6, 20*flow6/64.0, [H/64, W/64])
      
      deconv5 = slim.conv2d_transpose(cnv6b, 512, [4, 4], stride=2, scope='deconv5', weights_regularizer=None)
      flow6to5 = tf.image.resize_bilinear(flow6, [H/(2**5), (W/(2**5))])
      feature6to5 = tf.image.resize_bilinear(tf.concat([image1_6[:,:,:,0:3], image2_6[:,:,:,0:3], image1_6p[:,:,:,0:3], image1_6[:,:,:,0:3]-image1_6p[:,:,:,0:3]], axis=3), [H/(2**5), W/(2**5)])
      feature6to5.set_shape([batch_size, H/(2**5), W/(2**5), color_channels*4])
      
      concat5 = tf.concat([cnv5b, deconv5, sub_model(feature6to5, level=5)], axis=3)
      #concat5 = tf.concat([cnv5b, deconv5, flow6to5], axis=3)
      flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5') + flow6to5
      image1_5p, effect_mask5 = transformer(image2_5, 20*flow5/32.0, [H/32, W/32], image1_5)
      #image1_5p = transformer_old(image2_5, 20*flow5/32.0, [H/32, W/32])
      
      deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], stride=2, scope='deconv4', weights_regularizer=None)
      flow5to4 = tf.image.resize_bilinear(flow5, [H/(2**4), (W/(2**4))])
      feature5to4 = tf.image.resize_bilinear(tf.concat([image1_5[:,:,:,0:3], image2_5[:,:,:,0:3], image1_5p[:,:,:,0:3], image1_5[:,:,:,0:3]-image1_5p[:,:,:,0:3]], axis=3), [H/(2**4), (W/(2**4))])
      feature5to4.set_shape([batch_size, H/(2**4), W/(2**4), color_channels*4])
      
      concat4 = tf.concat([cnv4b, deconv4, sub_model(feature5to4, level=4)], axis=3)
      #concat4 = tf.concat([cnv4b, deconv4, flow5to4], axis=3)
      flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4') + flow5to4
      image1_4p, effect_mask4 = transformer(image2_4, 20*flow4/16.0, [H/16, W/16], image1_4)
      #image1_4p = transformer_old(image2_4, 20*flow4/16.0, [H/16, W/16])
      
      deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], stride=2, scope='deconv3', weights_regularizer=None)
      flow4to3 = tf.image.resize_bilinear(flow4, [H/(2**3), (W/(2**3))])
      feature4to3 = tf.image.resize_bilinear(tf.concat([image1_4[:,:,:,0:3], image2_4[:,:,:,0:3], image1_4p[:,:,:,0:3], image1_4[:,:,:,0:3]-image1_4p[:,:,:,0:3]], axis=3), [H/(2**3), (W/(2**3))])
      feature4to3.set_shape([batch_size, H/(2**3), W/(2**3), color_channels*4])
      
      concat3 = tf.concat([cnv3b, deconv3, sub_model(feature4to3, level=3)], axis=3)
      #concat3 = tf.concat([cnv3b, deconv3, flow4to3], axis=3)
      flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3') + flow4to3
      image1_3p, effect_mask3 = transformer(image2_3, 20*flow3/8.0, [H/8, W/8], image1_3)
      #image1_3p = transformer_old(image2_3, 20*flow3/8.0, [H/8, W/8])

      deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], stride=2, scope='deconv2', weights_regularizer=None)
      flow3to2 = tf.image.resize_bilinear(flow3, [H/(2**2), (W/(2**2))])
      feature3to2 = tf.image.resize_bilinear(tf.concat([image1_3[:,:,:,0:3], image2_3[:,:,:,0:3], image1_3p[:,:,:,0:3], image1_3[:,:,:,0:3]-image1_3p[:,:,:,0:3]], axis=3), [H/(2**2), (W/(2**2))])
      feature3to2.set_shape([batch_size, H/(2**2), W/(2**2), color_channels*4])
      
      concat2 = tf.concat([cnv2, deconv2, sub_model(feature3to2, level=2)], axis=3)
      #concat2 = tf.concat([cnv2, deconv2, flow3to2], axis=3)
      flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2') + flow3to2
      
      image1_2p, effect_mask2 = transformer(image2_2, 20*flow2/4.0, [H/4, W/4], image1_2)
      #image1_2p = transformer_old(image2_2, 20*flow2/4.0, [H/4, W/4])
      
      return flow2, flow3, flow4, flow5, flow6, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]


def construct_model_dropout(image1, image2, image1_pyrimad, image2_pyrimad, is_training=True):
  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
  #image1  = tf.nn.dropout(image1, keep_prob=0.8, noise_shape=[batch_size, H, W, 1])
  image1 = image1 - 0.5
  image2 = image2 - 0.5
  rand = tf.random_uniform([batch_size, H, W, 1])
  image1 = tf.where(tf.tile(rand < 0.2, [1,1,1,color_channels]), tf.zeros_like(image1), image1)
  
  images = tf.concat([image1, image2], axis=3)
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  #############################
  
  batch_norm_params = {'is_training': is_training}

  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"]):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(images, 64,  [7, 7], stride=2, scope='cnv1')
      cnv2  = slim.conv2d(cnv1, 128,  [5, 5], stride=2, scope='cnv2')
      cnv3  = slim.conv2d(cnv2, 256, [5, 5], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3,  256, [3, 3], stride=1, scope='cnv3b')
      cnv4  = slim.conv2d(cnv3b, 512, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4,  512, [3, 3], stride=1, scope='cnv4b')
      cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
      cnv6  = slim.conv2d(cnv5b, 1024, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6,  1024, [3, 3], stride=1, scope='cnv6b')
      
      flow6  =  slim.conv2d(cnv6b, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow6')
      image1_6p, effect_mask6 = transformer(image2_6, 20*flow6/64.0, [H/64, W/64], image1_6)
      #image1_6p = transformer_old(image2_6, 20*flow6/64.0, [H/64, W/64])
      
      deconv5 = slim.conv2d_transpose(cnv6b, 512, [4, 4], stride=2, scope='deconv5', weights_regularizer=None)
      flow6to5 = tf.image.resize_bilinear(flow6, [H/(2**5), (W/(2**5))])
      feature6to5 = tf.image.resize_bilinear(tf.concat([image1_6[:,:,:,0:3], image2_6[:,:,:,0:3], image1_6p[:,:,:,0:3], image1_6[:,:,:,0:3]-image1_6p[:,:,:,0:3]], axis=3), [H/(2**5), W/(2**5)])
      feature6to5.set_shape([batch_size, H/(2**5), W/(2**5), color_channels*4])
      
      concat5 = tf.concat([cnv5b, deconv5, sub_model(feature6to5, level=5)], axis=3)
      flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5') + flow6to5
      image1_5p, effect_mask5 = transformer(image2_5, 20*flow5/32.0, [H/32, W/32], image1_5)
      #image1_5p = transformer_old(image2_5, 20*flow5/32.0, [H/32, W/32])
      
      deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], stride=2, scope='deconv4', weights_regularizer=None)
      flow5to4 = tf.image.resize_bilinear(flow5, [H/(2**4), (W/(2**4))])
      feature5to4 = tf.image.resize_bilinear(tf.concat([image1_5[:,:,:,0:3], image2_5[:,:,:,0:3], image1_5p[:,:,:,0:3], image1_5[:,:,:,0:3]-image1_5p[:,:,:,0:3]], axis=3), [H/(2**4), (W/(2**4))])
      feature5to4.set_shape([batch_size, H/(2**4), W/(2**4), color_channels*4])
      
      concat4 = tf.concat([cnv4b, deconv4, sub_model(feature5to4, level=4)], axis=3)
      flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4') + flow5to4
      image1_4p, effect_mask4 = transformer(image2_4, 20*flow4/16.0, [H/16, W/16], image1_4)
      #image1_4p = transformer_old(image2_4, 20*flow4/16.0, [H/16, W/16])
      
      deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], stride=2, scope='deconv3', weights_regularizer=None)
      flow4to3 = tf.image.resize_bilinear(flow4, [H/(2**3), (W/(2**3))])
      feature4to3 = tf.image.resize_bilinear(tf.concat([image1_4[:,:,:,0:3], image2_4[:,:,:,0:3], image1_4p[:,:,:,0:3], image1_4[:,:,:,0:3]-image1_4p[:,:,:,0:3]], axis=3), [H/(2**3), (W/(2**3))])
      feature4to3.set_shape([batch_size, H/(2**3), W/(2**3), color_channels*4])
      
      concat3 = tf.concat([cnv3b, deconv3, sub_model(feature4to3, level=3)], axis=3)
      flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3') + flow4to3
      image1_3p, effect_mask3 = transformer(image2_3, 20*flow3/8.0, [H/8, W/8], image1_3)
      #image1_3p = transformer_old(image2_3, 20*flow3/8.0, [H/8, W/8])

      deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], stride=2, scope='deconv2', weights_regularizer=None)
      flow3to2 = tf.image.resize_bilinear(flow3, [H/(2**2), (W/(2**2))])
      feature3to2 = tf.image.resize_bilinear(tf.concat([image1_3[:,:,:,0:3], image2_3[:,:,:,0:3], image1_3p[:,:,:,0:3], image1_3[:,:,:,0:3]-image1_3p[:,:,:,0:3]], axis=3), [H/(2**2), (W/(2**2))])
      feature3to2.set_shape([batch_size, H/(2**2), W/(2**2), color_channels*4])
      
      concat2 = tf.concat([cnv2, deconv2, sub_model(feature3to2, level=2)], axis=3)
      flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2') + flow3to2
      
      image1_2p, effect_mask2 = transformer(image2_2, 20*flow2/4.0, [H/4, W/4], image1_2)
      #image1_2p = transformer_old(image2_2, 20*flow2/4.0, [H/4, W/4])
      
      return flow2, flow3, flow4, flow5, flow6, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]
    
def seg_net(image):
    H = image.get_shape()[1].value
    W = image.get_shape()[2].value
    
    image2 = tf.image.resize_bicubic(image, [449, 769])
    
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        activation_fn=tf.nn.relu):
        cnv1  = slim.conv2d(image2, 32,  [7, 7], stride=2, scope='cnv1_seg')
        cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b_seg')
        cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2_seg')
        cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b_seg')
        cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3_seg')
        cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b_seg')
        cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4_seg')
        cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b_seg')
        cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5_seg')
        cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b_seg')
        cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6_seg')
        cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b_seg')
        
        seg6 = slim.conv2d(cnv6b, 2, [3, 3], stride=1, activation_fn=tf.sigmoid, scope='seg6_seg')
        print(map(int, seg6.get_shape()[0:4]))
        seg6 = tf.image.resize_bilinear(seg6, [H/2**6, W/2**6])
        print(map(int, seg6.get_shape()[0:4]))
        
        upcnv5 = slim.conv2d_transpose(cnv6b, 256, [4, 4], stride=2, scope='upcnv5_seg')
        #i5_in  = tf.concat([upcnv5, cnv5b, tf.image.resize_bilinear(seg6, [np.int(H/2**5), np.int(W/2**5)])], axis=3)
        i5_in  = tf.concat([resize_like(upcnv5, cnv5b), cnv5b], axis=3)
        icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5_seg')
        seg5 = slim.conv2d(icnv5, 2, [3, 3], stride=1, activation_fn=tf.sigmoid, scope='seg5_seg')
        print(map(int, seg5.get_shape()[0:4]))
        seg5 = tf.image.resize_bilinear(seg5, [H/2**5, W/2**5])
        print(map(int, seg5.get_shape()[0:4]))
        
        upcnv4 = slim.conv2d_transpose(icnv5, 128, [4, 4], stride=2, scope='upcnv4_seg')
        #i4_in  = tf.concat([upcnv4, cnv4b, tf.image.resize_bilinear(seg5, [np.int(H/2**4), np.int(W/2**4)])], axis=3)
        i4_in  = tf.concat([resize_like(upcnv4, cnv4b), cnv4b], axis=3)
        icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4_seg')
        seg4 = slim.conv2d(icnv4, 2, [3, 3], stride=1, activation_fn=tf.sigmoid, scope='seg4_seg')
        print(map(int, seg4.get_shape()[0:4]))
        seg4 = tf.image.resize_bilinear(seg4, [H/2**4, W/2**4])
        print(map(int, seg4.get_shape()[0:4]))
        
        upcnv3 = slim.conv2d_transpose(icnv4, 64,  [4, 4], stride=2, scope='upcnv3_seg')
        #i3_in  = tf.concat([upcnv3, cnv3b, tf.image.resize_bilinear(seg4, [np.int(H/2**3), np.int(W/2**3)])], axis=3)
        i3_in  = tf.concat([resize_like(upcnv3, cnv3b), cnv3b], axis=3)
        icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3_seg')
        seg3 = slim.conv2d(icnv3, 2, [3, 3], stride=1, activation_fn=tf.sigmoid, scope='seg3_seg')
        print(map(int, seg3.get_shape()[0:4]))
        seg3 = tf.image.resize_bilinear(seg3, [H/2**3, W/2**3])
        print(map(int, seg3.get_shape()[0:4]))
        
        upcnv2 = slim.conv2d_transpose(icnv3, 32,  [4, 4], stride=2, scope='upcnv2_seg')
        #i2_in  = tf.concat([upcnv2, cnv2b, tf.image.resize_bilinear(seg3, [np.int(H/2**2), np.int(W/2**2)])], axis=3)
        i2_in  = tf.concat([resize_like(upcnv2, cnv2b), cnv2b], axis=3)
        icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2_seg')
        seg2 = slim.conv2d(icnv2, 2, [3, 3], stride=1, activation_fn=tf.sigmoid, scope='seg2_seg')
        print(map(int, seg2.get_shape()[0:4]))
        seg2 = tf.image.resize_bilinear(seg2, [H/2**2, W/2**2])
        print(map(int, seg2.get_shape()[0:4]))
        
        #seg3 = tf.nn.max_pool(seg2, [1,2,2,1], [1,2,2,1], padding="VALID")
        #seg4 = tf.nn.max_pool(seg3, [1,2,2,1], [1,2,2,1], padding="VALID")
        #seg5 = tf.nn.max_pool(seg4, [1,2,2,1], [1,2,2,1], padding="VALID")
        #seg6 = tf.nn.max_pool(seg5, [1,2,2,1], [1,2,2,1], padding="VALID")
        
        return seg2, seg3, seg4, seg5, seg6

def feature_pyramid(image, reuse):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"],
                      reuse=reuse):
                      #outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(image, 16, [3, 3], stride=2, scope="cnv1")
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
  batch_size, H, W, feature_num = map(int, feature1.get_shape()[0:4])  
  feature2 = tf.pad(feature2, [[0,0], [d,d], [d,d],[0,0]], "SYMMETRIC")
  cv = []
  for i in range(2*d+1):
    for j in range(2*d+1):
      cv.append(tf.reduce_mean(feature1*feature2[:, i:(i+H), j:(j+W), :], axis=3, keep_dims=True))
  return tf.concat(cv, axis=3)

# def cost_volumn(feature1, feature2, d=4):
#   return correlation(feature1, feature2, kernel_size=1, max_displacement=d, stride_1=1, stride_2=1, padding=d)

def optical_flow_decoder(inputs, level):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu):
                      #outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inputs, 128, [3, 3], stride=1, scope="cnv1_fd_"+str(level))
      cnv2 = slim.conv2d(cnv1, 128, [3, 3], stride=1, scope="cnv2_fd_"+str(level))
      cnv3 = slim.conv2d(cnv2, 96, [3, 3], stride=1, scope="cnv3_fd_"+str(level))
      cnv4 = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope="cnv4_fd_"+str(level))
      cnv5 = slim.conv2d(cnv4, 32, [3, 3], stride=1, scope="cnv5_fd_"+str(level))
      flow = slim.conv2d(cnv5, 2, [3, 3], stride=1, scope="cnv6_fd_"+str(level), activation_fn=None)
      
      return flow
    
def optical_flow_decoder_dc(inputs, level):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu):
                      #outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inputs, 128, [3, 3], stride=1, scope="cnv1_fd_"+str(level))
      cnv2 = slim.conv2d(cnv1, 128, [3, 3], stride=1, scope="cnv2_fd_"+str(level))
      cnv3 = slim.conv2d(tf.concat([cnv1, cnv2], axis=3), 96, [3, 3], stride=1, scope="cnv3_fd_"+str(level))
      cnv4 = slim.conv2d(tf.concat([cnv2, cnv3], axis=3), 64, [3, 3], stride=1, scope="cnv4_fd_"+str(level))
      cnv5 = slim.conv2d(tf.concat([cnv3, cnv4], axis=3), 32, [3, 3], stride=1, scope="cnv5_fd_"+str(level))
      flow = slim.conv2d(tf.concat([cnv4, cnv5], axis=3), 2, [3, 3], stride=1, scope="cnv6_fd_"+str(level), activation_fn=None)
      
      return flow, cnv5
    
def context_net(inputs):
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

      flow = slim.conv2d(cnv6, 2, [3, 3], rate=1, scope="cnv7_cn", activation_fn=None)
      return flow
    
def construct_model_pwc(image1, image2, image1_pyrimad, image2_pyrimad):

  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])  
  
  #############################
  feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = feature_pyramid(image1, reuse=False)
  feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = feature_pyramid(image2, reuse=True)
  
  cv6 = cost_volumn(feature1_6, feature2_6, d=4)
  flow6 = optical_flow_decoder(cv6, level=6)
  
  flow6to5 = tf.image.resize_bilinear(flow6, [H/(2**5), (W/(2**5))]) * 2.0
  feature2_5w = transformer_old(feature2_5, flow6to5, [H/32, W/32])
  cv5 = cost_volumn(feature1_5, feature2_5w, d=4)
  flow5 = optical_flow_decoder(tf.concat([cv5, feature1_5, flow6to5], axis=3), level=5) + flow6to5
  
  flow5to4 = tf.image.resize_bilinear(flow5, [H/(2**4), (W/(2**4))]) * 2.0
  feature2_4w = transformer_old(feature2_4, flow5to4, [H/16, W/16])
  cv4 = cost_volumn(feature1_4, feature2_4w, d=4)
  flow4 = optical_flow_decoder(tf.concat([cv4, feature1_4, flow5to4], axis=3), level=4) + flow5to4
  
  flow4to3 = tf.image.resize_bilinear(flow4, [H/(2**3), (W/(2**3))]) * 2.0
  feature2_3w = transformer_old(feature2_3, flow4to3, [H/8, W/8])
  cv3 = cost_volumn(feature1_3, feature2_3w, d=4)
  flow3 = optical_flow_decoder(tf.concat([cv3, feature1_3, flow4to3], axis=3), level=3) + flow4to3
  
  flow3to2 = tf.image.resize_bilinear(flow3, [H/(2**2), (W/(2**2))]) * 2.0
  feature2_2w = transformer_old(feature2_2, flow3to2, [H/4, W/4])
  cv2 = cost_volumn(feature1_2, feature2_2w, d=4)
  flow2 = optical_flow_decoder(tf.concat([cv2, feature1_2, flow3to2], axis=3), level=2) + flow3to2
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  image1_6p = transformer_old(image2_6, flow6, [H/64, W/64])
  image1_5p = transformer_old(image2_5, flow5, [H/32, W/32])
  image1_4p = transformer_old(image2_4, flow4, [H/16, W/16])
  image1_3p = transformer_old(image2_3, flow3, [H/8, W/8])
  image1_2p = transformer_old(image2_2, flow2, [H/4, W/4])
  
  return flow2*4.0/20.0, flow3*8.0/20.0, flow4*16.0/20.0, flow5*32.0/20.0, flow6*64.0/20.0, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]

def construct_model_pwc_full(image1, image2):

  image1_pyrimad = get_pyrimad(image1)
  image2_pyrimad = get_pyrimad(image2)

  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])  
  
  #############################
  feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = feature_pyramid(image1, reuse=False)
  feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = feature_pyramid(image2, reuse=True)
  
  cv6 = cost_volumn(feature1_6, feature2_6, d=4)
  flow6, _ = optical_flow_decoder_dc(cv6, level=6)
  
  flow6to5 = tf.image.resize_bilinear(flow6, [int(H/(2**5)), int(W/(2**5))]) * 2.0
  feature2_5w = transformer_old(feature2_5, flow6to5, [int(H/32), int(W/32)])
  cv5 = cost_volumn(feature1_5, feature2_5w, d=4)
  flow5, _ = optical_flow_decoder_dc(tf.concat([cv5, feature1_5, flow6to5], axis=3), level=5) 
  flow5 = flow5 + flow6to5
  
  flow5to4 = tf.image.resize_bilinear(flow5, [int(H/(2**4)), int(W/(2**4))]) * 2.0
  feature2_4w = transformer_old(feature2_4, flow5to4, [int(H/16), int(W/16)])
  cv4 = cost_volumn(feature1_4, feature2_4w, d=4)
  flow4, _ = optical_flow_decoder_dc(tf.concat([cv4, feature1_4, flow5to4], axis=3), level=4)
  flow4 = flow4 + flow5to4
  
  flow4to3 = tf.image.resize_bilinear(flow4, [int(H/(2**3)), int(W/(2**3))]) * 2.0
  feature2_3w = transformer_old(feature2_3, flow4to3, [int(H/8), int(W/8)])
  cv3 = cost_volumn(feature1_3, feature2_3w, d=4)
  flow3, _ = optical_flow_decoder_dc(tf.concat([cv3, feature1_3, flow4to3], axis=3), level=3)
  flow3 = flow3 + flow4to3
  
  flow3to2 = tf.image.resize_bilinear(flow3, [int(H/(2**2)), int(W/(2**2))]) * 2.0
  feature2_2w = transformer_old(feature2_2, flow3to2, [int(H/4), int(W/4)])
  cv2 = cost_volumn(feature1_2, feature2_2w, d=4)
  flow2_raw, f2 = optical_flow_decoder_dc(tf.concat([cv2, feature1_2, flow3to2], axis=3), level=2) 
  flow2_raw = flow2_raw + flow3to2
  
  flow2 = context_net(tf.concat([flow2_raw, f2], axis=3)) + flow2_raw
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  image1_6p = transformer_old(image2_6, flow6, [int(H/64), int(W/64)])
  image1_5p = transformer_old(image2_5, flow5, [int(H/32), int(W/32)])
  image1_4p = transformer_old(image2_4, flow4, [int(H/16), int(W/16)])
  image1_3p = transformer_old(image2_3, flow3, [int(H/8), int(W/8)])
  image1_2p = transformer_old(image2_2, flow2, [int(H/4), int(W/4)])
  
  return flow2*4.0/20.0, flow3*8.0/20.0, flow4*16.0/20.0, flow5*32.0/20.0, flow6*64.0/20.0, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]


def feature_pyramid_lossnet(image, trainable=True):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      trainable=trainable):
                      #outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(image, 8, [3, 3], stride=2, scope="cnv1_lossft")
      cnv2 = slim.conv2d(cnv1, 16, [3, 3], stride=2, scope="cnv2_lossft")
      #cnv3 = slim.conv2d(cnv2, 32, [3, 3], stride=2, scope="cnv3_lossft")
      #cnv4 = slim.conv2d(cnv3, 32, [3, 3], stride=2, scope="cnv4_lossft")
      #cnv5 = slim.conv2d(cnv4, 32, [3, 3], stride=2, scope="cnv5_lossft")
      #cnv6 = slim.conv2d(cnv5, 32, [3, 3], stride=2, scope="cnv6_lossft")
      
      return cnv2#, cnv3, cnv4, cnv5, cnv6

def smoothness_lossnet(flo, trainable=True):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      trainable=trainable):
                      #outputs_collections=end_points_collection):
      flo_cnv1 = slim.conv2d(flo, 8, [3, 3], stride=1, scope="cnv1_smooth_flo", activation_fn=None)
      flo_cnv2 = slim.conv2d(flo_cnv1, 8, [3, 3], stride=1, scope="cnv2_smooth_flo", activation_fn=None)
      
      #image_cnv1 = slim.conv2d(image, 8, [3, 3], stride=1, scope="cnv1_smooth_image", activation_fn=None)
      return tf.reduce_mean(tf.abs(flo_cnv2), axis=[3], keep_dims=True)


