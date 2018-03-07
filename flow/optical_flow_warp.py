# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf
from tensorflow.python.platform import app
import numpy as np
import math

def transformer(U, flo, out_size, target, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])
    
    
    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            
            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f-1) / 2.0
            y = (y + 1.0)*(height_f-1) / 2.0
            
            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            
            xn1 = x0 - 1
            yn1 = y0 - 1
            x2 = x0 + 2
            y2 = y0 + 2
            
            xn2 = x0 - 2
            yn2 = y0 - 2
            x3 = x0 + 3
            y3 = y0 + 3
            
            xn3 = x0 - 3
            yn3 = y0 - 3
            x4 = x0 + 4
            y4 = y0 + 4
            
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            
            target_flat = tf.reshape(target, tf.stack([-1, channels]))
            target_flat = tf.cast(target_flat, 'float32')
            
            def helper(x0_, x1_, y0_, y1_, x_range, y_range):
              scale = min(x_range, y_range)
              scale_area = x_range * y_range / (scale*scale)
              
              x0_c = tf.clip_by_value(x0_, zero, max_x)
              x1_c = tf.clip_by_value(x1_, zero, max_x)
              
              y0_c = tf.clip_by_value(y0_, zero, max_y)
              y1_c = tf.clip_by_value(y1_, zero, max_y)
              
              base_y0 = base + y0_c*dim2
              base_y1 = base + y1_c*dim2
              
              idx_a = base_y0 + x0_c
              idx_b = base_y1 + x0_c
              idx_c = base_y0 + x1_c
              idx_d = base_y1 + x1_c
              
              Ia = tf.gather(im_flat, idx_a)
              Ib = tf.gather(im_flat, idx_b)
              Ic = tf.gather(im_flat, idx_c)
              Id = tf.gather(im_flat, idx_d)
              
              # and finally calculate interpolated values
              x0_f = tf.cast(x0_, 'float32')
              x1_f = tf.cast(x1_, 'float32')
              y0_f = tf.cast(y0_, 'float32')
              y1_f = tf.cast(y1_, 'float32')
              
              x_mid = (x0_f + x1_f) / 2.0
              y_mid = (y0_f + y1_f) / 2.0
              
              x0_f = (x0_f - x_mid) / scale + x_mid
              x1_f = (x1_f - x_mid) / scale + x_mid
              y0_f = (y0_f - y_mid) / scale + y_mid
              y1_f = (y1_f - y_mid) / scale + y_mid
              
              wa = tf.expand_dims(((x1_f-x) * (y1_f-y) / scale_area), 1)
              wb = tf.expand_dims(((x1_f-x) * (y-y0_f) / scale_area), 1)
              wc = tf.expand_dims(((x-x0_f) * (y1_f-y) / scale_area), 1)
              wd = tf.expand_dims(((x-x0_f) * (y-y0_f) / scale_area), 1)
              output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
              return output, [tf.expand_dims(Ia, axis=-1), 
                              tf.expand_dims(Ib, axis=-1), 
                              tf.expand_dims(Ic, axis=-1), 
                              tf.expand_dims(Id, axis=-1)]
            
            output1, v1 = helper(x0, x1, y0, y1, 1.0, 1.0)
            output2, v2 = helper(x0, x1, yn1, y2, 1.0, 3.0)
            output3, v3 = helper(xn1, x2, y0, y1, 3.0, 1.0)
            output4, v4 = helper(xn1, x2, yn1, y2, 3.0, 3.0)
            
            output5, v5 = helper(xn2, x3, yn2, y3, 5.0, 5.0)
            output6, v6 = helper(x0, x1, yn2, y3, 1.0, 5.0)
            output7, v7 = helper(xn1, x2, yn2, y3, 3.0, 5.0)
            output8, v8 = helper(xn2, x3, y0, y1, 5.0, 1.0)
            output9, v9 = helper(xn2, x3, yn1, y2, 5.0, 3.0)
            
            output10, v10 = helper(xn3, x4, yn3, y4, 7.0, 7.0)
            output11, v11 = helper(xn3, x4, yn1, y2, 7.0, 3.0)
            output12, v12 = helper(xn1, x2, yn3, y4, 3.0, 7.0)
            
            
            candidates = tf.concat(v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12, axis=2)
            
            idx = tf.argmin(tf.reduce_mean(tf.abs(candidates - tf.expand_dims(target_flat, axis=-1)), axis=1, keep_dims=True), axis=2)
            idx = tf.tile(idx, [1, channels])
            

            error_small_pred = tf.tile(tf.reduce_mean(tf.abs(output1 - target_flat), axis=1, keep_dims=True), [1, channels]) < 100.0
            
            return tf.where(tf.logical_or(error_small_pred, tf.logical_and(idx>=0, idx<4)), output1, 
                            tf.where(tf.logical_and(idx>=4, idx<8), output2, 
                                     tf.where(tf.logical_and(idx>=8, idx<12), output3, 
                                              tf.where(tf.logical_and(idx>=12, idx<16), output4, 
                                                       tf.where(tf.logical_and(idx>=16, idx<20), output5,
                                                                tf.where(tf.logical_and(idx>=20, idx<24), output6,
                                                                         tf.where(tf.logical_and(idx>=24, idx<28), output7,
                                                                                  tf.where(tf.logical_and(idx>=28, idx<32), output8, 
                                                                                           tf.where(tf.logical_and(idx>=32, idx<36), output9, 
                                                                                                    tf.where(tf.logical_and(idx>=36, idx<40), output10, 
                                                                                                             tf.where(tf.logical_and(idx>=40, idx<44), output11, output12))))))))))), error_small_pred
            
    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            return x_t, y_t

    def _transform(flo, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            x_t, y_t = _meshgrid(out_height, out_width)
            x_t = tf.expand_dims(x_t, 0)
            x_t = tf.tile(x_t, [num_batch, 1, 1])
            
            y_t = tf.expand_dims(y_t, 0)
            y_t = tf.tile(y_t, [num_batch, 1, 1])
            
            x_s = x_t + flo[:, :, :, 0] / ((out_width-1.0) / 2.0)
            y_s = y_t + flo[:, :, :, 1] / ((out_height-1.0) / 2.0)
            
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed, effect_mask = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            effect_mask = tf.reshape(
                effect_mask, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output, tf.cast(effect_mask[:, :, :, 0:1], 'float32')

    with tf.variable_scope(name):
        output = _transform(flo, U, out_size)
        return output
      

def main(unused_argv):
  sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False))
  
  image = tf.constant([1,2,3,4,5,6,7,8,9], shape=[1, 3, 3, 1], dtype="float32")

  flo = np.zeros((1, 3, 3, 2))
  flo[0, 1, 1, 0] = 1.0
  #flo[0, 1, 1, 1] = 1.0
  flo = tf.constant(flo, dtype="float32")
  
  image2, effect_mask = transformer(image, flo, [3, 3], image)
  
  print(image2.eval(session=sess))
  print(effect_mask.eval(session=sess))
  
if __name__ == '__main__':
  app.run()


