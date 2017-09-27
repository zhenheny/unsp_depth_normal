from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [vec.get_shape().as_list()[0], 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat

def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat

def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, target_image):
    """Inverse warp a source image to the target image plane
       Part of the code modified from  
       https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
    Args:
        img: the source image (where to sample pixels) -- [B, H, W, 3]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones  = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    def _interpolate_ms(im, x, y, out_size, target, name='_interpolate'):
        with tf.variable_scope('_interpolate'):
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])
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
            
            # def helper(x0_, x1_, y0_, y1_, x_range, y_range):
            #     scale = min(x_range, y_range)
            #     scale_area = x_range * y_range / (scale*scale)

            #     x0_c = tf.clip_by_value(x0_, zero, max_x)
            #     x1_c = tf.clip_by_value(x1_, zero, max_x)

            #     y0_c = tf.clip_by_value(y0_, zero, max_y)
            #     y1_c = tf.clip_by_value(y1_, zero, max_y)

            #     base_y0 = base + y0_c*dim2
            #     base_y1 = base + y1_c*dim2

            #     idx_a = base_y0 + x0_c
            #     idx_b = base_y1 + x0_c
            #     idx_c = base_y0 + x1_c
            #     idx_d = base_y1 + x1_c

            #     Ia = tf.gather(im_flat, idx_a)
            #     Ib = tf.gather(im_flat, idx_b)
            #     Ic = tf.gather(im_flat, idx_c)
            #     Id = tf.gather(im_flat, idx_d)

            #     # and finally calculate interpolated values
            #     x0_f = tf.cast(x0_, 'float32')
            #     x1_f = tf.cast(x1_, 'float32')
            #     y0_f = tf.cast(y0_, 'float32')
            #     y1_f = tf.cast(y1_, 'float32')

            #     x_mid = (x0_f + x1_f) / 2.0
            #     y_mid = (y0_f + y1_f) / 2.0

            #     x0_f = (x0_f - x_mid) / scale + x_mid
            #     x1_f = (x1_f - x_mid) / scale + x_mid
            #     y0_f = (y0_f - y_mid) / scale + y_mid
            #     y1_f = (y1_f - y_mid) / scale + y_mid

            #     wa = tf.expand_dims(((x1_f-x) * (y1_f-y) / scale_area), 1)
            #     wb = tf.expand_dims(((x1_f-x) * (y-y0_f) / scale_area), 1)
            #     wc = tf.expand_dims(((x-x0_f) * (y1_f-y) / scale_area), 1)
            #     wd = tf.expand_dims(((x-x0_f) * (y-y0_f) / scale_area), 1)
            #     output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            #     return output, [tf.expand_dims(Ia, axis=-1), 
            #                   tf.expand_dims(Ib, axis=-1), 
            #                   tf.expand_dims(Ic, axis=-1), 
            #                   tf.expand_dims(Id, axis=-1)]
            
            # output1, v1 = helper(x0, x1, y0, y1, 1.0, 1.0)
            # output2, v2 = helper(x0, x1, yn1, y2, 1.0, 3.0)
            # output3, v3 = helper(xn1, x2, y0, y1, 3.0, 1.0)
            # output4, v4 = helper(xn1, x2, yn1, y2, 3.0, 3.0)
            
            # output5, v5 = helper(xn2, x3, yn2, y3, 5.0, 5.0)
            # output6, v6 = helper(x0, x1, yn2, y3, 1.0, 5.0)
            # output7, v7 = helper(xn1, x2, yn2, y3, 3.0, 5.0)
            # output8, v8 = helper(xn2, x3, y0, y1, 5.0, 1.0)
            # output9, v9 = helper(xn2, x3, yn1, y2, 5.0, 3.0)
            
            # output10, v10 = helper(xn3, x4, yn3, y4, 7.0, 7.0)
            # output11, v11 = helper(xn3, x4, yn1, y2, 7.0, 3.0)
            # output12, v12 = helper(xn1, x2, yn3, y4, 3.0, 7.0)
            
            
            # candidates = tf.concat(v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12, axis=2)
            
            # idx = tf.argmin(tf.reduce_mean(tf.abs(candidates - tf.expand_dims(target_flat, axis=-1)), axis=1, keep_dims=True), axis=2)
            # idx = tf.tile(idx, [1, channels])
            
            # error_small_pred = tf.tile(tf.reduce_mean(tf.abs(output1 - target_flat), axis=1, keep_dims=True), [1, channels]) < 0.1
            
            # output = tf.where(tf.logical_or(error_small_pred, tf.logical_and(idx>=0, idx<4)), output1, 
            #                 tf.where(tf.logical_and(idx>=4, idx<8), output2, 
            #                 tf.where(tf.logical_and(idx>=8, idx<12), output3, 
            #                 tf.where(tf.logical_and(idx>=12, idx<16), output4, 
            #                 tf.where(tf.logical_and(idx>=16, idx<20), output5,
            #                 tf.where(tf.logical_and(idx>=20, idx<24), output6,
            #                 tf.where(tf.logical_and(idx>=24, idx<28), output7,
            #                 tf.where(tf.logical_and(idx>=28, idx<32), output8, 
            #                 tf.where(tf.logical_and(idx>=32, idx<36), output9, 
            #                 tf.where(tf.logical_and(idx>=36, idx<40), output10, 
            #                 tf.where(tf.logical_and(idx>=40, idx<44), output11, output12)))))))))))

            def helper(x0_, x1_, y0_, y1_, scale):
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
                wa = tf.expand_dims(((x1_f-x) * (y1_f-y) / scale), 1)
                wb = tf.expand_dims(((x1_f-x) * (y-y0_f) / scale), 1)
                wc = tf.expand_dims(((x-x0_f) * (y1_f-y) / scale), 1)
                wd = tf.expand_dims(((x-x0_f) * (y-y0_f) / scale), 1)
                output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
                return output, [tf.expand_dims(Ia, axis=-1), 
                              tf.expand_dims(Ib, axis=-1), 
                              tf.expand_dims(Ic, axis=-1), 
                              tf.expand_dims(Id, axis=-1)]
            
            output1, v1 = helper(x0, x1, y0, y1, 1.0)
            output2, v2 = helper(x0, x1, yn1, y2, 3.0)
            output3, v3 = helper(xn1, x2, y0, y1, 3.0)
            output4, v4 = helper(xn1, x2, yn1, y2, 9.0)
            output5, v5 = helper(xn2, x3, yn2, y3, 25.0)
            
            candidates = tf.concat(v1+v2+v3+v4+v5, axis=2)
            
            idx = tf.argmin(tf.reduce_mean(tf.abs(candidates - tf.expand_dims(target_flat, axis=-1)), axis=1, keep_dims=True), axis=2)
            idx = tf.tile(idx, [1, channels])
            
            error_small_pred = tf.tile(tf.reduce_mean(tf.abs(output1 - target_flat), axis=1, keep_dims=True), [1, channels]) < 0.1
            output = tf.where(tf.logical_or(error_small_pred, tf.logical_and(idx>=0, idx<4)), output1, 
                        tf.where(tf.logical_and(idx>=4, idx<8), output2, 
                        tf.where(tf.logical_and(idx>=8, idx<12), output3, 
                        tf.where(tf.logical_and(idx>=12, idx<16), output4, output5))))
            output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))
            return output

    def _interpolate(im, x, y, name='_interpolate'):
        """Perform bilinear sampling on im given x,y coordinates.

        Implements the differentiable sampling mechanism with bilinear kerenl
        in https://arxiv.org/abs/1506.02025.

        x,y are tensors specifying normalized coordinates [-1,1] to be sampled on im.
        (e.g.) (-1,-1) in x,y corresponds to pixel location (0,0) in im, and
        (1,1) in x,y corresponds to the bottom right pixel in im.
        """
        with tf.variable_scope(name):
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])

            # constants
            num_batch = tf.shape(im)[0]
            _, height, width, channels = im.get_shape().as_list()

            x = tf.to_float(x)
            y = tf.to_float(y)
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            zero = tf.constant(0, dtype=tf.int32)
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width-1/height-1]
            x = (x + 1.0) * (width_f - 1.0) / 2.0
            y = (y + 1.0) * (height_f - 1.0) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width * height

            # Create base index
            base = tf.range(num_batch) * dim1
            base = tf.reshape(base, [-1, 1])
            base = tf.tile(base, [1, height * width])
            base = tf.reshape(base, [-1])

            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.to_float(im_flat)
            pixel_a = tf.gather(im_flat, idx_a)
            pixel_b = tf.gather(im_flat, idx_b)
            pixel_c = tf.gather(im_flat, idx_c)
            pixel_d = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x1_f = tf.to_float(x1)
            y1_f = tf.to_float(y1)

            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
            wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
            wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

            output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
            output = tf.reshape(output,
                                shape=tf.stack([num_batch, height, width, channels]))
            return output

    def _spatial_transformer(img, coords, target_image):
        """Spatial transforming the values in 'img' with bilinear sampling based on
          coordinates specified in 'coords'. This is just a wrapper of '_interpolate()'
          to take absolute coordinates as input.
          """
        img_height = tf.cast(tf.shape(img)[1], tf.float32)
        img_width = tf.cast(tf.shape(img)[2], tf.float32)
        img_channels = img.get_shape().as_list()[3]
        px = tf.slice(coords, [0, 0, 0, 0], [-1, -1, -1, 1])
        py = tf.slice(coords, [0, 0, 0, 1], [-1, -1, -1, 1])
        # determine which part "fly out" of the boundary of the target image
        flyout_mask = tf.cast((px<0) | (px>img_width) | (py<0) | (py>img_height), tf.float32)
        # print("shape of flyout_mask:")
        # print(flyout_mask.get_shape().as_list())
        flyout_mask = tf.tile(flyout_mask,[1,1,1,img_channels])
        # print("shape of target image:")
        # print(target_image.get_shape().as_list())
        # scale to normalized coordinates [-1, 1] to match the input to 'interpolate'
        px = tf.clip_by_value(px/img_width*2.0 - 1.0, -1.0, 1.0)
        py = tf.clip_by_value(py/img_height*2.0 - 1.0, -1.0, 1.0)
        out_img = _interpolate(img, px, py, 'spatial_transformer')
        out_size = tf.shape(target_image)[1:3]
        # print("shape of out image:")
        # print(out_img.get_shape().as_list())
        # out_img = _interpolate_ms(img, px, py, out_size, target_image, 'spatial_transformer')

        # the flyout part in out_image should be replaced with the same part in target image
        out_img = target_image*flyout_mask + out_img*(1.0-flyout_mask)
        return out_img, flyout_mask

    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height*img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height*img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)
    if len(pose.get_shape().as_list()) == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
    src_pixel_coords = _cam2pixel(cam_coords_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords, 
                                [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0,2,3,1])
    projected_img, flyout_mask = _spatial_transformer(img, src_pixel_coords, target_image)
    
    return projected_img, flyout_mask



def warp_occ_mask(img, depth, pose, intrinsics, intrinsics_inv):
    """Inverse warp a source image to the target image plane
       Part of the code modified from  
       https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
    Args:
        img: the source image (where to sample pixels) -- [B, H, W, 3]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    import tensorflow as tf
    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones  = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    def _interpolate(im, x, y, name='_interpolate'):
        """Perform bilinear sampling on im given x,y coordinates.

        Implements the differentiable sampling mechanism with bilinear kerenl
        in https://arxiv.org/abs/1506.02025.

        x,y are tensors specifying normalized coordinates [-1,1] to be sampled on im.
        (e.g.) (-1,-1) in x,y corresponds to pixel location (0,0) in im, and
        (1,1) in x,y corresponds to the bottom right pixel in im.
        """
        with tf.variable_scope(name):
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])

            # constants
            num_batch = tf.shape(im)[0]
            batch_size, height, width, channels = im.get_shape().as_list()

            x = tf.to_float(x)
            y = tf.to_float(y)
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            zero = tf.constant(0, dtype=tf.int32)
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width-1/height-1]
            x = (x + 1.0) * (width_f - 1.0) / 2.0
            y = (y + 1.0) * (height_f - 1.0) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0_c = tf.clip_by_value(x0, zero, max_x)
            x1_c = tf.clip_by_value(x1, zero, max_x)
            y0_c = tf.clip_by_value(y0, zero, max_y)
            y1_c = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width * height

            # Create base index
            base = tf.range(num_batch) * dim1
            base = tf.reshape(base, [-1, 1])
            base = tf.tile(base, [1, height * width])
            base = tf.reshape(base, [-1])

            base_y0 = base + y0_c * dim2
            base_y1 = base + y1_c * dim2
            idx_a = base_y0 + x0_c
            idx_b = base_y1 + x0_c
            idx_c = base_y0 + x1_c
            idx_d = base_y1 + x1_c

            # use indices to lookup pixels in the flat image and restore channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.to_float(im_flat)
            pixel_a = tf.gather(im_flat, idx_a)
            pixel_b = tf.gather(im_flat, idx_b)
            pixel_c = tf.gather(im_flat, idx_c)
            pixel_d = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x1_f = tf.to_float(x1_c)
            y1_f = tf.to_float(y1_c)

            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
            wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
            wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)
            zerof = tf.zeros_like(wa)
            zeros = tf.zeros(shape=[int(batch_size*height*width), int(channels)], dtype='float32')
            wa = tf.where(tf.logical_and(tf.equal(x1_c, x1), tf.equal(y1_c, y1)), wa, zerof)
            wb = tf.where(tf.logical_and(tf.equal(x1_c, x1), tf.equal(y0_c, y0)), wb, zerof)
            wc = tf.where(tf.logical_and(tf.equal(x0_c, x0), tf.equal(y1_c, y1)), wc, zerof)
            wd = tf.where(tf.logical_and(tf.equal(x0_c, x0), tf.equal(y0_c, y0)), wd, zerof)
            output = tf.Variable(zeros, 
                                 trainable=False,
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])
            init = tf.assign(output, zeros)
            with tf.control_dependencies([init]):
                output = tf.scatter_add(output, idx_a, im_flat*wa)
                output = tf.scatter_add(output, idx_b, im_flat*wb)
                output = tf.scatter_add(output, idx_c, im_flat*wc)
                output = tf.scatter_add(output, idx_d, im_flat*wd)

            output = tf.reshape(output,
                                shape=tf.stack([num_batch, height, width, channels]))
            return output

    def _spatial_transformer(img, coords):
        """Spatial transforming the values in 'img' with bilinear sampling based on
          coordinates specified in 'coords'. This is just a wrapper of '_interpolate()'
          to take absolute coordinates as input.
          """
        img_height = tf.cast(tf.shape(img)[1], tf.float32)
        img_width = tf.cast(tf.shape(img)[2], tf.float32)
        px = tf.slice(coords, [0, 0, 0, 0], [-1, -1, -1, 1])
        py = tf.slice(coords, [0, 0, 0, 1], [-1, -1, -1, 1])
        # scale to normalized coordinates [-1, 1] to match the input to 'interpolate'
        px = tf.clip_by_value(px/img_width*2.0 - 1.0, -1.0, 1.0)
        py = tf.clip_by_value(py/img_height*2.0 - 1.0, -1.0, 1.0)
        out_img = _interpolate(img, px, py, 'spatial_transformer')
        # out_size = tf.shape(target_image)[1:3]
        # out_img = _interpolate_ms(img, px, py, out_size, target_image, 'spatial_transformer')
        return out_img

    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height*img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height*img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)
    pose_mat = _pose_vec2mat(pose)
    # pose_mat = pose

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
    src_pixel_coords = _cam2pixel(cam_coords_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords, 
                                [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0,2,3,1])
    projected_img = _spatial_transformer(img, src_pixel_coords)
    
    return projected_img