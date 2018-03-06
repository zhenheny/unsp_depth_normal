from __future__ import division
import os
import sys
sys.path.insert(0, "./depth2normal/")
sys.path.append("./eval")

import time
import math
import nets
import pickle
import random
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as uts

import depth2normal_tf as d2n
import normal2depth_tf as n2d
import evaluate_kitti as kitti_eval
import evaluate_normal as normal_eval

import pdb

def is_exists(fields, name):
    for field in fields:
        if field in name:
            return True

    return False


class SfMLearner(object):
    def __init__(self):
        pass

    def gradient(self, pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    def gradient_weight(self, pred, alpha=0.5):
        gradx, grady = self.gradient(pred)
        grad = tf.sqrt(tf.square(gradx[:,1:,:,:])+tf.square(grady[:,:,1:,:]))
        grad = tf.reduce_mean(grad, axis=3, keep_dims=True)
        weight = tf.exp(-1*alpha*grad)
        padding = tf.constant([[0,0],[1,0],[1,0],[0,0]])
        weight = tf.pad(weight, padding, 'CONSTANT')
        return weight

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


    def gradient_loss(self, proj_image, tgt_image_grad):
        proj_image_grad_x, proj_image_grad_y = self.gradient(
                                    proj_image)
        tgt_image_grad_x, tgt_image_grad_y = tgt_image_grad[0], tgt_image_grad[1]
        proj_error_grad_x = tf.abs(tgt_image_grad_x - proj_image_grad_x)
        proj_error_grad_y = tf.abs(tgt_image_grad_y - proj_image_grad_y)
        # img_grad_loss += opt.img_grad_weight * tf.reduce_mean(curr_proj_error_grad_x * \
        #     tf.slice(tf.expand_dims(curr_exp[:,:,:,1], -1), slice_starts[j], slice_size))
        # img_grad_loss += opt.img_grad_weight * tf.reduce_mean(curr_proj_error_grad_y * \
        #     tf.slice(tf.expand_dims(curr_exp[:,:,:,1], -1), slice_starts[j], slice_size))
        img_grad_loss = tf.reduce_mean(proj_error_grad_x) \
                        + tf.reduce_mean(proj_error_grad_y)
        return img_grad_loss


    def compute_loss(self, s,
                     tgt_image,
                     src_image,
                     proj_cam2pix,
                     proj_pix2cam,
                     pred_depth_tgt,
                     pred_poses,
                     pred_edges=None,
                     dense_motion_maps=None,
                     pred_exp_logits=None,
                     depth_inverse=False):

        pixel_loss = 0
        exp_loss = 0
        smooth_loss = 0
        normal_smooth_loss = 0
        img_grad_loss = 0
        edge_loss = 0
        dm_loss = 0
        tgt_image_all = []
        src_image_all = []
        proj_image_stack_all = []
        proj_error_stack_all = []
        shifted_proj_image_stack_all = []
        shifted_proj_error_stack_all = []
        tgt_image_grad_weight_all = []
        exp_mask_stack_all = []
        pred_normals = []
        pred_disps2 = []
        flyout_map_all = []

        opt = self.opt

        # Construct a reference explainability mask (i.e. all
        # pixels are explainable)
        if opt.explain_reg_weight > 0:
            ref_exp_mask = self.get_reference_explain_mask(s)

        curr_tgt_image = tf.image.resize_bilinear(tgt_image,
                 [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
        curr_src_image = tf.image.resize_bilinear(src_image,
             [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

         ## depth2normal and normal2depth at each scale level
        intrinsic_mtx = proj_cam2pix
        intrinsics = tf.concat([
                        tf.expand_dims(intrinsic_mtx[:,0,0],1),
                        tf.expand_dims(intrinsic_mtx[:,1,1],1),
                        tf.expand_dims(intrinsic_mtx[:,0,2],1),
                        tf.expand_dims(intrinsic_mtx[:,1,2],1)], 1)

        # pdb.set_trace()
        pred_depth_tensor = tf.squeeze(pred_depth_tgt, axis=3)
        pred_normal = d2n.depth2normal_layer_batch(
                           pred_depth_tensor, intrinsics, depth_inverse)
        pred_depth2 = n2d.normal2depth_layer_batch(
                           pred_depth_tensor, pred_normal,
                           intrinsics, curr_tgt_image)
        pred_depth2 = tf.expand_dims(pred_depth2, -1)

        # normalize the depth with mean depth
        pred_depth2 = pred_depth2 /tf.reduce_mean(pred_depth2)

        # normal depth2 to avoid corner case of preddepth2=0
        pred_disp2 = 1.0 / pred_depth2

        pred_normals.append(pred_normal)
        pred_disps2.append(pred_disp2)

         ## 1. L2 loss as edge_loss; 2. cross_entropy loss as edge_loss; 3. L1 loss as edge_loss
         ## ref_edge_mask is all 0
        if opt.edge_mask_weight > 0:
             ## 1. L2 loss
             ref_edge_mask = self.get_reference_explain_mask(s)[:,:,:,0]
             edge_loss += opt.edge_mask_weight/(2**s) *\
                    tf.reduce_mean(tf.square(tf.squeeze(pred_edges, axis=3)-ref_edge_mask))

         ## compute smoothness loss for depth considering the predicted edges
        if opt.smooth_weight > 0:
            if opt.edge_mask_weight > 0:
                smooth_loss += tf.multiply(opt.smooth_weight/(2**s), \
                    self.compute_smooth_loss_wedge(pred_disp2, pred_edges, mode='l2'))
                    # self.compute_smooth_loss_wedge_3d(pred_disp2, pred_edges[s], intrinsics))
            else:
                smooth_loss += tf.multiply(opt.smooth_weight/(2**s), \
                    self.compute_smooth_loss(pred_disp2))

        ## compute smoothness loss for normal considering the predicted edges
        # self.compute_edge_aware_smooth_loss(pred_normal[:,3:-3,3:-3,:], ))
        if opt.normal_smooth_weight > 0:
            normal_smooth_loss += tf.multiply(opt.normal_smooth_weight/(2**s), \
                    self.compute_smooth_loss_wedge(pred_normal[:, 3:-3, 3:-3, :],
                        pred_edges[:,3:-3,3:-3,], mode='l2', alpha=0.1))
                    # self.compute_smooth_loss(pred_normal[:, 3:-3, 3:-3, :]))

        ## compute photometric loss the predicted edges
        curr_tgt_image_grad = self.gradient(curr_tgt_image)
        curr_tgt_image_grad_weight = self.gradient_weight(curr_tgt_image)

        for i in range(opt.num_source):
            # Inverse warp the source image to the target image frame
            # Use pred_depth and 8 pred_depth2 maps for inverse warping
            curr_proj_image, shifted_curr_proj_image, curr_flyout_map= uts.inverse_warp(
                        curr_src_image[:,:,:,3*i:3*(i+1)],
                        pred_depth2,
                        pred_poses[:,i,:], # [batchsize, num_source, 6]
                        proj_cam2pix,  ## [batchsize, scale, 3, 3]
                        proj_pix2cam,
                        curr_tgt_image,
                        dense_motion_maps[:,:,:,3*i:3*(i+1)])

            curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
            shifted_curr_proj_error = tf.abs(shifted_curr_proj_image - curr_tgt_image)

            if opt.gradient_filter_pixel_loss > 0:
                curr_proj_error *= curr_tgt_image_grad_weight
                shifted_curr_proj_error *= curr_tgt_image_grad_weight

            # Photo-consistency loss weighted by explainability
            if opt.explain_reg_weight > 0:
                curr_exp_logits = tf.slice(pred_exp_logits,
                                           [0, 0, 0, i*2],
                                           [-1, -1, -1, 2])
                curr_exp = tf.nn.softmax(curr_exp_logits)
                pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                pixel_loss += tf.reduce_mean(shifted_curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))

            elif opt.edge_as_explain > 0:
                pixel_loss += tf.reduce_mean(curr_proj_error * \
                            (1.0 - pred_edges))
                pixel_loss += tf.reduce_mean(shifted_curr_proj_error * \
                            (1.0 - pred_edges))

            else:
                pixel_loss += tf.reduce_mean(curr_proj_error)
                pixel_loss += tf.reduce_mean(shifted_curr_proj_error)

            # SSIM loss
            if opt.ssim_weight > 0:
                pixel_loss += opt.ssim_weight * tf.reduce_mean(
                                self.SSIM(curr_proj_image, curr_tgt_image))
                pixel_loss += opt.ssim_weight * tf.reduce_mean(
                                self.SSIM(shifted_curr_proj_image, curr_tgt_image))

            if opt.img_grad_weight > 0:
                img_grad_loss += self.gradient_loss(curr_proj_image, curr_tgt_image_grad)
                img_grad_loss += self.gradient_loss(shifted_curr_proj_image,
                        curr_tgt_image_grad)
                img_grad_loss *= opt.img_grad_weight

            # Cross-entropy loss as regularization for the explainability prediction
            if opt.explain_reg_weight > 0:
                exp_loss += opt.explain_reg_weight * \
                    self.compute_exp_reg_loss(curr_exp_logits,
                                              ref_exp_mask)

            # Dense_motion loss, disencourage dense motion
            if opt.dense_motion_weight > 0:
                ref_dm_map = self.get_reference_explain_mask(s)[:,:,:,0]
                ref_dm_map = tf.tile(ref_dm_map[:,:,:,None], [1,1,1,3])

                dm_loss += opt.dense_motion_weight / (2 ** s) * \
                                self.compute_smooth_loss_wedge(
                                dense_motion_maps[:,:,:,3*i:3*(i+1)],
                                pred_edges, mode='l2', alpha=1)

                dm_loss += opt.dense_motion_weight * 0.01 /(2**s) * \
                                tf.reduce_mean(tf.abs(
                                    dense_motion_maps[:,:,:,3*i:3*(i+1)] \
                                    - ref_dm_map))

            # Prepare images for tensorboard summaries
            if i == 0:
                proj_image_stack = curr_proj_image
                proj_error_stack = curr_proj_error
                tgt_image_grad_weight = curr_tgt_image_grad_weight
                shifted_proj_image_stack = shifted_curr_proj_image
                shifted_proj_error_stack = shifted_curr_proj_error
                flyout_map = curr_flyout_map

                if opt.explain_reg_weight > 0:
                    exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)

            else:
                proj_image_stack = tf.concat([proj_image_stack,
                                                  curr_proj_image], axis=3)
                shifted_proj_image_stack = tf.concat([shifted_proj_image_stack,
                        curr_proj_image], axis=3)
                proj_error_stack = tf.concat([proj_error_stack,
                                              curr_proj_error], axis=3)
                tgt_image_grad_weight = tf.concat([tgt_image_grad_weight,
                                            curr_tgt_image_grad_weight], axis=3)
                shifted_proj_error_stack = tf.concat([shifted_proj_error_stack,
                        shifted_curr_proj_error], axis=3)
                flyout_map = tf.concat([flyout_map, curr_flyout_map], axis=3)

                if opt.explain_reg_weight > 0:
                    exp_mask_stack = tf.concat([exp_mask_stack,
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)

        # pixel_loss /= len(pred_depths2)
        tgt_image_all.append(curr_tgt_image)
        src_image_all.append(curr_src_image)
        proj_image_stack_all.append(proj_image_stack)
        shifted_proj_image_stack_all.append(shifted_proj_image_stack)
        proj_error_stack_all.append(proj_error_stack)
        tgt_image_grad_weight_all.append(tgt_image_grad_weight)
        shifted_proj_error_stack_all.append(shifted_proj_error_stack)
        flyout_map_all.append(flyout_map)
        if opt.explain_reg_weight > 0:
            exp_mask_stack_all.append(exp_mask_stack)

        return [pixel_loss, \
                exp_loss, \
                smooth_loss, \
                normal_smooth_loss, \
                img_grad_loss, \
                edge_loss, dm_loss], \
               [tgt_image_all, \
                src_image_all, \
                tgt_image_grad_weight_all, \
                proj_image_stack_all, \
                proj_error_stack_all, \
                shifted_proj_image_stack_all, \
                shifted_proj_error_stack_all, \
                exp_mask_stack_all, \
                pred_normals, \
                pred_disps2, \
                flyout_map_all]


    def build_multi_train_graph(self):

        opt = self.opt
        optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)

        with tf.name_scope("data_loading"):
            seed = random.randint(0, 2**31 - 1)

            file_list = self.format_file_list(opt.dataset_dir, 'train')
            image_paths_queue = tf.train.string_input_producer(
                file_list['image_file_list'],
                seed=seed,
                shuffle=False)

            cam_paths_queue = tf.train.string_input_producer(
                file_list['cam_file_list'],
                seed=seed,
                shuffle=False)

            # Load images
            img_reader = tf.WholeFileReader()
            _, image_contents = img_reader.read(image_paths_queue)
            image_seq = tf.image.decode_jpeg(image_contents)
            image_seq = self.preprocess_image(image_seq)
            tgt_image, src_image_seq = \
                self.unpack_image_sequence_list(image_seq)

            # Load camera intrinsics
            cam_reader = tf.TextLineReader()
            _, raw_cam_contents = cam_reader.read(cam_paths_queue)
            rec_def = []
            for i in range(9):
                rec_def.append([1.])

            raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                        record_defaults=rec_def)
            raw_cam_vec = tf.stack(raw_cam_vec)
            raw_cam_mat = tf.reshape(raw_cam_vec, [3, 3])
            proj_cam2pix, proj_pix2cam = self.get_multi_scale_intrinsics(
                raw_cam_mat, opt.num_scales)

            # Form training batches
            input_batch = tf.train.batch(src_image_seq + \
                                   [tgt_image, proj_cam2pix, proj_pix2cam],
                                   batch_size=opt.batch_size,
                                   num_threads=4,
                                   capacity=64)

        num_gpu = len(self.opt.gpu_id)

        # split all the input images here to multiple gpu
        input_splits = []
        grads = [] # gradient from each tower
        for input_item in input_batch:
            input_splits.append(tf.split(input_item, num_gpu, axis=0))

        src_image_seq = [[input_splits[i][j] for i in range(opt.num_source)] for j in range(num_gpu)]
        tgt_image, proj_cam2pix_div, proj_pix2cam_div = input_splits[opt.num_source:]

        self.opt.batch_size_per_tower = int(opt.batch_size / num_gpu)
        print ("tgt_image batch images shape:")
        print (tgt_image[0].get_shape().as_list())

        print ("src_image shape")
        print (src_image_seq[0][0].get_shape().as_list())

        for gpu_id in range(num_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                op_scope_name = 'model' if gpu_id == 0 else 'tower_%d' %  gpu_id
                proj_cam2pix = proj_cam2pix_div[gpu_id]
                proj_pix2cam = proj_pix2cam_div[gpu_id]

                with tf.name_scope(op_scope_name):
                    ## depth prediction for both tgt and src image
                    image_stack = tf.concat([tgt_image[gpu_id]] + src_image_seq[gpu_id],\
                            axis=0)

                    with tf.name_scope("depth_prediction"):
                        pred_disp, pred_edges, depth_net_endpoints = nets.disp_net(
                                image_stack, do_edge=(opt.edge_mask_weight > 0),
                                reuse=False if gpu_id == 0 else True)
                        pred_disp = [d[:,:,:,:1] for d in pred_disp]
                        pred_depth = [1./d for d in pred_disp]

                    pred_depth_src = None
                    pred_depth_tgt = None

                    if self.opt.depth4pose:
                        pred_depth_src = []
                        pred_depth_tgt = []
                        for d in pred_depth:
                            pred_depth_tgt.append(tf.slice(d, [0, 0, 0, 0],
                                [self.opt.batch_size_per_tower, -1, -1, -1]))
                            depth_src = []
                            for i in range(opt.num_source):
                                depth_src.append(tf.slice(d,
                                    [(i + 1) * self.opt.batch_size_per_tower, 0, 0, 0],
                                    [self.opt.batch_size_per_tower, -1, -1, -1]))
                            pred_depth_src.append(depth_src)

                    for i in range(len(pred_edges)):
                        pred_edges[i] = tf.slice(pred_edges[i], [0, 0, 0, 0],
                                [self.opt.batch_size_per_tower, -1, -1, -1])

                with tf.name_scope("pose_prediction"):
                    pred_poses, pred_exp_logits, _, _= \
                        nets.pose_exp_net(tgt_image[gpu_id],
                               src_image_seq[gpu_id],
                               tgt_depth=pred_depth_tgt[0] if self.opt.depth4pose else None,
                               src_depth_seq=pred_depth_src[0] if self.opt.depth4pose else None,
                               do_exp=False,
                               do_dm=False,
                               reuse=False if gpu_id == 0 else True)

                with tf.name_scope("view_synthesis"):
                    pred_depth2 = self.depth_with_normal(
                            pred_depth_tgt[0],
                            proj_cam2pix[:, 0, :, :],
                            tgt_image[gpu_id])

                    proj_image_seq = []
                    proj_depth_seq = []

                    for i in range(opt.num_source):
                        proj_image, flyout_map= uts.inverse_warp(
                                    src_image_seq[gpu_id][i],
                                    pred_depth2,
                                    pred_poses[:,i,:], # [batchsize, num_source, 6]
                                    proj_cam2pix[:,0,:,:],  ## [batchsize, scale, 3, 3]
                                    proj_pix2cam[:,0,:,:],
                                    tgt_image[gpu_id])
                        proj_image_seq.append(proj_image)

                        proj_depth = uts.inverse_depth(
                                    pred_depth2,
                                    pred_depth_src[0][i],
                                    pred_poses[:,i,:], # [batchsize, num_source, 6]
                                    proj_cam2pix[:,0,:,:],  ## [batchsize, scale, 3, 3]
                                    proj_pix2cam[:,0,:,:])
                        proj_depth_seq.append(proj_depth)


                # using dense motion network for dense motion
                with tf.name_scope("dense_motion"):
                    if opt.motion_net == 'unet':
                        dense_motion_maps, _ = \
                            nets.dense_motion_u_net(tgt_image[gpu_id],
                                                    proj_image_seq,
                                                    pred_depth2,
                                                    proj_depth_seq,
                                              reuse=False if gpu_id == 0 else True)
                    elif opt.motion_net == 'pwc':
                        dense_motion_maps = \
                            nets.dense_motion_pwc_net(tgt_image[gpu_id],
                                                    proj_image_seq,
                                                    pred_depth2,
                                                    proj_depth_seq,
                                              reuse=False if gpu_id == 0 else True)
                    else:
                        raise ValueError('No such network {}'.format(opt.motion_net))


                with tf.name_scope("compute_loss"):
                    pixel_loss = 0
                    exp_loss = 0
                    smooth_loss = 0
                    normal_smooth_loss = 0
                    img_grad_loss = 0
                    edge_loss = 0
                    dm_loss = 0

                    tgt_image_all = []
                    src_image_all = []
                    proj_image_stack_all = []
                    proj_error_stack_all = []
                    shifted_proj_image_stack_all = []
                    shifted_proj_error_stack_all = []
                    tgt_image_grad_weight_all = []
                    exp_mask_stack_all = []
                    pred_normals = []
                    pred_disps2 = []
                    flyout_map_all = []

                    src_image = tf.concat(src_image_seq[gpu_id], axis=3)

                    for s in range(opt.num_scales):
                        loss, outputs = self.compute_loss(s,
                                tgt_image[gpu_id],
                                src_image,
                                proj_cam2pix[:, s, :, :],
                                proj_pix2cam[:, s, :, :],
                                pred_depth_tgt[s],
                                pred_poses,
                                pred_edges[s],
                                dense_motion_maps[s],
                                pred_exp_logits[s],
                                depth_inverse=False)

                        pixel_loss += loss[0]
                        exp_loss += loss[1]
                        smooth_loss += loss[2]
                        normal_smooth_loss += loss[3]
                        img_grad_loss += loss[4]
                        edge_loss += loss[5]
                        dm_loss += loss[6]

                        tgt_image_all += outputs[0]
                        src_image_all += outputs[1]
                        tgt_image_grad_weight_all += outputs[2]
                        proj_image_stack_all += outputs[3]
                        proj_error_stack_all += outputs[4]
                        shifted_proj_image_stack_all += outputs[5]
                        shifted_proj_error_stack_all += outputs[6]
                        exp_mask_stack_all += outputs[7]
                        pred_normals += outputs[8]
                        pred_disps2 += outputs[9]
                        flyout_map_all += outputs[10]

                    total_loss = pixel_loss + \
                                 smooth_loss + \
                                 exp_loss + \
                                 normal_smooth_loss + \
                                 img_grad_loss + \
                                 edge_loss + \
                                 dm_loss

                if gpu_id == 0:
                    train_vars = [var for var in tf.trainable_variables() \
                            if is_exists(opt.trainable_var_scope, var.name) ]

                grads.append(optim.compute_gradients(total_loss, var_list=train_vars))

                # for grad, var in zip(grads[gpu_id], train_vars):
                #     print(grad[0])
                #     print(var)

        if num_gpu > 1:
            grads_and_vars = self.average_gradients(grads)
        else:
            grads_and_vars = grads[0]

        with tf.name_scope("train_op"):
            self.train_op = optim.apply_gradients(grads_and_vars)
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth_tgt = pred_depth_tgt
        self.pred_disp = pred_disp
        self.pred_normals = pred_normals
        self.pred_disps2 = pred_disps2
        self.pred_poses = pred_poses
        self.opt.steps_per_epoch = \
            int(len(file_list['image_file_list'])//opt.batch_size)
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.edge_loss = edge_loss
        self.dm_loss = dm_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_all = src_image_all
        self.proj_image_stack_all = proj_image_stack_all
        self.tgt_image_grad_weight_all = tgt_image_grad_weight_all
        self.shifted_proj_image_stack_all = shifted_proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.shifted_proj_error_stack_all = shifted_proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all
        self.flyout_map_all = flyout_map_all
        self.pred_edges = pred_edges
        self.dense_motion_maps = dense_motion_maps


    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size_per_tower,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def compute_smooth_loss_multiscale(self, pred_disp):
        def gradient(pred, scale):
            D_dy = pred[:, scale:, :, :] - pred[:, :(-1 * scale), :, :]
            D_dx = pred[:, :, scale:, :] - pred[:, :, :(-1 * scale), :]
            return D_dx, D_dy
        scales = [2]
        loss = 0
        for scale in scales:
            dx, dy = gradient(pred_disp, scale)
            dx2, dxdy = gradient(dx, scale)
            dydx, dy2 = gradient(dy, scale)
            loss += tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

        return loss / len(scales)

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_gradient_loss(self, disp, image):
        ## compute disp and image gradient
        ## loss is designed to minimize the gradient difference

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        disp_gradients_x, disp_gradients_y = gradient(disp)
        image_gradients_x, image_gradients_y = gradient(image)
        diff_x = disp_gradients_x - image_gradients_x
        diff_y = disp_gradients_y - image_gradients_y
        gradient_loss = tf.reduce_mean(tf.abs(diff_x)) + \
                        tf.reduce_mean(tf.abs(diff_y))

        return gradient_loss


    def compute_edge_aware_smooth_loss(self, disp, image):
        ## compute edge aware smoothness loss
        ## image should be a rank 4 tensor

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        alpha = 10
        disp_gradients_x, disp_gradients_y = gradient(disp)
        dx2, dxdy = gradient(disp_gradients_x)
        dydx, dy2 = gradient(disp_gradients_y)
        image_gradients_x, image_gradients_y = gradient(image)

        weights_x = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        # smoothness_x = disp_gradients_x * weights_x
        # smoothness_y = disp_gradients_y * weights_y
        # smoothness_dxdy = dxdy * weights_x[:,1:,:,:]
        # smoothness_dydx = dydx * weights_y[:,:,1:,:]

        smoothness_dx2 = dx2 * weights_x[:,:,1:,:]
        smoothness_dy2 = dy2 * weights_y[:,1:,:,:]

        # smoothness_loss = tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

        smoothness_loss_2nd = tf.reduce_mean(tf.abs(smoothness_dx2)) + \
                              tf.reduce_mean(tf.abs(smoothness_dy2))
        return smoothness_loss_2nd


    def compute_smooth_loss_wedge(self, disp, edge, mode='l1', alpha=10.0):
        ## in edge, 1 represents edge, disp and edge are rank 3 vars

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        disp_grad_x, disp_grad_y = gradient(disp)
        dx2, dxdy = gradient(disp_grad_x)
        dydx, dy2 = gradient(disp_grad_y)

        # edge_grad_x, edge_grad_y = gradient(edge)
        weight_x = tf.exp(-1*alpha*tf.abs(edge))
        weight_y = tf.exp(-1*alpha*tf.abs(edge))

        if mode == "l2":
            # smoothness_loss = tf.reduce_mean(tf.abs(dx2 * weight_x[:,:,1:-1,:])) + \
            #               tf.reduce_mean(tf.abs(dy2 * weight_y[:,1:-1,:,:]))
            smoothness_loss = tf.reduce_mean(tf.clip_by_value(dx2 * weight_x[:,:,1:-1,:], 0.0, 10.0)) + \
                          tf.reduce_mean(tf.clip_by_value(dy2 * weight_y[:,1:-1,:,:], 0.0, 10.0)) #+ \
                          # tf.reduce_mean(tf.clip_by_value(dxdy * weight_x[:,1:,1:,:], 0.0, 10.0)) + \
                          # tf.reduce_mean(tf.clip_by_value(dydx * weight_y[:,1:,1:,:], 0.0, 10.0))
        if mode == "l1":
            smoothness_loss = tf.reduce_mean(tf.abs(disp_grad_x * weight_x[:,:,1:,:])) + \
                          tf.reduce_mean(tf.abs(disp_grad_y * weight_y[:,1:,:,:]))

        return smoothness_loss


    def compute_smooth_loss_wedge_3d(self, disp, edge, intrinsics, alpha=10.0):

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        batch, height, width = disp.get_shape().as_list()[:3]
        fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
        fx = tf.tile(fx[:,None,None,None], [1, height, width-1, 1])
        fy = tf.tile(fy[:,None,None,None], [1, height-1, width, 1])
        cx = tf.tile(cx[:,None,None,None], [1, height, width-1, 1])
        cy = tf.tile(cy[:,None,None,None], [1, height-1, width, 1])

        disp_grad_y, disp_grad_x = disp[:,1:,:,:]-disp[:,:-1,:,:], disp[:,:,1:,:]-disp[:,:,:-1,:]

        x, y = tf.meshgrid(np.arange(width), np.arange(height))
        x = tf.cast(tf.tile(x[None,:,:,None],[batch,1,1,1]),tf.float32)
        y = tf.cast(tf.tile(y[None,:,:,None],[batch,1,1,1]), tf.float32)
        x1, y1 = x[:,:,:-1,:], y[:,:-1,:,:]
        x2, y2 = x[:,:,1:,:], y[:,1:,:,:]

        disp_grad_y_3d = disp_grad_y/(tf.sigmoid(((y2-cy)*disp[:,1:,:,:]-(y1-cy)*disp[:,:-1,:,:])/fy-1.0)*2.0+1.0)
        disp_grad_x_3d = disp_grad_x/(tf.sigmoid(((x2-cx)*disp[:,:,1:,:]-(x1-cx)*disp[:,:,:-1,:])/fx-1.0)*2.0+1.0)
        dx2, dxdy = gradient(disp_grad_x_3d)
        dydx, dy2 = gradient(disp_grad_y_3d)

        weight_x = tf.exp(-1*alpha*tf.abs(edge))
        weight_y = tf.exp(-1*alpha*tf.abs(edge))
        smoothness_loss = tf.reduce_mean(tf.clip_by_value(dx2 * weight_x[:,:,1:-1,:], 0.0, 10.0)) + \
                          tf.reduce_mean(tf.clip_by_value(dy2 * weight_y[:,1:-1,:,:], 0.0, 10.0))

        return smoothness_loss


    def compute_smooth_loss_wedge_noexp(self, disp, edge):
        ## in edge, 1 represents edge, disp and edge are rank 3 vars

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        disp_grad_x, disp_grad_y = gradient(disp)
        dx2, dxdy = gradient(disp_grad_x)
        dydx, dy2 = gradient(disp_grad_y)
        # edge_grad_x, edge_grad_y = gradient(edge)
        weight_x = 1-edge
        weight_y = 1-edge

        smoothness_loss = tf.reduce_mean(tf.abs(dx2 * weight_x[:,:,1:-1,:])) + \
                          tf.reduce_mean(tf.abs(dy2 * weight_y[:,1:-1,:,:]))

        return smoothness_loss

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        if opt.edge_mask_weight > 0:
            tf.summary.scalar("edge_loss", self.edge_loss)
        if opt.dense_motion_weight > 0:
            tf.summary.scalar("dm_loss", self.dm_loss)

        tf.summary.image("pred_normal", (self.pred_normals[0]+1.0)/2.0)
        tf.summary.image("pred_disp2", self.pred_disps2[0])
        # for s in range(opt.num_scales):
        s = 0
        tf.summary.histogram("scale%d_depth" % s, self.pred_depth_tgt[s])
        tf.summary.image('scale%d_depth_image' % s, self.pred_depth_tgt[s])
        tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth_tgt[s])
        tf.summary.image('scale%d_target_image' % s, \
                         self.deprocess_image(self.tgt_image_all[s]))
        tf.summary.image('scale%d_edge_map' % s, self.pred_edges[s])

        for i in range(opt.num_source):
            if opt.explain_reg_weight > 0:
                tf.summary.image(
                    'scale%d_exp_mask_%d' % (s, i),
                    tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
            # tf.summary.image(
            #     'scale%d_source_image_%d' % (s, i),
            #     self.deprocess_image(self.src_image_all[s][:, :, :, i*3:(i+1)*3]))

            tf.summary.image('scale%d_projected_image_%d' % (s, i),
                self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
            tf.summary.image('scale%d_proj_error_%d' % (s, i),
                tf.expand_dims(self.proj_error_stack_all[s][:,:,:,i], -1))
            tf.summary.image('scale%d_tgt_image_grad_wt_%d' % (s, i),
                tf.expand_dims(self.tgt_image_grad_weight_all[s][:,:,:,i], -1))
            tf.summary.image('scale%d_shifted_projected_image_%d' % (s, i),
                    self.deprocess_image(self.shifted_proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
            tf.summary.image('scale%d_shifted_proj_error_%d' % (s, i),
                    tf.expand_dims(self.shifted_proj_error_stack_all[s][:,:,:,i], -1))

            tf.summary.image('scale%d_flyout_mask_%d' % (s,i), self.flyout_map_all[s][:,:,:,i*3:(i+1)*3])
            tf.summary.image('scale%d_dense_motion_%d' % (s,i), self.dense_motion_maps[s][:,:,:,i*3:(i+1)*3])

            # tf.summary.image('scale%d_src_error_%d' % (s, i),
            #     self.deprocess_image(tf.abs(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3] - self.src_image_all[s][:, :, :, i*3:(i+1)*3])))
            tf.summary.histogram("tx", self.pred_poses[:,:,0])
            tf.summary.histogram("ty", self.pred_poses[:,:,1])
            tf.summary.histogram("tz", self.pred_poses[:,:,2])
            tf.summary.histogram("rx", self.pred_poses[:,:,3])
            tf.summary.histogram("ry", self.pred_poses[:,:,4])
            tf.summary.histogram("rz", self.pred_poses[:,:,5])

        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        assert (opt.batch_size % len(opt.gpu_id) == 0)

        opt.num_scales = 4
        self.opt = opt

        if not ((opt.continue_train==True) and (opt.checkpoint_continue=="")):
            with open("./eval/"+opt.eval_txt, "w") as f:
                f.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(
                    'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))

        with tf.variable_scope("training"):
            self.build_multi_train_graph()

        with tf.variable_scope("training", reuse=True):
            self.setup_inference(opt.img_height, opt.img_width, "depth")

        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])

        load_saver_vars = [var for var in tf.model_variables() \
                if not is_exists(opt.rm_var_scope, var.name)]
        self.load_saver = tf.train.Saver(load_saver_vars + [self.global_step], max_to_keep=40)
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                    max_to_keep=40)

        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=opt.gpu_fraction)

        with sv.managed_session(config=tf.ConfigProto(
                allow_soft_placement=True, \
                log_device_placement=False,\
                gpu_options=gpu_options)) as sess:

            print('Trainable variables: ')
            for var in tf.trainable_variables():
                if is_exists(opt.trainable_var_scope, var.name):
                    print(var.name)

            print("parameter_count =", sess.run(parameter_count))

            if opt.continue_train:
                print("Resume training from previous checkpoint")
                if opt.checkpoint_continue == "":
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                    checkpoint = opt.checkpoint_dir + "/model.latest"
                    self.saver.restore(sess, checkpoint)
                else:
                    checkpoint = opt.checkpoint_continue
                    self.load_saver.restore(sess, checkpoint)

            for step in range(0, opt.max_steps):
                start_time = time.time()
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    if self.smooth_loss != 0:
                        fetches["smooth_loss"] = self.smooth_loss
                    if self.edge_loss != 0:
                        fetches["edge_loss"] = self.edge_loss

                    fetches['pred_depth_tgt'] = self.pred_depth_tgt
                    fetches['pred_disp'] = self.pred_disp
                    fetches['pred_normal'] = self.pred_normals[0]
                    # fetches['pred_depth2'] = self.pred_depth2
                    fetches['pred_disp2'] = self.pred_disps2[0]
                    fetches['pred_poses'] = self.pred_poses
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / opt.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * opt.steps_per_epoch
                    # print(results['pred_disp2'].max())
                    # print(results['pred_disp2'].min())
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, opt.steps_per_epoch, \
                                time.time() - start_time, results["loss"]))
                    if "smooth_loss" in results:
                        print(results['edge_loss'])
                    if np.any(np.isnan(results['pred_depth_tgt'][-1])):
                        # np.save("./depth.npy", results['pred_depth'][-1])
                        print (results['pred_depth_tgt'][-1])
                        print ("-----------")
                        print (results["pred_normal"])
                        break

                if step % opt.eval_freq == 0:
                    with tf.name_scope("evaluation"):
                        dataset_name = opt.dataset_dir.split("/")[-2]

                        ## evaluation for kitti dataset
                        if dataset_name in ["kitti",'cityscapes']:
                            modes = ["kitti"]
                            root_img_path = opt.eval_data_path
                            normal_gt_path = opt.eval_data_path + "/kitti_normal_gt_monofill_mask/"
                            input_intrinsics = pickle.load(open(
                                opt.eval_data_path + "intrinsic_matrixes.pkl",'rb'))
                            with open("./log/"+opt.eval_txt,"a") as write_file:
                                    write_file.write("Evaluation at iter [" + str(step)+"]: \n")

                            for mode in modes:
                                test_result_depth, test_result_normal = [], []
                                test_fn = root_img_path+"test_files_"+mode+".txt"
                                with open(test_fn) as f:
                                    for file in f:
                                        if file.split("/")[-1].split("_")[0] in input_intrinsics:
                                            input_intrinsic = np.array([input_intrinsics[file.split("/")[-1].split("_")[0]]])[:,[0,4,2,5]]
                                        else:
                                            input_intrinsic = [[opt.img_width, opt.img_height, 0.5*opt.img_width, 0.5*opt.img_height]]
                                        img = sm.imresize(sm.imread(root_img_path+file.rstrip()), (opt.img_height, opt.img_width))
                                        img = np.expand_dims(img, axis=0)
                                        pred_depth2_np, pred_normal_np = sess.run(
                                                [self.pred_depth_test, self.pred_normal_test],
                                                feed_dict = {self.inputs: img,
                                                    self.input_intrinsics: input_intrinsic})
                                        test_result_depth.append(np.squeeze(pred_depth2_np))
                                        # pred_normal_np = np.squeeze(pred_normal_np)
                                        # pred_normal_np[:,:,1] *= -1
                                        # pred_normal_np[:,:,2] *= -1
                                        # pred_normal_np = (pred_normal_np + 1.0) / 2.0
                                        test_result_normal.append(pred_normal_np)

                                ## depth evaluation
                                print ("Evaluation at iter ["+str(step)+"] "+mode)
                                gt_depths, pred_depths, gt_disparities = kitti_eval.load_depths(test_result_depth, mode, root_img_path, test_fn)
                                abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = kitti_eval.eval_depth(gt_depths, pred_depths, gt_disparities, mode)

                                ## normal evaluation
                                if mode == "kitti":
                                    pred_normals, gt_normals = normal_eval.load_normals(test_result_normal, mode, normal_gt_path, test_fn)
                                    dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30 = normal_eval.eval_normal(pred_normals, gt_normals, mode)

                                with open("../eval/"+opt.eval_txt,"a") as write_file:
                                    write_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))
                                    if mode == "kitti":
                                        write_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30))

                        ## evaluation for nyuv2 dataset
                        if dataset_name == "nyuv2":
                            root_img_path = "/home/zhenheng/datasets/nyuv2/"
                            normal_gt_path = "/home/zhenheng/datasets/nyuv2/normals_gt/"
                            test_fn = "/home/zhenheng/datasets/nyuv2/test_file_list.txt"
                            input_intrinsic = [[5.1885790117450188e+02, 5.1946961112127485e+02, 3.2558244941119034e+02, 2.5373616633400465e+02]]
                            print ("Evaluation at iter ["+str(step)+"] ")
                            abs_rel, sq_rel, rms, log_rms, a1, a2, a3, \
                            dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30 = self.evaluation_depth_normal_nyu(sess, root_img_path, normal_gt_path, test_fn, input_intrinsic)
                            with open("../eval/"+opt.eval_txt,"a") as write_file:
                                write_file.write("Evaluation at iter [" + str(step)+"]: \n")
                                write_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))
                                write_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30))

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % (opt.save_latest_freq * 2) == 0:
                    self.save(sess, opt.checkpoint_dir, gs)


    def evaluation_depth_normal_nyu(self, sess, root_img_path, normal_gt_path, test_fn, input_intrinsic):
        test_result_depth, test_result_normal = [], []
        with open(test_fn) as f:
            for file in f:
                img = sm.imresize(sm.imread(root_img_path+file.rstrip()), (self.opt.img_height, self.opt.img_width))
                img = np.expand_dims(img, axis=0)
                pred_depth2_np, pred_normal_np = sess.run(
                    [self.pred_depth_test, self.pred_normal_test],
                    feed_dict = {self.inputs: img, self.input_intrinsics: input_intrinsic})

                test_result_depth.append(np.squeeze(pred_depth2_np))
                pred_normal_np[:,:,1] *= -1
                pred_normal_np[:,:,2] *= -1
                pred_normal_np = (pred_normal_np + 1.0) / 2.0
                test_result_normal.append(pred_normal_np)
        ## evaluate depth estimation
        gt_depths, pred_depths, gt_disparities = kitti_eval.load_depths(test_result_depth, "nyuv2", root_img_path, test_fn)
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = kitti_eval.eval_depth(gt_depths, pred_depths, gt_disparities, "nyuv2")
        ## evaluate normal estimation
        pred_normals, gt_normals = normal_eval.load_normals(test_result_normal, "nyuv2", normal_gt_path, test_fn)
        dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30 = normal_eval.eval_normal(pred_normals, gt_normals)

        return abs_rel, sq_rel, rms, log_rms, a1, a2, a3, dgr_mean, dgr_median, dgr_11, dgr_22, dgr_30


    def depth_with_normal(self, depth, intrinsic_mtx, tgt_image,
                          depth_inverse=False):
        with tf.name_scope('depth_with_normal'):
            intrinsics = tf.concat([
                            tf.expand_dims(intrinsic_mtx[:,0,0],1),
                            tf.expand_dims(intrinsic_mtx[:,1,1],1),
                            tf.expand_dims(intrinsic_mtx[:,0,2],1),
                            tf.expand_dims(intrinsic_mtx[:,1,2],1)], 1)
            pred_depth_tensor = tf.squeeze(depth, axis=3)
            pred_normal = d2n.depth2normal_layer_batch(
                               pred_depth_tensor, intrinsics, depth_inverse)
            pred_depth2 = n2d.normal2depth_layer_batch(
                               pred_depth_tensor, pred_normal,
                               intrinsics, tgt_image)
            pred_depth2 = tf.expand_dims(pred_depth2, -1)

        return pred_depth2

    def build_depth_normal_test_graph(self):

        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        intrinsics = tf.placeholder(tf.float32, [self.batch_size, 4])

        input_mc = self.preprocess_image(input_uint8)
        # with tf.variable_scope('training', reuse=True):

        with tf.name_scope("depth_prediction"):
            pred_disp, pred_edges, depth_net_endpoints = nets.disp_net(input_mc, do_edge=True)
            pred_disp = [d[:,:,:,:1] for d in pred_disp]
            pred_depth = [1. / disp for disp in pred_disp]
            pred_normal = d2n.depth2normal_layer_batch(
                    tf.squeeze(pred_depth[0], axis=3), intrinsics, False)
            pred_depths2 = n2d.normal2depth_layer_batch(
                tf.squeeze(pred_depth[0], axis=3), pred_normal, intrinsics, input_mc, nei=1)
            pred_depths2_avg = pred_depths2
            # pred_depths2_avg = tf.reduce_mean([pred_depths2[i] for i in range(len(pred_depths2))], axis=0)
            print("shape of pred_depths2_avg")
            print(pred_depths2_avg.shape)
            print("shape of pred_normal")
            print(pred_normal.shape)

        self.inputs = input_uint8
        self.input_intrinsics = intrinsics
        self.pred_edges_test = pred_edges
        self.pred_depth_test = pred_depth[0]
        self.pred_depth2_test = tf.expand_dims(pred_depths2_avg, axis=-1)
        self.pred_normal_test = pred_normal
        self.pred_disp_test = pred_disp
        self.depth_epts = depth_net_endpoints


    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        # intrinscs = tf.placeholder(tf.float32, [self.batch_size, 4])

        input_mc = self.preprocess_image(input_uint8)
        with tf.variable_scope('training', reuse=True):
            with tf.name_scope("depth_prediction"):
                pred_disp, depth_net_endpoints = nets.disp_net(input_mc)
                pred_depth = [1./disp for disp in pred_disp]

        self.inputs = input_uint8
        self.pred_depth_test = pred_depth[0]
        self.pred_disp_test = pred_disp
        self.depth_epts = depth_net_endpoints

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        batch_size=1):

        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size

        if self.mode == 'depth':
            self.build_depth_normal_test_graph()

    def inference(self, inputs, intrinsics, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth_test
            fetches['disp'] = self.pred_disp_test
            fetches['edges'] = self.pred_edges_test
            if intrinsics != []:
                fetches['depth2'] = self.pred_depth2_test
                fetches['normals'] = self.pred_normal_test
                results = sess.run(fetches, feed_dict={self.inputs:inputs, self.input_intrinsics:intrinsics})
            else:
                results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def unpack_image_sequence(self, image_seq):
        opt = self.opt
        # Assuming the center image is the target frame
        tgt_start_idx = int(opt.img_width * (opt.num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, opt.img_width, -1])
        # Source fames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(opt.img_width * (opt.num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + opt.img_width), 0],
                               [-1, int(opt.img_width * (opt.num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image = tf.concat([tf.slice(src_image_seq,
                                    [0, i*opt.img_width, 0],
                                    [-1, opt.img_width, -1])
                                    for i in range(opt.num_source)], axis=2)
        src_image.set_shape([opt.img_height,
                             opt.img_width,
                             opt.num_source * 3])
        tgt_image.set_shape([opt.img_height, opt.img_width, 3])
        return tgt_image, src_image


    def unpack_image_sequence_list(self, image_seq):
        opt = self.opt
        # Assuming the center image is the target frame
        tgt_start_idx = int(opt.img_width * (opt.num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, opt.img_width, -1])

        # Source fames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(opt.img_width * (opt.num_source//2)), -1])

        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + opt.img_width), 0],
                               [-1, int(opt.img_width * (opt.num_source//2)), -1])
        src_image_seq = [src_image_1, src_image_2]

        for src_image in src_image_seq:
            src_image.set_shape([opt.img_height, opt.img_width, 3])

        tgt_image.set_shape([opt.img_height, opt.img_width, 3])

        return tgt_image, src_image_seq


    def get_multi_scale_intrinsics(self, raw_cam_mat, num_scales):
        proj_cam2pix = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = raw_cam_mat[0,0]/(2 ** s)
            fy = raw_cam_mat[1,1]/(2 ** s)
            cx = raw_cam_mat[0,2]/(2 ** s)
            cy = raw_cam_mat[1,2]/(2 ** s)
            r1 = tf.stack([fx, 0, cx])
            r2 = tf.stack([0, fy, cy])
            r3 = tf.constant([0.,0.,1.])
            proj_cam2pix.append(tf.stack([r1, r2, r3]))
        proj_cam2pix = tf.stack(proj_cam2pix)
        proj_pix2cam = tf.matrix_inverse(proj_cam2pix)
        proj_cam2pix.set_shape([num_scales,3,3])
        proj_pix2cam.set_shape([num_scales,3,3])
        return proj_cam2pix, proj_pix2cam


    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list


    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
