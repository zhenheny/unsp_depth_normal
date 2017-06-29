from __future__ import division
import os
import sys
import time
import math
import random
import numpy as np
import scipy.misc as sm
import tensorflow as tf
from nets import *
from utils import *
sys.path.insert(0, "/home/zhenheng/works/unsp_depth_normal/depth2normal/")
sys.path.append("../eval")
from depth2normal_tf import *
from normal2depth_tf import *
from evaluate_kitti import *

class SfMLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        with tf.name_scope("data_loading"):
            seed = random.randint(0, 2**31 - 1)
            # seed = 654

            # Load the list of training files into queues
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
            tgt_image, src_image_stack = \
                self.unpack_image_sequence(image_seq)

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
            src_image_stack, tgt_image, proj_cam2pix, proj_pix2cam = \
                    tf.train.batch([src_image_stack, tgt_image, proj_cam2pix, 
                                    proj_pix2cam], batch_size=opt.batch_size)
            print ("tgt_image batch images shape:")
            print (tgt_image.get_shape().as_list())

        ## depth prediction network
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(tgt_image, 
                                                      is_training=True)
            pred_depth = [1./d for d in pred_disp]


        with tf.name_scope("pose_and_explainability_prediction"):
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_exp_net(tgt_image,
                             src_image_stack, 
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            pred_normals = []
            pred_disps2 = []
            depth_inverse = False
            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    # Construct a reference explainability mask (i.e. all 
                    # pixels are explainable)
                    ref_exp_mask = self.get_reference_explain_mask(s)
                # Scale the source and target images for computing loss at the 
                # according scale.
                curr_tgt_image = tf.image.resize_bilinear(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_bilinear(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                print ("the shape of curr_tgt_image")
                print (curr_tgt_image.get_shape().as_list())

                ## depth2normal and normal2depth at each scale level
                intrinsic_mtx = proj_cam2pix[:,s,:,:]
                intrinsics = tf.concat([tf.expand_dims(intrinsic_mtx[:,0,0],1), tf.expand_dims(intrinsic_mtx[:,1,1],1), tf.expand_dims(intrinsic_mtx[:,0,2],1), tf.expand_dims(intrinsic_mtx[:,1,2],1)], 1)
                pred_depth_tensor = tf.squeeze(pred_depth[s])

                pred_normal = depth2normal_layer_batch(pred_depth_tensor, intrinsics, depth_inverse)

                # pred_depth2 = normal2depth_layer_batch(pred_depth_tensor, tf.squeeze(pred_normal), intrinsics)
                # pred_depth2 = tf.expand_dims(pred_depth2, -1)
                # # normal depth2 to avoid corner case of preddepth2=0
                # pred_disp2 = 1.0 / (pred_depth2 - tf.reduce_min(pred_depth2) + 1e-2)

                # # print (pred_depth2.shape)

                pred_depths2 = normal2depth_layer_batch(pred_depth_tensor, tf.squeeze(pred_normal), intrinsics)

                if opt.smooth_weight > 0:
                    for i in range(len(pred_depths2)):
                        pred_disp2 = tf.expand_dims(1.0 / (pred_depths2[i] - tf.reduce_min(pred_depths2[i]) + 1e-2), -1)
                        smooth_loss += tf.multiply(opt.smooth_weight/(2**s), \
                        self.compute_smooth_loss(pred_disp2, curr_tgt_image))
                        # self.compute_smooth_loss(pred_disp2))
                    smooth_loss /= len(pred_depths2)
                    
                    pred_normals.append(pred_normal)
                    pred_disps2.append(pred_disp2)

                    # smooth_loss += tf.multiply(opt.smooth_weight/(2**s), \
                    #     self.compute_smooth_loss(pred_disp[s]))
                    #     # self.compute_smooth_loss(pred_disp2[:, 2:-2, :-4, :]))

                for i in range(opt.num_source):
                    # Cross-entropy loss as regularization for the 
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        curr_exp_logits = tf.slice(pred_exp_logits[s], 
                                                   [0, 0, 0, i*2], 
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)

                    # Inverse warp the source image to the target image frame
                    # Use pred_depth and 8 pred_depth2 maps for inverse warping
                    # curr_proj_image = inverse_warp(
                    #     curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                    #     pred_depths2[0], 
                    #     # pred_depth[s], 
                    #     pred_poses[:,i,:], 
                    #     proj_cam2pix[:,s,:,:], 
                    #     proj_pix2cam[:,s,:,:])
                    # curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)

                    # # Photo-consistency loss weighted by explainability
                    # if opt.explain_reg_weight > 0:
                    #     pixel_loss += tf.reduce_mean(curr_proj_error * \
                    #         tf.expand_dims(curr_exp[:,:,:,1], -1))
                    # else:
                    #     pixel_loss += tf.reduce_mean(curr_proj_error) 

                    # for j in range(len(pred_depths2)):

                    for j in range(len(pred_depths2)):
                        curr_proj_image = inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        pred_depths2[j], 
                        # pred_depth[s], 
                        pred_poses[:,i,:], 
                        proj_cam2pix[:,s,:,:], 
                        proj_pix2cam[:,s,:,:])
                        curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)

                    # Photo-consistency loss weighted by explainability
                        if opt.explain_reg_weight > 0:
                            pixel_loss += tf.reduce_mean(curr_proj_error * \
                                tf.expand_dims(curr_exp[:,:,:,1], -1))
                        else:
                            pixel_loss += tf.reduce_mean(curr_proj_error)
                    # pixel_loss /= len(pred_depths2)

                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack, 
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + smooth_loss + exp_loss
        

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            self.grads_and_vars = optim.compute_gradients(total_loss, 
                                                          var_list=train_vars)
            self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)

        self.pred_depth = pred_depth
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
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
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

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_edge_aware_smooth_loss(self, disp, image):
        ## compute edge aware smoothness loss
        ## image should be a rank 4 tensor

        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        disp_gradients_x, disp_gradients_y = gradient(disp)
        dx2, dxdy = gradient(disp_gradients_x)
        dydx, dy2 = gradient(disp_gradients_y)
        image_gradients_x, image_gradients_y = gradient(image)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y
        smoothness_dx2 = dx2 * weights_x[:,:,1:,:]
        smoothness_dxdy = dxdy * weights_x[:,1:,:,:]
        smoothness_dydx = dydx * weights_y[:,:,1:,:]
        smoothness_dy2 = dy2 * weights_y[:,1:,:,:]

        smoothness_loss = tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))
        smoothness_loss_2nd = tf.reduce_mean(tf.abs(smoothness_dx2)) + \
                             tf.reduce_mean(tf.abs(smoothness_dxdy)) + \
                             tf.reduce_mean(tf.abs(smoothness_dydx)) + \
                             tf.reduce_mean(tf.abs(smoothness_dy2))
        return smoothness_loss_2nd

        # disp_gradients_x = [self.gradient_x(d) for d in disp]
        # disp_gradients_y = [self.gradient_y(d) for d in disp]

        # image_gradients_x = [self.gradient_x(img) for img in pyramid]
        # image_gradients_y = [self.gradient_y(img) for img in pyramid]

        # weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        # weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        # smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        # smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        # return smoothness_x + smoothness_y

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        tf.summary.image("pred_normal", (self.pred_normals[0]+1.0)/2.0)
        tf.summary.image("pred_disp2", self.pred_disps2[0])
        # for s in range(opt.num_scales):
        s = 0
        tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
        tf.summary.image('scale%d_depth_image' % s, self.pred_depth[s])
        tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
        tf.summary.image('scale%d_target_image' % s, \
                         self.deprocess_image(self.tgt_image_all[s]))
        for i in range(opt.num_source):
            if opt.explain_reg_weight > 0:
                tf.summary.image(
                    'scale%d_exp_mask_%d' % (s, i), 
                    tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
            # tf.summary.image(
            #     'scale%d_source_image_%d' % (s, i), 
            #     self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
            tf.summary.image('scale%d_projected_image_%d' % (s, i), 
                self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
            tf.summary.image('scale%d_proj_error_%d' % (s, i),
                tf.expand_dims(self.proj_error_stack_all[s][:,:,:,i], -1))
        # tf.summary.histogram("tx", self.pred_poses[:,:,0])
        # tf.summary.histogram("ty", self.pred_poses[:,:,1])
        # tf.summary.histogram("tz", self.pred_poses[:,:,2])
        # tf.summary.histogram("rx", self.pred_poses[:,:,3])
        # tf.summary.histogram("ry", self.pred_poses[:,:,4])
        # tf.summary.histogram("rz", self.pred_poses[:,:,5])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)
        for grad, var in self.grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        with open("../eval/"+opt.eval_txt, "w") as f:
            f.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
        with tf.variable_scope("training"):
            self.build_train_graph()
        with tf.variable_scope("training", reuse=True):
            self.setup_inference(opt.img_height, opt.img_width, "depth")
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step], 
                                    max_to_keep=40)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)

        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                self.saver.restore(sess, checkpoint)
            for step in range(1, opt.max_steps):
                start_time = time.time()
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["smooth_loss"] = self.smooth_loss
                    fetches['pred_depth'] = self.pred_depth
                    fetches['pred_disp'] = self.pred_disp
                    fetches['pred_normal'] = self.pred_normals[0]
                    # fetches['pred_depth2'] = self.pred_depth2
                    fetches['pred_poses'] = self.pred_poses
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / opt.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * opt.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, opt.steps_per_epoch, \
                                time.time() - start_time, results["loss"]))

                    print(results['smooth_loss'])
                    if np.any(np.isnan(results['pred_depth'][-1])):
                        # np.save("./depth.npy", results['pred_depth'][-1])
                        print (results['pred_depth'][-1])
                        print ("-----------")
                        print (results["pred_normal"])
                        break

                if step % opt.eval_freq == 0:
                    modes = ["eigen", "kitti"]
                    root_img_path = "/home/zhenheng/datasets/kitti/"
                    test_result = []
                    for mode in modes[1:]:
                        filelist_path = root_img_path+"test_files_"+mode+".txt"
                        with open(filelist_path) as f:
                            for file in f:
                                img = sm.imresize(sm.imread(root_img_path+file.rstrip()), (opt.img_height, opt.img_width))
                                img = np.expand_dims(img, axis=0)
                                test_result.append(np.squeeze(sess.run(self.pred_depth_test, feed_dict = {self.inputs: img})))
                        
                        print ("Evaluation at iter ["+str(step)+"] "+mode)
                        gt_depths, pred_depths, gt_disparities = load_depths(test_result, mode, root_img_path)
                        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth(gt_depths, pred_depths, gt_disparities, mode)
                        with open("../eval/"+opt.eval_txt,"a") as write_file:
                            write_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % (opt.save_latest_freq * 2) == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth2_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        intrinsics = tf.placeholder(tf.float32, [self.batch_size, 4])

        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(input_mc)
            pred_depth = [1./disp for disp in pred_disp]
            print ("shape of pred depth test")   

            pred_normal = depth2normal_layer_batch(tf.squeeze(pred_depth[0], axis=3), intrinsics, False)
            pred_depth2 = normal2depth_layer_batch(tf.squeeze(pred_depth[0], axis=3), pred_normal, intrinsics)
        # pred_depth2 = 1.0 / pred_dsip2
        self.inputs = input_uint8
        self.input_intrinsics = intrinsics
        self.pred_depth_test = pred_depth[0]
        self.pred_depth2_test = tf.expand_dims(pred_depth2, axis=-1)
        self.pred_disp_test = pred_disp
        self.depth_epts = depth_net_endpoints

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        # intrinscs = tf.placeholder(tf.float32, [self.batch_size, 4])

        input_mc = self.preprocess_image(input_uint8)
        with tf.variable_scope('training'):
            with tf.name_scope("depth_prediction"):
                pred_disp, depth_net_endpoints = disp_net(input_mc)
                pred_depth = [1./disp for disp in pred_disp]
        #     pred_normal = depth2normal_layer_batch(pred_depth[0], intrinsics, False)
        #     pred_depth2 = normal2depth_layer_batch(pred_depth[0], tf.squeeze(pred_normal), intrinsics)
        # pred_depth2 = 1.0 / pred_dsip2

        self.inputs = input_uint8
        self.pred_depth_test = pred_depth[0]
        self.pred_disp_test = pred_disp
        self.depth_epts = depth_net_endpoints

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

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
            self.build_depth_test_graph()

    def inference(self, inputs, intrinsics, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth_test
            fetches['disp'] = self.pred_disp_test
            if intrinsics != []:
                fetches['depth2'] = self.pred_depth2_test
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
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*opt.img_width, 0], 
                                    [-1, opt.img_width, -1]) 
                                    for i in range(opt.num_source)], axis=2)
        src_image_stack.set_shape([opt.img_height, 
                                   opt.img_width, 
                                   opt.num_source * 3])
        tgt_image.set_shape([opt.img_height, opt.img_width, 3])
        return tgt_image, src_image_stack

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
