from __future__ import division
import os
import sys
sys.path.append("../eval")
import numpy as np
import scipy.misc
import tensorflow as tf
import argparse
from SfMLearner import SfMLearner
from evaluate_kitti import *
from utils import *


def test_image(filename):

    mode = 'depth'
    img_height=128
    img_width=416
    # ckpt_file = 'models/model-145248'
    # ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/sfmlearner/chpts/model.latest'
    ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/d2nn2d_1pt/depth2normal_test/model-62875'
    img_path = "/home/zhenheng/Datasets_ssd/Datasets/kitti/training/image_2/"
    # I = scipy.misc.imread('misc/sample.png')
    I = scipy.misc.imread(img_path+filename)
    I = scipy.misc.imresize(I, (img_height, img_width))

    sfm = SfMLearner()
    sfm.setup_inference(img_height, img_width, mode=mode)

    saver = tf.train.Saver([var for var in tf.trainable_variables()])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) 
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, ckpt_file)
        pred = sfm.inference(I[None,:,:,:], sess, mode=mode)
    # np.save("./visualization.npy",normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.figure()
    # plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.savefig("./visualization.png", bbox_inches="tight")
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1); plt.imshow(I)
    # plt.subplot(1,2,2); plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.savefig("./visualization.pdf", bbox_inches="tight")
    scipy.misc.imsave("./visualize_2.png", normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # np.save("./1.npy", pred['disp'])

def test_filelist(filelist, split, eval_depth_bool, ckpt_file):

    if split == "kitti":
        intrinsic_matrixes = pickle.load(open("/home/zhenheng/datasets/kitti/intrinsic_matrixes.pkl", "rb"))
    
    save_path = "/home/zhenheng/Works/SfMLearner/eval/kitti/"
    root_img_path = "/home/zhenheng/datasets/kitti/"
    mode = 'depth'
    img_height=128
    img_width=416
    # ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/d2nn2d_1pt_new/model-40249'

    sfm = SfMLearner()
    with tf.variable_scope("training"):
        sfm.setup_inference(img_height, img_width, mode=mode)

    saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # with tf.name_scope("training"):
        saver.restore(sess, ckpt_file)
        npyfile = []
        for file in filelist:
            resize_ratio = img_height / 435.0
            # intrinsic = np.expand_dims(np.array(intrinsic_matrixes[file.split("/")[-1].split("_")[0]])[[0,4,2,5]] * resize_ratio, axis=0)
            I = scipy.misc.imread(file)
            I = scipy.misc.imresize(I, (img_height, img_width))

            # pred = sfm.inference(I[None,:,:,:], intrinsic, sess, mode=mode)
            pred = sfm.inference(I[None,:,:,:], [], sess, mode=mode)
            npyfile.append(pred['depth'][0,:,:,0])

    if eval_depth_bool:
        gt_depths, pred_depths, gt_disparities = load_depths(npyfile, split, root_img_path)
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth(gt_depths, pred_depths, gt_disparities, split, vis=True)
            # scipy.misc.imsave(save_path+"visualization/"+file.split("/")[-1], normalize_depth_for_display(pred['depth'][0,:,:,0]))
        #np.save(save_path+"npy_files/sfmlearner_depth.npy",npyfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
    parser.add_argument('--split', type=str, help='data split, kitti or eigen', required=True)
    parser.add_argument('--gpu_id', type=str, help='gpu id for evaluation', default="0")
    parser.add_argument('--ckpt_file', type=str, help='model checkpoint', required=True, default='models/model-145248')
    parser.add_argument('--type',  type=str, help='test type, img or filelist', default='filelist')
    parser.add_argument('--eval_depth_bool', type=bool, help="evaluate the depth estimation based on standard metrics", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    if args.type == "filelist":
        filelist = []
        ## kitti
        if args.split == "kitti":
            img_path = "/home/zhenheng/datasets/kitti/training/image_2/"

            files = os.listdir(img_path)
            for file in files:
                if "10.png" in file:
                    filelist.append(img_path+file)
            filelist.sort()
            test_filelist(filelist, args.split, args.eval_depth_bool, args.ckpt_file)
            ## eigen
        else:
            test_file_list = "/home/zhenheng/datasets/kitti/test_files_eigen.txt"
            with open(test_file_list) as f:
                for line in f:
                    filelist.append("/home/zhenheng/datasets/kitti/"+line.rstrip())
            test_filelist(filelist, args.split, args.eval_depth_bool, args.ckpt_file)
    else:
        filename = "000001_10.png"
        test_image(filename)
