import tensorflow as tf
import numpy as np
import scipy.misc as sm
from utils import *
import os



def preprocess_image(image):
    # Assuming input image is uint8
    image = np.array(image, dtype=np.float32)
    return image * 2. -1.

def generate_extrinsic(extrinsic_path, extrinsic_list):
	extrinsic_mtx = []
	with open(extrinsic_path + extrinsic_list[-1]) as f:
		lines = f.readlines()
	for i, line in enumerate(lines):
		if i % 3 == 0:
			extrinsic_mtx.append([])
		extrinsic_mtx[i/3].append([float(item) for item in lines[i].rstrip().split(" ")])
	extrinsic_mtx = np.array(extrinsic_mtx)
	print(extrinsic_mtx.shape)

	return extrinsic_mtx ##shape is [img_num, 3, 4]

def convert_extrincmtx_vector(extrinsic_mtx, intrinsic_mtx):
	extrinsic_vector = [0.0 ,0.0, 0.0]
	# extrinsic_vector = extrinsic_mtx[:, 3] / intrinsic_mtx[0, 0, 0]
	# extrinsic_vector = extrinsic_mtx[:, 3]
	r_x = np.arctan2(extrinsic_mtx[2,1], extrinsic_mtx[2,2])
	r_z = np.arcsin(-1*extrinsic_mtx[2,0])
	r_y = np.arctan2(extrinsic_mtx[1,0], extrinsic_mtx[0,0])
	extrinsic_vector = np.hstack((extrinsic_vector, [r_x, r_y, r_z]))

	return extrinsic_vector # shape is [6,]

def read_intrinsic(intrinsic_file):
	intrinsic = []
	with open(intrinsic_file) as f:
		for line in f:
			intrinsic.append([float(ele) for ele in line.rstrip().split(" ")])

	return np.array(intrinsic)


root_path = "/home/zhenheng/datasets/sun3d/brown_bm_2/brown_bm_2/"
save_path = "/home/zhenheng/datasets/sun3d/warp_examples/"
img_path = root_path + "image/"
depth_path = root_path + "depth/"
extrinsic_path = root_path + "extrinsics/"
intrinsic_file = root_path+"intrinsics.txt"

img_list, depth_list = [], []
for file in os.listdir(img_path):
	if file[0] != ".":
		img_list.append(file)
img_list.sort()
for file in os.listdir(depth_path):
	if file[0] != ".":
		depth_list.append(file)
depth_list.sort()
extrinsic_list = os.listdir(extrinsic_path)
extrinsic_list.sort()

image_ph = tf.placeholder(tf.float32, [1, 480, 640, 3])
depth_ph = tf.placeholder(tf.float32, [1, 480, 640])
pose_ph = tf.placeholder(tf.float32, [1, 4, 4])
intrinsic_ph = tf.placeholder(tf.float32, [1, 3, 3])
invintrinsic_ph = tf.placeholder(tf.float32, [1, 3, 3])

extrinsic_mtx = generate_extrinsic(extrinsic_path, extrinsic_list)

intrinsic_mtx = read_intrinsic(intrinsic_file)
invintrinsic_mtx = np.linalg.inv(intrinsic_mtx)
intrinsic_mtx, invintrinsic_mtx = np.expand_dims(intrinsic_mtx, 0), np.expand_dims(invintrinsic_mtx, 0)

proj_img = inverse_warp(image_ph, depth_ph, pose_ph, intrinsic_ph, invintrinsic_ph, image_ph)

with tf.Session() as sess:

	for i in range(min(len(img_list), len(depth_list))):
		# extrinsic_vector = convert_extrincmtx_vector(extrinsic_mtx[i], intrinsic_mtx)
		extrinsic_vector = np.vstack((extrinsic_mtx[i], np.array([0.0, 0.0, 0.0, 1.0])))
		img = np.array(sm.imread(img_path + img_list[i]), dtype=np.float32)
		img = np.expand_dims(preprocess_image(img), 0)
		depth = np.array(sm.imread(depth_path + depth_list[i]), dtype=np.float32)
		depth /= 255.0
		depth = np.expand_dims(depth, 0)
		extrinsic_vector = np.expand_dims(extrinsic_vector, 0)

		input_dict = {image_ph:img, depth_ph:depth, pose_ph:extrinsic_vector, intrinsic_ph:intrinsic_mtx, invintrinsic_ph:invintrinsic_mtx}
		proj_img_np = np.squeeze(sess.run(proj_img, feed_dict=input_dict))
		sm.imsave(save_path+img_list[i], proj_img_np)



