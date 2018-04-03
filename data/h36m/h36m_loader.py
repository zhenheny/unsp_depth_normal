from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob
import random

class h36m_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split='train',
                 crop_bottom=False,
                 sample_gap=3,  # Sample three two frames to match KITTI frame rate
                 img_height=171, 
                 img_width=416,
                 seq_length=3):
        self.dataset_dir = dataset_dir
        self.split = split
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd!'
        self.frames = self.collect_frames()
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)
        
    def collect_frames(self):
        train_subs = ['S1','S5','S6','S7','S8']
        test_subs = ['S9','S11']
        img_dir = self.dataset_dir
        frames = []
        for subject in train_subs:
            video_list = os.listdir(img_dir+'/'+subject+'/frames/')
            for video in video_list:
                if video[-1] in ['1','8']:
                    img_files = glob(img_dir+'/'+subject+'/frames/'+video+'/*.jpg')
                    for f in img_files:
                        frame_id = f.split("//")[1]
                        rand_num = random.uniform(0,1)
                        if rand_num <= 0.05:
                            frames.append(frame_id[:-4]) ## add left-view images
                            if video[-1] == '1':
                                new_video = frame_id.split('/')[2].split(".")[0]+".60457274"
                            else:
                                new_video = frame_id.split('/')[2].split(".")[0]+'.54138969'
                            frames.append("/".join(frame_id.split("/")[:2])+"/"+new_video+"/"+frame_id.split("/")[-1][:-4])
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_frame_id):
            return False
        example = self.load_example(self.frames[tgt_idx])
        return example

    def load_intrinsics(self, frame_id, split):
        fx = np.float32(self.img_width)
        fy = np.float32(self.img_height)
        u0 = 0.5*fx
        v0 = 0.5*fy
        intrinsics = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0,  0,  1]])
        return intrinsics

    def is_valid_example(self, tgt_frame_id):
        subject_id, _, video_id, tgt_local_frame_id = tgt_frame_id.split('/')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%04d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s/frames/%s/%s' % (subject_id, video_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, curr_frame_id+".jpg")
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, crop_bottom):
        subject_id, _, video_id, tgt_local_frame_id = tgt_frame_id.split('/')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%04d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s/frames/%s/%s' % (subject_id, video_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, curr_frame_id+".jpg")
            curr_img = scipy.misc.imread(curr_image_file)
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height/raw_shape[0]
                zoom_x = self.img_width/raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            if crop_bottom:
                ymax = int(curr_img.shape[0] * 0.75)
                curr_img = curr_img[:ymax]
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y
    
    def load_example(self, tgt_frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length, self.crop_bottom)
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)

        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = "/".join(tgt_frame_id.split('/')[:-1])
        example['file_name'] = tgt_frame_id.split("/")[-1]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out