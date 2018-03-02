from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob
import random

class citydriving_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split='train',
                 crop_bottom=False, # Get rid of the car logo
                 sample_gap=3,  # Sample every two frames to match KITTI frame rate
                 img_height=171, 
                 img_width=416,
                 seq_length=5):
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
        img_dir = self.dataset_dir + '/frames/'
        video_list = os.listdir(img_dir)
        frames = []
        for video in video_list:
            img_files = glob(img_dir + video + '/*.jpg')
            for f in img_files:
                frame_id = os.path.basename(f)
                rand_num = random.uniform(0,1)
                if rand_num <= 0.02:
                    frames.append(frame_id)
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        tgt_timestamp = self.frames[tgt_idx]
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_timestamp):
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
        _, video_id, tgt_local_frame_id = tgt_frame_id.split('.')[0].split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%08d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = 'video_%s_%s' % (video_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'frames/video_'+video_id, 
                                 curr_frame_id+".jpg")
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, crop_bottom):
        _, video_id, tgt_local_frame_id = tgt_frame_id.split('.')[0].split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%08d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = 'video_%s_%s' % (video_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'frames/video_'+video_id, 
                                 curr_frame_id+".jpg")
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
        example['folder_name'] = "video_"+tgt_frame_id.split('.')[0].split('_')[1]
        example['file_name'] = tgt_frame_id[:-4]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out