from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "nyuv2", "citydriving"])
# parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
parser.add_argument("--sample_gap", type=int, default=1, help="gap between two warping frames")
parser.add_argument("--subset", type=str, default="", help="subset of dataset scenes that contain the keyword")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root+"/", example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)
        dump_root = args.dataset_dir+"/frame_seq_"+str(args.img_width)+"_"+str(args.img_height)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)        

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)
        dump_root = args.dataset_dir+"/frame_seq_"+str(args.img_width)+"_"+str(args.img_height)

    if args.dataset_name == 'citydriving':
        from citydriving.citydriving_loader import citydriving_loader
        data_loader = citydriving_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)
        dump_root = args.dataset_dir+"/frame_seq_"+str(args.img_width)+"_"+str(args.img_height)

    if args.dataset_name == "nyuv2":
        from nyuv2.nyuv2_loader import nyuv2_loader
        data_loader = nyuv2_loader(args.dataset_dir,
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       sample_gap=args.sample_gap,
                                       subset=args.subset)
        dump_root = args.dataset_dir+"/data_format_train_gap"+str(args.sample_gap)+"_"+args.subset

    args.dump_root = dump_root
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + '/train.txt', 'w') as tf:
        with open(args.dump_root + '/val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.jpg')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()

