import tensorflow as tf
import numpy as np
import scipy.misc as sm
from flowlib import read_flow_png
from optical_flow_warp_fwd import transformerFwd

def compute_3dpts_batch(pts, intrinsics):
    
    ## pts is the depth map of rank3 [batch, h, w], intrinsics is in [batch, 4]
    pts = tf.squeeze(pts, axis=3)
    fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3] 
    pts_shape = pts.get_shape().as_list()
    pts_3d = tf.zeros(pts.get_shape().as_list()+[3])
    pts_z = pts
    x = tf.range(0, pts.get_shape().as_list()[2])
    x = tf.cast(x, tf.float32)
    y = tf.range(0, pts.get_shape().as_list()[1])
    y = tf.cast(y, tf.float32)
    cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    pts_x = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[0], 0), [pts_shape[0], 1, 1]) - cx_tile) / fx_tile * pts
    pts_y = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[1], 0), [pts_shape[0], 1, 1]) - cy_tile) / fy_tile * pts
    pts_3d = tf.concat([[pts_x], [pts_y], [pts_z]], 0)
    pts_3d = tf.transpose(pts_3d, perm = [1,2,3,0])

    return pts_3d

def gen_3d_flow(depth1, depth2, flow):

    depth1 = tf.cast(depth1, tf.float32)
    depth2 = tf.cast(depth2, tf.float32)
    flow = tf.cast(flow, tf.float32)
    batch, height, width = depth1.get_shape().as_list()[:3]
    intrinsics = tf.tile(tf.constant([[width, height, 0.5*width, 0.5*height]], tf.float32), [batch, 1])
    pts_3d_1 = compute_3dpts_batch(depth1, intrinsics)
    pts_3d_2 = compute_3dpts_batch(depth2, intrinsics)
    x = tf.range(0, width)
    y = tf.range(0, height) 
    x, y = tf.meshgrid(x,y)
    x = tf.tile(tf.cast(x[None,:,:], tf.float32), [batch,1,1])
    y = tf.tile(tf.cast(y[None,:,:], tf.float32), [batch,1,1])
    new_x = tf.rint(x+flow[:,:,:,0])
    new_y = tf.rint(y+flow[:,:,:,1])
    flyout_mask = tf.cast((new_x<0) | (new_x>=width) | (new_y<0) | (new_y>=height), tf.float32)
    new_x = tf.cast(tf.clip_by_value(new_x, 0, width-1), tf.int32)
    new_y = tf.cast(tf.clip_by_value(new_y, 0, height-1), tf.int32)
    x, y = tf.cast(x, tf.int32), tf.cast(y, tf.int32)
    new_coords = tf.concat([new_y[:,:,:,None],new_x[:,:,:,None]], axis=3)
    old_coords = tf.concat([y[:,:,:,None], x[:,:,:,None]], axis=3)
    hom_filler = tf.cast(tf.concat([tf.ones([1, height, width,1])*i for i in range(batch)], axis=0), tf.int32)
    print(hom_filler.get_shape().as_list())
    print(new_coords.get_shape().as_list())
    new_coords = tf.concat([hom_filler, new_coords], axis=3)
    old_coords = tf.concat([hom_filler, old_coords], axis=3)
    new_inds = tf.reshape(new_coords, [batch*height*width, 3])
    old_inds = tf.reshape(old_coords, [batch*height*width, 3])
    
    new_3d_points = tf.gather_nd(pts_3d_2, new_inds)
    old_3d_points = tf.gather_nd(pts_3d_1, old_inds)
    new_3d_map = tf.reshape(new_3d_points, [batch, height, width, -1])
    old_3d_map = tf.reshape(old_3d_points, [batch, height, width, -1])
    flow_3d = (new_3d_map - old_3d_map)#*tf.tile(flyout_mask[:,:,None], [1,1,3])

    return flow_3d

def gen_occ_mask(flow):

    flow = tf.constant(flow, tf.float32)
    occ_mask = tf.clip_by_value(transformerFwd(tf.ones(shape = flow.get_shape().as_list(), dtype=tf.float32)), 
                                clip_value_min=0.0, clip_value_max=1.0)
    return occ_mask

def gen_occ_mask_numpy(depth1, depth2, flow):

    height, width = depth1.shape
    x, y = np.meshgrid(np.arange(0,width), np.arange(0,height))
    new_x, new_y = np.rint(x+flow[:,:,0]), np.rint(y+flow[:,:,1])
    new_xy = np.concatenate([new_x[:,:,None], new_y[:,:,None]], axis=2)
    new_xy_flatten = np.reshape(new_xy, [width*height, 2])
    _, u_inds, u_counts = np.unique(new_xy_flatten, axis=0, return_index=True, return_counts=True)
    # u_counts = 1-((u_counts-1).astype(np.bool)).astype(np.int)
    # u_inds *= u_counts
    occ_mask = np.zeros([width*height, 1])
    occ_mask[u_inds,0] = 1
    occ_mask = np.reshape(occ_mask, [height, width])
    return occ_mask

def main():

    depth1_dir = "/home/zhenheng/datasets/kitti/training/disp_occ_0/000000_10.png"
    depth1_dir = "/home/zhenheng/works/unsp_depth_normal/monodepth/res/disparities_kitti/000000_10.png"
    depth2_dir = "/home/zhenheng/datasets/kitti/training/disp_occ_1_warp/000000_11.png"
    depth2_dir = "/home/zhenheng/works/unsp_depth_normal/monodepth/res/disparities_kitti_2nd/000000_11.png"
    flow_dir = "/home/zhenheng/datasets/kitti/training/flow_occ/000000_10.png"
    flow_dir = "/home/zhenheng/works/unsp_depth_normal/occ_flow/res/kitti/000000_10.png"
    depth1 = sm.imread(depth1_dir)
    depth2 = sm.imread(depth2_dir)
    flow = read_flow_png(flow_dir)

    # occ_mask = gen_occ_mask(flow)
    flow_3d = eval_3d_flow(depth1, depth2, flow)
    with tf.Session() as sess:
        flow_3d_np = sess.run(flow_3d)
    # flow_3d_np = flow_3d_np*np.tile(occ_mask[:,:,None], [1,1,3])
    sm.imsave('./flow.png', np.uint16(flow_3d_np+32768))

if __name__ == "__main__":
    main()
