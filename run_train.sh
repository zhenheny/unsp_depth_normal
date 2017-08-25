python3 train.py --dataset_dir=/home/zhenheng/datasets/kitti/eigen_process \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smooth0.25_sfmpy0723_bnrestoredecay0.9_depth1normal_eval \
	--eval_txt eval_kitti_d2nn2d_1pty0_depthsmooth_smooth0.25_sfmpy0723_bnrestoredecay0.9_depth1normal_eval.txt \
	--img_width=416 --img_height=128 --batch_size=4 \
	--smooth_weight=0.25 --normal_smooth_weight=0 --img_grad_weight=0 --explain_reg_weight=0.4 --occ_mask=0.0 --depth_consistency=0.0\
	--gpu_id 1 --gpu_fraction 0.4\
	--continue_train False \
	--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smooth0.25_oldsfmpy0723_depth1normal_eval_new/model-40001


	#--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_normalnei1_depth1_eval/model-30001
