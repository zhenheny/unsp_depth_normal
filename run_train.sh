python3 train.py --dataset_dir=/home/zhenheng/datasets/kitti/eigen_process \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_clip_4ptedaw1.0_depthsmooth_smooth0.25_pxlossnoedge_expwt0_occmask1_depth1normal_eval \
	--eval_txt eval_kitti_d2nn2d_clip_4ptedaw1.0_depthsmooth_smooth0.25_pxlossnoedge_expwt0_occmask1_depth1normal_eval.txt \
	--img_width=416 --img_height=128 --batch_size=4 \
	--smooth_weight=0.25 --normal_smooth_weight=0 --img_grad_weight=0 --explain_reg_weight=0 --occ_mask=1.0 --gpu_id 0 --gpu_fraction 0.8\
	--continue_train False
	

	#--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_normalnei1_depth1_eval/model-30001