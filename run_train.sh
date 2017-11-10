python3 train.py --dataset_dir=/home/zhenheng/datasets/kitti/eigen_process_832_256 \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_wedge3d_l2_alpha10_clip0_wt2_normal_wedgel2_alpha0.1_wt0.01_edge_input417_l2_deconvk4_noscaling_wt0.15_expwt0.0_sfmpy0723_depth1normal_eval_cont \
	--eval_txt eval_kitti_d2nn2d_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_wedge3d_l2_alpha10_clip0_wt2_normal_wedgel2_alpha0.1_wt0.01_edge_input417_l2_deconvk4_noscaling_wt0.15_expwt0.0_sfmpy0723_depth1normal_eval_cont.txt \
	--img_width=832 --img_height=256 --batch_size=4 \
	--smooth_weight=2 --explain_reg_weight=0.0 --edge_mask_weight=0.15 --edge_as_explain 0.0\
	--normal_smooth_weight=0.01 --img_grad_weight=0 --occ_mask=0.0 --depth_consistency=0.0 \
	--gpu_id 0 --gpu_fraction 0.4\
	--continue_train False
	# --checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_dilated2_4pteadw0.0_noflyout_depthsmooth_smooth0.25_sfmpy0723_depth1normal_eval/model-50001
	#--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_normalnei1_depth1_eval/model-30001
