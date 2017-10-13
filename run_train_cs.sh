python3 train.py --dataset_dir=/home/zhenheng/datasets/cityscapes/frame_seq_832_342 \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/cs_4pt0.0_noflyout_dilated2_d2nnei3_depthsmooth_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_l2_nnupsample_noscaling_wt0.1_expwt0.4_sfmpy0723_depth1normal_eval_cont \
	--eval_txt eval_kitti_train_cs_4pt0.0_noflyout_dilated2_d2nnei3_depthsmooth_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_l2_nnupsample_noscaling_wt0.1_expwt0.4_sfmpy0723_depth1normal_eval_cont.txt \
	--img_width=832 --img_height=256 --batch_size=4 \
	--smooth_weight=2 --explain_reg_weight=0.4 --edge_mask_weight=0.1 --edge_as_explain 0.0\
	--normal_smooth_weight=0 --img_grad_weight=0 --occ_mask=0.0 --depth_consistency=0.0 \
	--gpu_id 0 --gpu_fraction 0.4\
	--continue_train False
	# --checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_dilated2_4pteadw0.0_noflyout_depthsmooth_smooth0.25_sfmpy0723_depth1normal_eval/model-50001
	#--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_normalnei1_depth1_eval/model-30001
