python3 train.py --dataset_dir=/home/zhenheng/datasets/h36m/frame_seq_832_832_gap3 \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_human_pose/d2nn2d_dilated2_d2nnei3_stereomono_ssim_wt1.0_depthsmooth_wedge_l2_alpha10_clip0_wt4_normal_wedgel2_alpha0.1_wt0.1_edge_input417_l2_wt0.6_depth1normal_eval \
	--eval_txt eval_kitti_d2nn2d_dilated2_d2nnei3_stereomono_ssim_wt1.0_depthsmooth_wedge_l2_alpha10_clip0_wt4_normal_wedgel2_alpha0.1_wt0.1_edge_input417_l2_wt0.6_depth1normal_eval.txt \
	--img_width=384 --img_height=384 --batch_size=4 \
	--smooth_weight=4 \
	--explain_reg_weight=0.0 \
	--edge_mask_weight=0.6 \
	--edge_as_explain 0.0 \
	--ssim_weight=1.0\
	--normal_smooth_weight=0.1 \
	--img_grad_weight=0 \
	--occ_mask=0.0 \
	--depth_consistency=0.0 \
	--gpu_id 1 --gpu_fraction 0.8\
	--rm_var_scope=/motion_net/,/edge/ \
	--continue_train False
	# --checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_stereomono_ssim_wt0.1_depthsmooth_wedge3d_l2_alpha10_clip0_wt2_normal_wedgel2_alpha0.1_wt0.01_edge_input417_l2_deconvk4_noscaling_wt0.15_expwt0.0_sfmpy0723_depth1normal_eval/model-60001
	# --checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_dilated2_4pteadw0.0_noflyout_depthsmooth_smooth0.25_sfmpy0723_depth1normal_eval/model-50001
	#--checkpoint_continue /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_normalnei1_depth1_eval/model-30001
