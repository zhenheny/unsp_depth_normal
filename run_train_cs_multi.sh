python3 train.py \
    --dataset_dir=/home/zhenheng/datasets/cityscapes/frame_seq_832_342_gap4/ \
    --eval_data_path=/home/zhenheng/datasets/kitti/ \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/cs_dilated2_d2nnei3_n2dedgeremove_stereomono_depthsmooth_noedge_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_input417_l2_deconvk4_noscaling_wt0.2_expwt0.0_dm_xz_2loss_l1wt0.01_depth4pose_depth1normal_eval_edgepretrainedk_pwc_m \
	--eval_txt=ektc_dilated2_d2nnei3_n2dedgeremove_stereomono_depthsmooth_noedge_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_input417_l2_deconvk4_noscaling_wt0.2_expwt0.0_dm_xz_2loss_l1wt0.01_depth4pose_depth1normal_eval_pretrainedk_pwc_m.txt \
	--img_width=832 \
    --img_height=256 \
    --batch_size=2 \
	--smooth_weight=2 \
    --edge_as_explain=0.0 \
    --explain_reg_weight=0.0 \
    --edge_mask_weight=0.2 \
    --ssim_weight=0.1 \
    --img_grad_weight=0.1 \
	--normal_smooth_weight=0.05 \
    --occ_mask=0.0 \
    --depth_consistency=0.0 \
	--dense_motion_weight=0.03 \
	--gpu_id=0,1 \
    --gpu_fraction=0.5 \
    --depth4pose=True \
    --motion_net=pwc \
	--continue_train=True \
    --trainable_var_scope=/edge/,/motion_net/,/dense_motion_pwc_net/ \
    --rm_var_scope=/dense_motion_pwc_net/,/motion_net/ \
    --checkpoint_continue=/home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_stereomono_ssim_wt1.0_depthsmooth_wedge3d_l2_alpha10_clip0_wt4_normal_wedgel2_alpha0.1_wt0.1_edge_input417_l2_deconvk4_noscaling_wt0.6_expwt0.0_sfmpy0723_depth1normal_eval/model-100002
	# --checkpoint_continue=./output/cs_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_noedge_wedgel2_alpha10_clip0_wt2_edge_lossscalefactor_input417_l2_deconvk4_nnupsample_noscaling_wt0.2_expwt0.0_dm_xz_2loss_l1wt0.01_depth4pose_depth1normal_eval_edgepretrained_cs/model.latest
