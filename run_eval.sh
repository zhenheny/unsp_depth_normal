sudo python3 main.py --split cs \
	--ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/cs_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_noedge_wedgel2_alpha10_clip0_wt4_normal_smooth_wt0.05_edge_lossscalefactor_input417_l2_deconvk4_nnupsample_noscaling_wt0.2_expwt0.8_sfmpy0723_depth1normal_eval_cont/model-160002 \
	--gpu_id 1
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_d2nnei3_depthsmooth_wedgel2_alpha10_clip0_wt2_edge_l2_deconvk4_noedgeinput_noscaling_wt0.1_expwt0.4_sfmpy0723_depth1normal_eval_cont/model-80002 \
#--ckpt_file /home/zhenheng/works/unsp_depth_normal/SfMLearner/models/model-145248 \
# # --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_smoothloss_depth0edge_depth1normal_eval/model-140001  \
# # --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_avgdepth_depthsmooth_smoothwt0.25_pxlossnoedge_depth1normal_eval/model-50001 \
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smooth0.25_oldsfmpy0723_depth1normal_eval_new/model-50001
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_imggrad0.25_pxlossnoedge_depth1normal_eval/model-50001 \
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_depthsmooth_wedgel2_alpha10_clip0_wt0.2_edgel2_noscaling_wt0.2_expwt0.4_sfmpy0723_depth1normal_eval_cont/model-80002 \