sudo python3 main.py --split kitti \
	 --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_4pt0.0_noflyout_dilated2_depthsmooth_wedgel2_alpha10_clip0_wt0.2_edgel2_noscaling_wt0.2_expwt0.4_sfmpy0723_depth1normal_eval_cont/model-80002 \
	--gpu_id 1
#--ckpt_file /home/zhenheng/works/unsp_depth_normal/SfMLearner/models/model-145248 \
# # --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_smoothloss_depth0edge_depth1normal_eval/model-140001  \
# # --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_avgdepth_depthsmooth_smoothwt0.25_pxlossnoedge_depth1normal_eval/model-50001 \
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smooth0.25_oldsfmpy0723_depth1normal_eval_new/model-50001
# --ckpt_file /home/zhenheng/tf_events/unsp_depth_normal/d2nn2d_1pty0_depthsmooth_smoothwt0.25_imggrad0.25_pxlossnoedge_depth1normal_eval/model-50001 \