sudo python3 train_nyuv2.py --dataset_dir=/home/zhenheng/datasets/nyuv2/data_format_train_gap1_study \
	--checkpoint_dir=/home/zhenheng/tf_events/unsp_depth_normal/nyuv2_gap1_study_1pty0_depthsmooth_smoothwt0.25_expwt0_imggrad0.25_ddepth1_eval \
	--eval_txt eval_nyuv2_gap1_study_1pty0_depthsmooth_smoothwt0.25_expwt0_imggrad0.25_depth1_eval.txt \
	--img_width=320 --img_height=224 --batch_size=4 \
	--smooth_weight=0.25 --normal_smooth_weight=0 --explain_reg_weight=0 --img_grad_weight=0.25 --occ_mask=0 --gpu_id 0 --gpu_fraction 0.8 \
	--continue_train False
