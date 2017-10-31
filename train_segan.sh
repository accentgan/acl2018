#!/bin/bash

export CUDA_VISIBLE_DEVICES="1" 
python main.py --init_noise_std 0. --save_path vctk_ddpg_ne_neww --model gru --slice_num 6 --num_runs 1 \
                                          --init_l1_weight 100. --batch_size 16 --g_nl prelu --accent_class 20\
                                          --preemph 0.95 --epoch 100 --bias_deconv True --start_epoch 2 \
                                          --bias_downconv True --bias_D_conv True --e2e_dataset data/vctk.tfrecords \
																					--sample_rate 8192 \
