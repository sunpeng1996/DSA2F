
###
 # @Author: Wenhu Zhang
 # @Date: 2021-06-07 17:39:22
 # @LastEditTime: 2021-06-07 18:13:00
 # @LastEditors: Wenhu Zhang
 # @Description: 
 # @FilePath: /github/wh/DSA2F/launch_test.sh
### 

CUDA_VISIBLE_DEVICES="0" python -u main.py --phase test --test_dataset NJUD --begin_epoch 1 --end_epoch 97 --exp_name 0428debug > results/0428debug_NJUD.txt &
CUDA_VISIBLE_DEVICES="0" python -u main.py --phase test --test_dataset DUT-RGBD --begin_epoch 20 --end_epoch 97 --exp_name 0428debug > results/0428debug_DUT-RGBD.txt &
CUDA_VISIBLE_DEVICES="1" python -u main.py --phase test --test_dataset NLPR --begin_epoch 1 --end_epoch 97 --exp_name 0428debug > results/0428debug_NLPR.txt &
CUDA_VISIBLE_DEVICES="1" python -u main.py --phase test --test_dataset SSD --begin_epoch 20000 --end_epoch 20000 --exp_name 0428debug > results/0428debug_SSD.txt 

CUDA_VISIBLE_DEVICES="2" python -u main.py --phase test --test_dataset STEREO --begin_epoch 1 --end_epoch 97 --exp_name 0428debug > results/0428debug_STEREO.txt &
CUDA_VISIBLE_DEVICES="2" python -u main.py --phase test --test_dataset LFSD --begin_epoch 1 --end_epoch 97 --exp_name 0428debug > results/0428debug_LFSD.txt &
CUDA_VISIBLE_DEVICES="3" python -u main.py --phase test --test_dataset RGBD135 --begin_epoch 1 --end_epoch 8 --exp_name 0428debug > results/0428debug_RGBD135.txt &
CUDA_VISIBLE_DEVICES="3" python -u main.py --phase test --test_dataset SIP --begin_epoch 1 --end_epoch 8 --exp_name 0428debug > results/0428debug_SIP.txt &
CUDA_VISIBLE_DEVICES="3" python -u main.py --phase test --test_dataset ReDWeb --begin_epoch 1 --end_epoch 8 --exp_name 0428debug > results/0428debug_ReDWeb.txt &