
###
 # @Author: Wenhu Zhang
 # @Date: 2021-06-07 17:39:22
 # @LastEditTime: 2021-06-07 18:09:35
 # @LastEditors: Wenhu Zhang
 # @Description: 
 # @FilePath: /github/wh/DSA2F/launch_train.sh
### 
CUDA_VISIBLE_DEVICES="0" python main.py --phase train --epoch 60 \
--save_path /4T/wenhu/pami21/ \
--pretrain_path /home/wenhu/pami21/ckpt_best.pth.tar \
--train_dataroot /4T/wenhu/dataset/SOD-RGBD/train_data-augment/ \
--test_dataroot /4T/wenhu/dataset/SOD-RGBD/val/ \
--exp_name 0607debug