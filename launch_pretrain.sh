

###############  imagenet pretraining
python imagenet_pretrain.py -n 1 -g 4 -nr 0 \
--phase train --epoch 90 --batchsize 4 --lr 0.00625 --momentum 0.9 --weight_decay 5e-4 \
--data_root /4T/sunpeng/ImageNet \
--save_path /4T/wenhu/pami21/github_test