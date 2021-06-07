"""
    DSA^2 F: Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion
"""
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torchvision
import logging
import sys
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from dataset_loader import MyData, MyTestData
from utils.functions import *
from training import Trainer
from utils.evaluateFM import get_FM
from models.model_depth import DepthNet
from models.model_rgb import RgbNet
from models.model_fusion import NasFusionNet
import warnings
warnings.filterwarnings("ignore")

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=5e-9,
        momentum=0.9,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=100,
    ),
}
parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
parser.add_argument('--train_dataroot', type=str, default='/4T/wenhu/dataset/SOD-RGBD/train_data-augment', help=
                                                          'path to train data')
parser.add_argument('--test_dataroot', type=str, default='/4T/wenhu/dataset/SOD-RGBD/val/', help=
                                                      'path to test data')
parser.add_argument('--pretrain_path', type=str, default='')

parser.add_argument('--exp_name', type=str, default='debug', help='save & log path')
parser.add_argument('--save_path', type=str, default='/4T/wenhu/pami21/', help='save & log path')
parser.add_argument('--snapshot_root', type=str, default='None', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='None', help='path to saliency map')
parser.add_argument('--log_root', type=str, default='path to logs')

parser.add_argument('--test_dataset', type=str, default='LFSD')
parser.add_argument('--begin_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=0)
parser.add_argument('--parse_method', type=str, default='darts', help='parse the code method')

parser.add_argument('--batchsize', type=int, default=2, help='batchsize')
parser.add_argument('--epoch', type=int, default=60, help='epoch')
parser.add_argument("--local_rank", default=-1)
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations



args.save_path = args.save_path + args.exp_name
if args.phase =='train':
    if os.path.exists(args.save_path):
        print(".... error!!!!!!!!!! save path already exist .....")
        logging.info(".... error!!!!!!!!!! save path already exist .....")
        sys.exit()
    else :
        os.mkdir(args.save_path)

args.snapshot_root = args.save_path +'/snapshot/'
args.salmap_root = args.save_path + '/sal_map/'
args.log_root = args.save_path + '/logs/'
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)

cuda = torch.cuda.is_available

if args.phase =='train':
    create_exp_dir(args.log_root, scripts_to_save=None)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log_root, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


"""""""""""dataset loader"""""""""

train_dataRoot = args.train_dataroot

if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)

if args.phase == 'train':
    SnapRoot = args.snapshot_root           # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size = args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
else:
    test_dataRoot = args.test_dataroot +args.test_dataset
    max_F_dict = {}
    min_mae_dict = {}
    MapRoot = args.salmap_root +args.test_dataset
    if not os.path.exists(MapRoot):
        os.mkdir(MapRoot)
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
print ('data already')

"""""""""""train_data/test_data through nets"""""""""
cuda = torch.cuda.is_available
start_epoch = 0
start_iteration = 0

model_depth = DepthNet()
model_rgb = RgbNet()
model_fusion = NasFusionNet()

print("model_rgb param size = %fMB", count_parameters_in_MB(model_rgb))
print("model_depth param size = %fMB", count_parameters_in_MB(model_depth))
print("nas-model param size = %fMB", count_parameters_in_MB(model_fusion))

if args.begin_epoch == args.end_epoch:
    test_check_list = [args.end_epoch]
else:
    test_epoch_list = [i*16418 for i in range(1,61)]
    test_iter_list = [i*10000 for i in range(args.begin_epoch, args.end_epoch+1)]
    test_check_list = test_epoch_list + test_iter_list
    test_check_list.sort()
for ckpt_i in test_check_list:   # When training, remove this line.ssss
    best_F = -float('inf')
    best_mae = float('inf')

    if args.phase == 'test':
        ckpt = str(ckpt_i)
        print(".... load checkpoint "+ ckpt +" for test .....")
        model_depth.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'depth_snapshot_iter_' + ckpt + '.pth')))
        model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'rgb_snapshot_iter_'+ckpt+'.pth')))
        model_fusion.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'fusion_snapshot_iter_'+ckpt+'.pth')))
    
    elif (args.pretrain_path):
        pretrained_dict = load_pretrain(args.pretrain_path, model_depth.state_dict(), "model_depth.")
        model_depth.load_state_dict(pretrained_dict) 

        pretrained_dict = load_pretrain(args.pretrain_path, model_rgb.state_dict(), "model_rgb.")
        model_rgb.load_state_dict(pretrained_dict) 

        model_fusion.init_weights()
        pretrained_dict = load_pretrain(args.pretrain_path, model_fusion.state_dict(), "model_fusion.")
        model_fusion.load_state_dict(pretrained_dict) 
        logging.info(".... load imagenet pretrain models .....")
    
    else:
        logging.info(".... norm init .....")
        model_depth.init_weights()
        vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
        model_rgb.copy_params_from_vgg19_bn(vgg19_bn)
        model_fusion.init_weights()
    
    if cuda:
        model_depth = model_depth.cuda()
        model_rgb =  model_rgb.cuda()
        model_fusion =  model_fusion.cuda()

    if args.phase == 'train':
        cudnn.benchmark = True
        # torch.manual_seed(444)
        cudnn.enabled=True
        # torch.cuda.manual_seed(444)
        writer = SummaryWriter(args.log_root)
        model_rgb.cuda()
        model_depth.cuda()
        model_fusion.cuda()
        
        training = Trainer(
            cuda=cuda,
            cfg=cfg,
            model_depth=model_depth,
            model_rgb=model_rgb,
            model_fusion=model_fusion,
            train_loader=train_loader,
            test_data_list = ["DUT-RGBD","NJUD","NLPR","SSD","STEREO","LFSD","RGBD135","SIP","ReDWeb"],
            test_data_root = args.test_dataroot,
            salmap_root = args.salmap_root,
            outpath=args.snapshot_root,
            logging=logging,
            writer=writer,
            max_epoch=args.epoch,
        )
        training.epoch = start_epoch
        training.iteration = start_iteration
        training.train()
    else:
        # -------------------------- inference --------------------------- #
        res = []
        for id, (data, depth, bins, img_name, img_size) in enumerate(test_loader):
            # print('testing bach %d' % id)
            inputs = Variable(data).cuda()
            depth = Variable(depth).cuda()
            bins = Variable(bins).cuda()
            n, c, h, w = inputs.size()
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                h1, h2, h3, h4, h5 = model_rgb(inputs, bins, gumbel=False)
                d0, d1, d2, d3, d4 = model_depth(depth)
                predict_mask = model_fusion(h1, h2, h3, h4, h5, d0, d1, d2, d3, d4)
            torch.cuda.synchronize()
            end = time.time()

            res.append(end - start)
            outputs_all = F.softmax(predict_mask, dim=1)
            outputs = outputs_all[0][1]
            outputs = outputs.cpu().data.resize_(h, w)

            imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)
        time_sum = 0
        for i in res:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(res))))
        # -------------------------- validation --------------------------- #
        torch.cuda.empty_cache()
        print('the testing process has finished!')
        F_measure, mae = get_FM(salpath=MapRoot+'/', gtpath=test_dataRoot+'/test_masks/')
        print(args.test_dataset + ' F_measure:', F_measure)
        print(args.test_dataset + ' MAE:', mae)
    
        F_key = args.test_dataset +'_Fb'
        M_key = args.test_dataset +'_mae'
        ckpt_key = args.test_dataset +'_ckpt'
        if F_key in max_F_dict.keys():
            if F_measure > max_F_dict[F_key]:
                max_F_dict[F_key] = F_measure
                max_F_dict[M_key] = mae
                max_F_dict[ckpt_key] = ckpt 
        else:
            max_F_dict[F_key] = F_measure
            max_F_dict[M_key] = mae
            max_F_dict[ckpt_key] = ckpt 
                
        if M_key in min_mae_dict.keys():
            if mae < min_mae_dict[M_key]:
                min_mae_dict[F_key] = F_measure
                min_mae_dict[M_key] = mae
                min_mae_dict[ckpt_key] = ckpt 
        else:
            min_mae_dict[F_key] = F_measure
            min_mae_dict[M_key] = mae
            min_mae_dict[ckpt_key] = ckpt 

        if args.phase == 'test': 
            print ("max_F_dict")
            print (max_F_dict)
            print ("min_mae_dict")
            print (min_mae_dict)

print("finish!!!!!!!!")

