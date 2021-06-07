"""
    DSA^2 F: Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion
"""
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
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
from utils.functions import *
from models.model_depth import DepthNet
from models.model_rgb import RgbNet
from models.model_fusion import NasFusionNet_pre
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr
    if epoch >= 30:
        lr = 0.1 * lr
    if epoch >= 60:
        lr = 0.1 * lr
    if epoch >= 80:
        lr = 0.1 * lr
    optimizer.param_groups[0]['lr'] = lr


def train(train_loader, models, CE, optimizers, epoch, logger, logging):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    for m in models:
        m.train()
    end = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        target = target.cuda()
        inputs = inputs.cuda()

        # print(gpu,models[0].device_ids, inputs.device)
        b,c,h,w = inputs.size()
        depth = torch.mean(inputs,dim = 1).view(b,1,h,w).repeat(1, c, 1, 1)
        # print("inpus:",inputs.shape)
        h1, h2, h3, h4, h5 = models[0](inputs, depth, gumbel=True)
        d0, d1, d2, d3, d4 = models[1](depth)
        output = models[2](h1, h2, h3, h4, h5, d0, d1, d2, d3, d4)

        # A loss
        loss = CE( output, target) * 1.0
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), inputs.size(0))
        top1.update(to_python_float(prec1), inputs.size(0))
        top5.update(to_python_float(prec5), inputs.size(0))


        # compute gradient and do SGD step
        for op in optimizers:
            op.zero_grad()

        loss.backward()

        for op in optimizers:
            op.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 and args.rank ==0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} \t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top5 = top5))
            
            logger.add_scalar('train/losses', losses.avg, global_step=global_step)
            logger.add_scalar('train/top1', top1.avg, global_step=global_step)
            logger.add_scalar('train/top5', top5.avg, global_step=global_step)
            logger.add_scalar('train/lr', optimizers[0].param_groups[0]['lr'], global_step=global_step)
       

def validate(valid_loader, models, CE, epoch, logger, logging):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for m in models:
        m.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(valid_loader):
        target = target.cuda()
        inputs = inputs.cuda()
        with torch.no_grad():
            b,c,h,w = inputs.size()
            depth = torch.mean(inputs,dim = 1).view(b,1,h,w).repeat(1, c, 1, 1)

            h1, h2, h3, h4, h5 = models[0](inputs, depth, gumbel=False)
            d0, d1, d2, d3, d4 = models[1](depth)
            output = models[2](h1, h2, h3, h4, h5, d0, d1, d2, d3, d4)

            loss = CE(output, target)

        # measure accuracy and record loss
        prec1 , prec5 = accuracy(output.data, target, topk=(1,5))

        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), inputs.size(0))
        top1.update(to_python_float(prec1), inputs.size(0))
        top5.update(to_python_float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 and args.rank == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(valid_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5 = top5))
       

    logger.add_scalar('valid/top1', top1.avg, global_step=epoch)
    logger.add_scalar('valid/top5', top5.avg, global_step=epoch)

    logging.info(' * Prec@1 {top1.avg:.3f} * Prec@5 {top5.avg:.3f}   '.format(top1=top1, top5=top5))

    return top1.avg



def main_worker(gpu, argss):
    global args
    args = argss

    torch.cuda.set_device(gpu)
    rank = args.nr * args.gpus + gpu
    args.rank = rank
    exp_name = '/imagenet_pretrain'
    args.save_path = args.save_path + exp_name
    args.snapshot_root = args.save_path +'/snapshot/'
    args.log_root = args.save_path + '/logs/train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))

    if args.phase =='train' and args.rank ==0 :
        create_exp_dir(args.log_root, scripts_to_save=None)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.log_root, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
    
    if not os.path.exists(args.snapshot_root) and args.rank ==0 :
        os.mkdir(args.snapshot_root)

    dist.init_process_group(
        backend='nccl',    
        init_method=args.dist_url, 
        world_size=args.world_size, 
        rank=args.rank)

    
    """""""""""dataset loader"""""""""
    # ImageNet Data loading code
    train_dataset = datasets.ImageFolder(
            os.path.join(args.data_root, 'train'),
            transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
            ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
        num_replicas=args.world_size,
    	rank=rank,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batchsize,
        num_workers=0, pin_memory=True, sampler = train_sampler)


    valid_dataset = datasets.ImageFolder(
            os.path.join(args.data_root, 'val'), 
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
    	valid_dataset,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = args.batchsize, 
        num_workers=0, pin_memory=True, sampler = valid_sampler)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    logging.info('data already')

    """""""""""train_data/test_data through nets"""""""""

    model_depth = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DepthNet())
    model_rgb = torch.nn.SyncBatchNorm.convert_sync_batchnorm(RgbNet())
    model_fusion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(NasFusionNet_pre())

    model_depth.init_weights()
    vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
    model_rgb.copy_params_from_vgg19_bn(vgg19_bn)
    model_fusion.init_weights()

    if args.rank==0:        
        print("model_rgb param size = %fMB", count_parameters_in_MB(model_rgb))
        print("model_depth param size = %fMB", count_parameters_in_MB(model_depth))
        print("nas-model param size = %fMB", count_parameters_in_MB(model_fusion))

    model_depth = model_depth.cuda()
    model_rgb = model_rgb.cuda()
    model_fusion = model_fusion.cuda()

    if args.distributed:
        model_depth = DDP(model_depth, device_ids=[gpu])
        model_rgb = DDP(model_rgb, device_ids=[gpu])
        model_fusion = DDP(model_fusion, device_ids=[gpu])

    optimizer_depth = optim.SGD(model_depth.parameters(), lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_rgb = optim.SGD(model_rgb.parameters(), lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_fusion = optim.SGD(model_fusion.parameters(), lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # print(optimizer_depth.param_groups[0]['lr'])
    CE = nn.CrossEntropyLoss().cuda()

    logger = SummaryWriter(args.log_root)

    best_prec1 = -1
    for epoch in range(0, args.epoch):

        adjust_learning_rate(optimizer_depth, epoch, args)
        adjust_learning_rate(optimizer_rgb, epoch, args)
        adjust_learning_rate(optimizer_fusion, epoch, args)
        if args.rank==0:  
            print("lr:",optimizer_rgb.param_groups[0]['lr'])
        # train for one epoch
        train_sampler.set_epoch(epoch)
        train(train_loader, [model_rgb, model_depth, model_fusion], CE, [optimizer_rgb, optimizer_depth, optimizer_fusion], epoch, logger, logging)

        # evaluate on validation set
        prec1 = validate(valid_loader, [model_rgb, model_depth, model_fusion], CE, epoch, logger, logging)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if args.rank ==0:
            logging.info('Best accuracy: %f' % best_prec1)
            logger.add_scalar('best/accuracy', best_prec1, global_step=epoch)

            savename_depth = ('%s/depth_pre_epoch%d.pth' % (args.snapshot_root, epoch))
            torch.save(model_depth.state_dict(), savename_depth)
            print('save: (snapshot: %d)' % (epoch))

            savename_rgb = ('%s/rgb_pre_epoch%d.pth' % (args.snapshot_root, epoch))
            torch.save(model_rgb.state_dict(), savename_rgb)
            print('save: (snapshot: %d)' % (epoch))

            savename_fusion = ('%s/fusion_pre_epoch%d.pth' % (args.snapshot_root, epoch))
            torch.save(model_fusion.state_dict(), savename_fusion)
            print('save: (snapshot: %d)' % (epoch))

            if is_best:
                savename_depth = ('%s/depth_pre.pth' % (args.snapshot_root))
                torch.save(model_depth.state_dict(), savename_depth)
                print('save: (snapshot: %d)' % (epoch))

                savename_rgb = ('%s/rgb_pre.pth' % (args.snapshot_root))
                torch.save(model_rgb.state_dict(), savename_rgb)
                print('save: (snapshot: %d)' % (epoch))

                savename_fusion = ('%s/fusion_pre.pth' % (args.snapshot_root))
                torch.save(model_fusion.state_dict(), savename_fusion)
                print('save: (snapshot: %d)' % (epoch))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
    parser.add_argument('--data_root', type=str, default='/4T/sunpeng/ImageNet')

    parser.add_argument('--save_path', type=str, default='/home/wenhu/pami21/runs/', help='save & log path')
    parser.add_argument('--snapshot_root', type=str, default='None', help='path to snapshot')
    parser.add_argument('--log_root', type=str, default='path to logs')

    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('--parse_method', type=str, default='darts', help='parse the code method')

    parser.add_argument('--batchsize', type=int, default=2, help='batchsize')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('-n', '--nodes', default=1,
                            type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()

    args.distributed = True


    args.world_size = args.gpus * args.nodes 
    port = find_free_port()
    args.dist_url = f"tcp://127.0.0.1:{port}"
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))  

if __name__ == '__main__':
    main()