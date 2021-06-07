import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.optim as optim
from dataset_loader import MyTestData
import logging
from tqdm import tqdm
import time
from utils.functions import *
from utils.evaluateFM import get_FM
from loss import cross_entropy2d, iou, BinaryDiceLoss
running_loss_final = 0
iou_final = 0
aux_final = 0

class Trainer(object):

    def __init__(self, cuda, cfg, model_depth, model_rgb, model_fusion, train_loader, test_data_list, test_data_root, salmap_root, outpath, logging, writer, max_epoch):
        self.cuda = cuda
        self.model_depth = model_depth
        self.model_rgb = model_rgb
        self.model_fusion = model_fusion

        self.optim_depth = optim.SGD(self.model_depth.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
        self.optim_rgb = optim.SGD(self.model_rgb.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])
        self.optim_fusion = optim.SGD(self.model_fusion.parameters(), lr=cfg[1]['lr'], momentum=cfg[1]['momentum'], weight_decay=cfg[1]['weight_decay'])

        self.train_loader = train_loader
        self.test_data_list = test_data_list
        self.test_data_root = test_data_root
        self.salmap_root = salmap_root
        self.test_loaders={}
        self.best_f = {} 
        self.best_m = {} 
        for data_i in self.test_data_list:
            MapRoot = self.salmap_root + data_i
            TestRoot = self.test_data_root + data_i
            if not os.path.exists(MapRoot):
                os.mkdir(MapRoot)
            loader_i = torch.utils.data.DataLoader(MyTestData(TestRoot, transform=True),
                                               batch_size = 1, shuffle=True, num_workers=0, pin_memory=True)
            self.test_loaders[data_i] = loader_i
            self.best_f[data_i] = -1
            self.best_m[data_i] = 10000

        self.epoch = 0
        self.iteration = 0
        self.max_iter = 0
        self.snapshot = cfg[1]['spshot']
        self.outpath = outpath
        self.sshow = cfg[1]['sshow']
        self.logging = logging
        self.writer = writer
        self.max_epoch = max_epoch
        self.base_lr = cfg[1]['lr']
        
        self.dice = BinaryDiceLoss()
 

    def train_epoch(self):
        self.logging.info("length trainloader: %s", len(self.train_loader))
        self.logging.info("current_lr is : %s", self.optim_fusion.param_groups[0]['lr'])
        for batch_idx, (img, mask, depth, bins) in enumerate(tqdm(self.train_loader)):
            ########## for debug
            # if batch_idx % 10==0 and batch_idx>10:
            #     self.save_test(iteration)
            iteration = batch_idx + self.epoch * len(self.train_loader)
    
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.cuda:
                img, mask, depth, bins = img.cuda(), mask.cuda(), depth.cuda(), bins.cuda()
                img, mask, depth, bins = Variable(img), Variable(mask), Variable(depth), bins.cuda()
            # print(img.size())
            n, c, h, w = img.size()  # batch_size, channels, height, weight
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)

            self.optim_depth.zero_grad()
            self.optim_rgb.zero_grad()
            self.optim_fusion.zero_grad()

            global running_loss_final ,iou_final, aux_final

            d0, d1, d2, d3, d4 = self.model_depth(depth)
            h1, h2, h3, h4, h5 = self.model_rgb(img, bins, gumbel=True)
            predict_mask = self.model_fusion(h1, h2, h3, h4, h5, d0, d1, d2, d3, d4)
            
            ce_loss = cross_entropy2d(predict_mask, mask, size_average=False)
            iou_loss = torch.zeros(1)
            aux_ce_loss = torch.zeros(1)
            # iou_loss = iou(predict_mask, mask,size_average=False ) * 0.2
            # iou_loss = self.dice(predict_mask, mask)
            loss = ce_loss  #+  iou_loss + aux_ce_loss

            running_loss_final += ce_loss.item()
            iou_final += iou_loss.item()
            aux_final += aux_ce_loss.item()

            if iteration % self.sshow == (self.sshow - 1):
                self.logging.info('\n [%3d, %6d,   RGB-D Net ce_loss: %.3f aux_loss: %.3f  iou_loss: %.3f]' % (
                self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow), aux_final / (n * self.sshow), iou_final / (n * self.sshow)))

                self.writer.add_scalar('train/iou_loss', iou_final / (n * self.sshow), iteration + 1)
                self.writer.add_scalar('train/aux_loss', aux_final / (n * self.sshow), iteration + 1)
                
                self.writer.add_scalar('train/lr', self.optim_fusion.param_groups[0]['lr'] , iteration + 1)
                self.writer.add_scalar('train/iter_ce_loss', running_loss_final / (n * self.sshow), iteration + 1)

                self.writer.add_scalar('train/epoch_ce_loss', running_loss_final / (n * self.sshow), self.epoch + 1)
                running_loss_final = 0.0
                iou_final= 0.0
                aux_final=0.0

            loss.backward()
            self.optim_depth.step()
            self.optim_rgb.step()
            self.optim_fusion.step()
            
            if iteration <= 200000:
                if iteration % self.snapshot == (self.snapshot - 1):
                    self.save_test(iteration)
            else:
                if iteration % 10000 == (10000 - 1):
                    self.save_test(iteration)
                    
    def test(self,iteration, test_data):
        res = []
        MapRoot = self.salmap_root + test_data
        for id, (data, depth, bins, img_name, img_size) in enumerate(self.test_loaders[test_data]):
            # print('testing bach %d' % id)
            inputs = Variable(data).cuda()
            depth = Variable(depth).cuda()
            bins = Variable(bins).cuda()
            n, c, h, w = inputs.size()
            depth = depth.view(n, 1, h, w).repeat(1, c, 1, 1)
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                h1, h2, h3, h4, h5 = self.model_rgb(inputs, bins, gumbel=False)
                d0, d1, d2, d3, d4 = self.model_depth(depth)
                predict_mask = self.model_fusion(h1, h2, h3, h4, h5, d0, d1, d2, d3, d4)
            torch.cuda.synchronize()
            end = time.time()

            res.append(end - start)
            outputs_all = F.softmax(predict_mask, dim=1)
            outputs = outputs_all[0][1]
            # import pdb; pdb.set_trace()
            outputs = outputs.cpu().data.resize_(h, w)

            imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)
        time_sum = 0
        for i in res:
            time_sum += i
        self.logging.info("FPS: %f" % (1.0 / (time_sum / len(res))))
        # -------------------------- validation --------------------------- #
        torch.cuda.empty_cache()
        F_measure, mae = get_FM(salpath=MapRoot+'/', gtpath=self.test_data_root + test_data+'/test_masks/')

        self.writer.add_scalar('test/'+ test_data +'_F_measure', F_measure, iteration +1)
        self.writer.add_scalar('test/'+ test_data +'_MAE', mae, iteration+1)
                
        self.logging.info(MapRoot.split('/')[-1] + ' F_measure: %f' , F_measure)
        self.logging.info(MapRoot.split('/')[-1] + ' MAE: %f', mae)
        print('the testing process has finished!')

        return F_measure, mae


    def save_test(self, iteration, epoch = -1):
        self.save(iteration, epoch)
        for data_i in self.test_data_list:
            f, m = self.test(iteration, data_i)

            self.best_f[data_i] = max(f, self.best_f[data_i])
            self.best_m[data_i] = min(m, self.best_m[data_i])
            self.writer.add_scalar('best/'+ data_i +'_MAE', self.best_m[data_i], iteration)
            self.writer.add_scalar('best/'+ data_i +'_Fmeasure', self.best_f[data_i], iteration)

    def save(self, iteration=-1, epoch=-1):
        savename_depth = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
        torch.save(self.model_depth.state_dict(), savename_depth)
        self.logging.info('save: (snapshot: %d)' % (iteration + 1))

        savename_rgb = ('%s/rgb_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
        torch.save(self.model_rgb.state_dict(), savename_rgb)
        self.logging.info('save: (snapshot: %d)' % (iteration + 1))

        savename_fusion = ('%s/fusion_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
        torch.save(self.model_fusion.state_dict(), savename_fusion)
        self.logging.info('save: (snapshot: %d)' % (iteration + 1))
        

        if epoch > 0 :
            savename_depth = ('%s/depth_snapshot_epoch_%d.pth' % (self.outpath, epoch + 1))
            torch.save(self.model_depth.state_dict(), savename_depth)
            self.logging.info('save: (snapshot: %d)' % (self.epoch + 1))

            savename_rgb = ('%s/rgb_snapshot_epoch_%d.pth' % (self.outpath, epoch + 1))
            torch.save(self.model_rgb.state_dict(), savename_rgb)
            self.logging.info('save: (snapshot: %d)' % (self.epoch + 1))

            savename_fusion = ('%s/fusion_snapshot_epoch_%d.pth' % (self.outpath, epoch + 1))
            torch.save(self.model_fusion.state_dict(), savename_fusion)
            self.logging.info('save: (snapshot: %d)' % (self.epoch + 1))
            
        

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
        lr = self.base_lr
        if epoch >= 20:
            lr = 0.1 * lr
        if epoch >= 40:
            lr = 0.1 * lr

        self.optim_depth.param_groups[0]['lr']= lr
        self.optim_rgb.param_groups[0]['lr']= lr
        self.optim_fusion.param_groups[0]['lr']= lr

    def train(self):
        max_epoch = self.max_epoch
        print ("max_epoch", max_epoch)
        self.max_iter = int(math.ceil(len(self.train_loader) * self.max_epoch))
        print ("max_iter", self.max_iter)
        
        for epoch in range(max_epoch):
            # self.adjust_learning_rate(epoch)
            self.epoch = epoch
            self.train_epoch()
            #  save each epoch.
            self.save_test(self.iteration, epoch = self.epoch )

        self.logging.info('all training process finished')
        print(self.best_f) 
        print(self.best_m)

