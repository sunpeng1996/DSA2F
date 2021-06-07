from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from operations import *
import genotypes
from genotypes import attention_snas_3_4_1
from genotypes import PRIMITIVES
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

# Take four inputs
class FusionCell(nn.Module):
  def __init__(self, genotype, index, steps, multiplier, parse_method):
    super(FusionCell, self).__init__()
    
    self.index = index
    if self.index == 0:
        op_names, indices = zip(*genotype.fusion1)
        concat = genotype.fusion1_concat
        C = 128 #128 // 2 # Fusion Scale 64x64
        # two rgb feats (64x64 128c, 32x32s 256c)
        # two depth feats (64x64 128c, 64x64 32c)
        self.preprocess0_rgb = nn.Sequential(
            nn.Conv2d(128, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        self.preprocess1_rgb = nn.Sequential(
            nn.Conv2d(256, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.preprocess0_depth = nn.Sequential(
            nn.Conv2d(128, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        self.preprocess1_depth = nn.Sequential(
            nn.Conv2d(32, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    elif self.index == 1:
        op_names, indices = zip(*genotype.fusion2)
        concat = genotype.fusion2_concat
        C = 128 #128 // 2 # Fusion Scale 64x64
        # two rgb feats (32x32 256c, 16x16 512c)
        # two depth feats (64x64 32c, 64x64 32c)
        self.preprocess0_rgb = nn.Sequential(
            nn.Conv2d(256, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.preprocess1_rgb = nn.Sequential(
            nn.Conv2d(512, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.preprocess0_depth = nn.Sequential(
            nn.Conv2d(32, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        self.preprocess1_depth = nn.Sequential(
            nn.Conv2d(32, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    else:
        op_names, indices = zip(*genotype.fusion3)
        concat = genotype.fusion3_concat
        C = 128 #256 // 2 # Fusion Scale 32x32
        # two rgb feats (16x16 512c, 8x8 512c)
        # two depth feats (64x64 32c, 64x64 128c)
        self.preprocess0_rgb = nn.Sequential(
            nn.Conv2d(512, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.preprocess1_rgb = nn.Sequential(
            nn.Conv2d(512, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.preprocess0_depth = nn.Sequential(
            nn.Conv2d(32, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        self.preprocess1_depth = nn.Sequential(
            nn.Conv2d(128, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        
    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)
        
  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, s2, s3, drop_prob):

    # print("s_input:",s0.shape, s1.shape, s2.shape, s3.shape)
    s0 = self.preprocess0_rgb(s0)
    s1 = self.preprocess1_rgb(s1)
    s2 = self.preprocess0_depth(s2)
    s3 = self.preprocess1_depth(s3)

    # print("s_prepoce:",s0.shape, s1.shape, s2.shape, s3.shape)
    states = [s0, s1, s2, s3]
    for i in range(self._steps):
        h1 = states[self._indices[4*i]]
        h2 = states[self._indices[4*i+1]]
        h3 = states[self._indices[4*i+2]]
        h4 = states[self._indices[4*i+3]]
        op1 = self._ops[4*i]
        op2 = self._ops[4*i+1]
        op3 = self._ops[4*i+2]
        op4 = self._ops[4*i+3]
        h1 = op1(h1)
        h2 = op2(h2)
        h3 = op3(h3)
        h4 = op4(h4)
        if self.training and drop_prob > 0.:
            if not isinstance(op1, Identity):
              h1 = drop_path(h1, drop_prob)
            if not isinstance(op2, Identity):
              h2 = drop_path(h2, drop_prob)
            if not isinstance(op3, Identity):
              h3 = drop_path(h3, drop_prob)
            if not isinstance(op4, Identity):
              h4 = drop_path(h4, drop_prob)
        # print("h:",h1.shape, h2.shape, h3.shape, h4.shape)
        s = h1 + h2 + h3 + h4
        states += [s]

    return torch.cat([states[i] for i in self._concat], dim=1) # N，C，H, W

# Take three inputs
class AggregationCell(nn.Module):
  def __init__(self, genotype, steps, multiplier, parse_method):
    super(AggregationCell, self).__init__()
    C = 128
    self.preprocess0 = None
    self.preprocess1 = None
    self.preprocess2 = None
        
    op_names, indices = zip(*genotype.aggregation)
    concat = genotype.aggregation_concat
    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)
        
  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, s2, drop_prob):
    #  print("000:",s0.shape, s1.shape, s2.shape)
     s0 = self.preprocess0(s0)
     s1 = self.preprocess1(s1)
     s2 = self.preprocess2(s2)
    #  print("111:",s0.shape, s1.shape, s2.shape)
    
     states = [s0, s1, s2]
     for i in range(self._steps):
         h1 = states[self._indices[3*i]]
         h2 = states[self._indices[3*i+1]]
         h3 = states[self._indices[3*i+2]]
         op1 = self._ops[3*i]
         op2 = self._ops[3*i+1]
         op3 = self._ops[3*i+2]
         h1 = op1(h1)
         h2 = op2(h2)
         h3 = op3(h3)
         if self.training and drop_prob > 0.:
             if not isinstance(op1, Identity):
                 h1 = drop_path(h1, drop_prob)
             if not isinstance(op2, Identity):
                 h2 = drop_path(h2, drop_prob)
             if not isinstance(op3, Identity):
                 h3 = drop_path(h3, drop_prob)
         s = h1 + h2 + h3
         states += [s]
     return torch.cat([states[i] for i in self._concat], dim=1) # N，C，H, W

class AggregationCell_1(AggregationCell):
  def __init__(self, genotype, steps, multiplier, parse_method, C_in = [768,768,768]):
    super().__init__(genotype, steps, multiplier, parse_method)
    C = 128
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True)
            )
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            )
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

class AggregationCell_2(AggregationCell):
  def __init__(self, genotype, steps, multiplier, parse_method, C_in = [512,128,768]):
    super().__init__(genotype, steps, multiplier, parse_method)
    C = 128
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[1], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[2], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))

class AggregationCell_3(AggregationCell):
  def __init__(self, genotype, steps, multiplier, parse_method, C_in = [256,32,768]):
    super().__init__(genotype, steps, multiplier, parse_method)
    C = 128
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[1], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            )
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[2], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

class AggregationCell_4(AggregationCell):
  def __init__(self, genotype, steps, multiplier, parse_method, C_in = [512,32,768]):
    super().__init__(genotype, steps, multiplier, parse_method)
    C = 128
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[0], C*2, kernel_size=1, bias=False),
            nn.BatchNorm2d(C*2, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(C*2, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[1], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            )
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[2], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))

# Take foul inputs
class GlobalAggregationCell(nn.Module):
  def __init__(self, genotype, steps, multiplier, parse_method):
    super(GlobalAggregationCell, self).__init__()
    C = 256
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(768, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(768, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(768, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    self.preprocess3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(768, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
        
    op_names, indices = zip(*genotype.final_agg)
    concat = genotype.final_aggregation_concat
    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)
        
  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, s2, s3, drop_prob):
     s0 = self.preprocess0(s0)
     s1 = self.preprocess1(s1)
     s2 = self.preprocess2(s2)
     s3 = self.preprocess3(s3)
    
     states = [s0, s1, s2, s3]
     for i in range(self._steps):
         h1 = states[self._indices[4*i]]
         h2 = states[self._indices[4*i+1]]
         h3 = states[self._indices[4*i+2]]
         h4 = states[self._indices[4*i+3]]
         op1 = self._ops[4*i]
         op2 = self._ops[4*i+1]
         op3 = self._ops[4*i+2]
         op4 = self._ops[4*i+3]
         h1 = op1(h1)
         h2 = op2(h2)
         h3 = op3(h3)
         h4 = op4(h4)
         if self.training and drop_prob > 0.:
             if not isinstance(op1, Identity):
                 h1 = drop_path(h1, drop_prob)
             if not isinstance(op2, Identity):
                 h2 = drop_path(h2, drop_prob)
             if not isinstance(op3, Identity):
                 h3 = drop_path(h3, drop_prob)
             if not isinstance(op4, Identity):
                 h4 = drop_path(h4, drop_prob)
         s = h1 + h2 + h3 + h4
         states += [s]
     return torch.cat([states[i] for i in self._concat], dim=1) # N，C，H, W

# Take three inputs
class Low_High_aggregation(AggregationCell):
  
  def __init__(self, genotype, steps, multiplier, parse_method, C_in=[64,64,128]):
    super().__init__(genotype, steps, multiplier, parse_method)
    C = 32
    self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[0], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True)
            )
    self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[1], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True),
            )
    self.preprocess2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in[2], C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, affine=True))
    op_names, indices = zip(*genotype.low_high_agg)
    concat = genotype.low_high_agg_concat
    self._compile(C, op_names, indices, concat)



class NasFusionNet(nn.Module):
    def __init__(self, fusion_cell_number=3, steps=8, multiplier=6, agg_steps=8, agg_multiplier=6, genotype
                 =attention_snas_3_4_1, parse_method='darts', op_threshold=0.85, drop_path_prob=0):
        self.inplanes = 64
        super(NasFusionNet, self).__init__()
        self.drop_path_prob = 0
        self._multiplier = 6
        self.parse_method = parse_method
        self.op_threshold = op_threshold
        self._steps = steps
        # init the fusion cells
        self.MM_cells = nn.ModuleList()

        for i in range(fusion_cell_number):
          cell = FusionCell(genotype, i, steps, multiplier, parse_method)
          self.MM_cells += [cell]

        self.MS_cell_1 = AggregationCell_1(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_2 = AggregationCell_2(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_3 = AggregationCell_3(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_4 = AggregationCell_4(genotype, agg_steps, agg_multiplier, parse_method)
    
        self.GA_cell = GlobalAggregationCell(genotype, agg_steps, agg_multiplier, parse_method)
        self.SR_cell_1 = Low_High_aggregation(genotype, 4, 4, parse_method, C_in = [128,128,256])
        self.SR_cell_2 = Low_High_aggregation(genotype, 4, 4, parse_method, C_in = [64, 64, 128])

        self.final_layer0 = nn.Sequential(
          nn.Conv2d(1536, 512, kernel_size=1), nn.BatchNorm2d(512, affine=True), nn.ReLU(inplace=True), #  256
          nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256, affine=True), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256, affine=True), nn.ReLU(inplace=True),
        )

        self.final_layer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256+128, 256, kernel_size=1), nn.BatchNorm2d(256, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128, affine=True), nn.ReLU(inplace=True)
        )

        self.final_layer2 = nn.Sequential(
            nn.Conv2d(128+128, 64, kernel_size=1), nn.BatchNorm2d(64, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, affine=True), nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 2, kernel_size=1)
        )


    def forward(self, h1, h2, h3, h4, h5, d0, d1, d2, d3, d4):
        # print(h2.shape,d1.shape, h5.shape,d4.shape)
        
        output1 = self.MM_cells[0](h2, h3, d1, d2, self.drop_path_prob)
        output2 = self.MM_cells[1](h3, h4, d2, d3, self.drop_path_prob)
        output3 = self.MM_cells[2](h4, h5, d3, d4, self.drop_path_prob)

        agg_features1 = self.MS_cell_1(output1, output2, output3,self.drop_path_prob)
        agg_features2 = self.MS_cell_2(h5, d4, output2, self.drop_path_prob)
        agg_features3 = self.MS_cell_3(h3, d2, output3, self.drop_path_prob)
        agg_features4 = self.MS_cell_4(h4, d3, output1, self.drop_path_prob)

        agg_features = self.GA_cell(agg_features1, agg_features2, agg_features3, agg_features4, self.drop_path_prob)
        predict_mask = self.final_layer0(agg_features) #  c=256

        low_high_combined1 = self.SR_cell_1(h2, d1, predict_mask, self.drop_path_prob) # c==128 
        predict_mask = torch.cat([predict_mask, low_high_combined1], dim=1) # 256 + 128

        predict_mask = F.upsample(predict_mask, scale_factor=2, mode='bilinear', align_corners=True)
        predict_mask = self.final_layer1(predict_mask)   # 128

        low_high_combined2 = self.SR_cell_2(h1, d0, predict_mask, self.drop_path_prob) # 128
        predict_mask = torch.cat([predict_mask, low_high_combined2], dim=1)
        predict_mask = F.upsample(predict_mask, scale_factor=2, mode='bilinear', align_corners=True)
        predict_mask = self.final_layer2(predict_mask)

        return F.sigmoid(predict_mask)

    def init_weights(self):
        logger.info('=> NAS Fusion model init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)




class NasFusionNet_pre(nn.Module):
    def __init__(self, fusion_cell_number=3, steps=8, multiplier=6, agg_steps=8, agg_multiplier=6, genotype
                 =attention_snas_3_4_1, parse_method='darts', op_threshold=0.85, drop_path_prob=0):
        self.inplanes = 64
        super(NasFusionNet_pre, self).__init__()
        self.drop_path_prob = 0
        self._multiplier = 6
        self.parse_method = parse_method
        self.op_threshold = op_threshold
        self._steps = steps
        # init the fusion cells
        self.MM_cells = nn.ModuleList()

        for i in range(fusion_cell_number):
          cell = FusionCell(genotype, i, steps, multiplier, parse_method)
          self.MM_cells += [cell]

        self.MS_cell_1 = AggregationCell_1(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_2 = AggregationCell_2(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_3 = AggregationCell_3(genotype, agg_steps, agg_multiplier, parse_method)
        self.MS_cell_4 = AggregationCell_4(genotype, agg_steps, agg_multiplier, parse_method)
    
        self.GA_cell = GlobalAggregationCell(genotype, agg_steps, agg_multiplier, parse_method)
        ######## for pretrain
        self.class_head = nn.Sequential(
            nn.AvgPool2d((56, 56)))
        self.classifier = nn.Linear(1536, 1000)


    def forward(self, h1, h2, h3, h4, h5, d0, d1, d2, d3, d4):
        
        output1 = self.MM_cells[0](h2, h3, d1, d2, self.drop_path_prob)
        output2 = self.MM_cells[1](h3, h4, d2, d3, self.drop_path_prob)
        output3 = self.MM_cells[2](h4, h5, d3, d4, self.drop_path_prob)

        agg_features1 = self.MS_cell_1(output1, output2, output3,self.drop_path_prob)
        agg_features2 = self.MS_cell_2(h5, d4, output2, self.drop_path_prob)
        agg_features3 = self.MS_cell_3(h3, d2, output3, self.drop_path_prob)
        agg_features4 = self.MS_cell_4(h4, d3, output1, self.drop_path_prob)

        agg_features = self.GA_cell(agg_features1, agg_features2, agg_features3, agg_features4, self.drop_path_prob)

        ######## for pretrain
        # print(agg_features.shape)
        class_feature = self.class_head(agg_features).view(agg_features.size(0), -1)
        logits = self.classifier(class_feature)
        return logits

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

