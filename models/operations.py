import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'conv_3x3' :  lambda C, stride, affine : ReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'conv_1x1' :  lambda C, stride, affine : ReLUConvBN(C, C, 1, stride, 0, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_3x3_2dil' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_3x3_4dil' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 4, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
   nn.ReLU(inplace=False),
   nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
   nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
   nn.BatchNorm2d(C, affine=affine)
   ),
  'spatial_attention': lambda C, stride, affine : SpatialAttentionLayer(C, C, 8, stride, affine),
  'channel_attention': lambda C, stride, affine : ChannelAttentionLayer(C, C, 8, stride, affine)
}

class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)



class SpatialAttentionLayer(nn.Module):
    def __init__(self, C_in, C_out, reduction=16, stride=1, affine=True, BN=nn.BatchNorm2d):
        super(SpatialAttentionLayer, self).__init__()
        self.stride = stride
        if stride == 1:
            self.fc = nn.Sequential(
                nn.Conv2d(C_in, C_in // reduction, kernel_size=3, stride=1, padding=1, bias=False),
                BN(C_in // reduction, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in // reduction, 1,kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
                )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(C_in, C_in // reduction, kernel_size=3, stride=2, padding=1, bias=False),
                BN(C_in // reduction, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in // reduction, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
                )
            self.reduce_map = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
                BN(C_out, affine=affine)     
                )
            
    def forward(self, x):
        y = self.fc(x)
        if self.stride == 2:
            x = self.reduce_map(x)
        return x * y

        
## Channel Attention (CA) Layer
class ChannelAttentionLayer(nn.Module):
    def __init__(self, C_in, C_out, reduction=16, stride=1, affine=True, BN=nn.BatchNorm2d):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.stride = stride
        # feature channel downscale and upscale --> channel weight
        if stride == 1:
            self.conv_du = nn.Sequential(
                    nn.Conv2d(C_in, C_in // reduction, 1, padding=0, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in // reduction, C_out, 1, padding=0, bias=False),
                    nn.Sigmoid()
            )
        else:
            self.conv_du = nn.Sequential(
                    nn.Conv2d(C_in, C_in // reduction, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in // reduction, C_out, 1, padding=0, bias=False),
                    nn.Sigmoid()
            )
            self.reduce_map = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
                BN(C_out, affine=affine)     
            )

    def forward(self, x):
        if self.stride == 2:
            x = self.reduce_map(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ReLUConvBN(nn.Module):
  """
  ReLu -> Conv2d -> BatchNorm2d
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
  """
  Dilation Convolution ： ReLU -> DilConv -> Conv2d -> BatchNorm2d
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.) # N * C * W * H


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
    out = self.bn(out)
    return out
