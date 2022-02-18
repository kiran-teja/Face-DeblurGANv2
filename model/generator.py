from typing_extensions import Self
from webbrowser import get
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pretrainedmodels import inceptionresnetv2



def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def get_norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


## Channel Attention (CA) Layer
class SEModule(nn.Module):
    def __init__(self, n_feats=64, reduction=16):
        super(SEModule, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attention = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_attention(y)
        return x * y
    

class ResBlock(nn.Module):
    """ ResNet Block composed of 2 conv blocks"""
    def __init__(self, 
                 n_feats=64, 
                 norm_type=None, 
                 act_type='leakyrelu', 
                 use_channel_attention=True,
                 res_scale=1
                 ):
        super(ResBlock, self).__init__()

        blocks = []
        for i in range(2):
            blocks.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))
            if norm_type:
                blocks.append(get_norm(norm_type, n_feats))
            if act_type and i==0:
                blocks.append(get_act(act_type))        
        if use_channel_attention:
            blocks.append(SEModule(n_feats))
        self.blocks = nn.Sequential(*blocks) 
        self.res_scale = res_scale

    def forward(self, x):
        res = self.blocks(x)
        res = res * self.res_scale
        output = res + x
        return output 


class simpleResNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_feats, 
                 n_blocks, 
                 norm_type=None, 
                 act_type='leakyrelu', 
                 use_channel_attention=True, 
                 use_global_residual=True, 
                 use_tanh=True,
                 res_scale=1):
        super(simpleResNet, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1, bias=True)
        
        body = [ResBlock(n_feats, norm_type, act_type, use_channel_attention, res_scale=res_scale) for _ in range(n_blocks)]        
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(n_feats, out_channels, 3, 1, 1)

        self.use_global_residual = use_global_residual
        self.use_tanh = use_tanh
        
    def forward(self, x):       
        x = self.head(x)
        output = self.body(x)        
        if self.use_global_residual:
            output = output + x
        output = self.tail(output)
        if self.use_tanh:
            output = torch.tanh(output)        
        return output


class simpleUshapeNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_feats=64, 
                 n_blocks=16, 
                 norm_type=None, 
                 act_type='leakyrelu',
                 use_channel_attention=True,
                 use_global_residual=True, 
                 use_tanh=True,
                 res_scale=1):
        super(simpleUshapeNet, self).__init__()
        self.use_global_residual = use_global_residual
        # self.use_tanh = use_tanh

        # self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        self.en1 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.down1 = nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)  # downsample     
        self.en2 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.down2 = nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)

        blocks = []
        for _ in range(n_blocks):
            block = ResBlock(n_feats, norm_type, act_type, use_channel_attention, res_scale=res_scale)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.up1 = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1) # upsample

        self.de1 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.up2 = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1) #

        self.de2 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )


    def forward(self, x):        
        # x = self.head(x)
        e1 = self.en1(x)
        x = self.down1(e1)
        
        e2 = self.en2(x)
        res = self.down2(e2)
        
        x = self.middle(res)
        if self.use_global_residual:
            x = x + res
        
        
        x = self.up1(x)
        if self.use_global_residual:
            x = x + e2
        x = self.de1(x)
        
        x = self.up2(x)
        if self.use_global_residual:
            x = x + e1
        x = self.de2(x)        

        return x 

#########################################################################
class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class FPN(nn.Module):
    def __init__(self, 
                 n_feats=64, # num_filters
                 norm_type=None):

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.toRGB = nn.Conv2d(n_feats, 3, kernel_size=1)

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        ) # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080

        self.td1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                                 get_norm(norm_type, n_feats),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                                 get_norm(norm_type, n_feats),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                                 get_norm(norm_type, n_feats),
                                 nn.ReLU(inplace=True))

        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, n_feats, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, n_feats, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, n_feats, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, n_feats, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, n_feats, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        x1 = self.toRGB(x)

        enc0 = self.enc0(x1)

        enc1 = self.enc1(enc0) # 256

        enc2 = self.enc2(enc1) # 512

        enc3 = self.enc3(enc2) # 1024

        enc4 = self.enc4(enc3) # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4


class FPNInception(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_feats=64, 
                 n_blocks=16, 
                 norm_type=None, 
                 act_type='leakyrelu',
                 use_channel_attention=True,
                 use_global_residual=True, 
                 use_tanh=True,
                 res_scale=1):

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(n_feats=n_feats, norm_type=norm_type)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(n_feats, n_feats, n_feats)
        self.head2 = FPNHead(n_feats, n_feats, n_feats)
        self.head3 = FPNHead(n_feats, n_feats, n_feats)
        self.head4 = FPNHead(n_feats, n_feats, n_feats) 

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * n_feats, n_feats, kernel_size=3, padding=1),
            get_norm(norm_type, n_feats),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, padding=1),
            get_norm(norm_type, n_feats // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(n_feats // 2, 3, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(3, n_feats, kernel_size=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x, x1):
        #print(x.shape)
        map0, map1, map2, map3, map4 = self.fpn(x)

        # print(x.shape)
        # print(map0.shape)
        # print(map1.shape)
        # print(map2.shape)
        # print(map3.shape)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = final + x1
        res2 = torch.tanh(self.final2(res))

        return res2

        
        

##################################################################################
     


class MSPL_Generator(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 n_feats=128,  
                 n_blocks=[4,4,4,4], 
                 norm_type='batch', 
                 act_type='leakyrelu',
                 use_channel_attention=True,
                 use_global_residual=True, 
                 use_tanh=False):
        super(MSPL_Generator, self).__init__()
        
        self.from_rgb = nn.Conv2d(in_channels, n_feats, (1, 1), bias=True) 

        self.net0 = FPNInception(n_feats, n_feats, n_feats, n_blocks[0], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb0 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 
        
        self.net1 = FPNInception(n_feats, n_feats, n_feats, n_blocks[1], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb1 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 

        self.net2 = FPNInception(n_feats, n_feats, n_feats, n_blocks[2], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb2 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 
        
        self.net3 = FPNInception(n_feats, n_feats, n_feats, n_blocks[3], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)        
        self.to_rgb3 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True)

    def forward(self, x):  
        
        feat = self.from_rgb(x) 

        feat0 = self.net0(feat, x)
        feat1 = self.net1(feat0, x)
        feat2 = self.net2(feat1, x)
        feat3 = self.net3(feat2, x) 
        # feat4 = self.net4(feat3)

        out0 = torch.tanh(self.to_rgb0(feat0)) 
        out1 = torch.tanh(self.to_rgb1(feat1)) 
        out2 = torch.tanh(self.to_rgb2(feat2)) 
        out3 = torch.tanh(self.to_rgb3(feat3))
        # out4 = torch.tanh(self.to_rgb3(feat4)) 
        # print(out3.shape)  
        return out0, out1, out2, out3

    def unfreeze(self):
        self.net0.unfreeze()
        self.net1.unfreeze()
        self.net2.unfreeze()
        self.net3.unfreeze()

