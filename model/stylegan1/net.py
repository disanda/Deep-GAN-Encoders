# 三个网络的实现Gm,Gs,D

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import model.stylegan1.lreq as ln
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
#from dlutils.pytorch import count_parameters, millify

if False:
    def lerp(s, e, x):
        return s + (e-s) * x
    def rsqrt(x):
        return 1.0 / x ** 0.5
    def addcmul(x, value, tensor1, tensor2):
        return x + value * tensor1 * tensor2
    torch.lerp = lerp
    torch.rsqrt = rsqrt
    torch.addcmul = addcmul

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1) # [n,1024] -> [n,2,512,1,1]
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1) # style1 + x*style2


def upscale2d(x, factor=2):
    #    return F.upsample(x, scale_factor=factor, mode='bilinear', align_corners=True)
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)

class Blur(nn.Module): #将每一个特征图[h,w]按照卷积核的比例进行缩放，即根据卷积核中心点实现数据集中，但周围模糊. 且能将值压缩一定比例
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :] # size: [3,1]*[1,3] =  [3,3] , 中间大，两头小
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1) # [channels, 1, 3, 3] 
        self.register_buffer('weight', kernel)
        self.groups = channels
    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)

def minibatch_stddev_layer(x, group_size=4):
    group_size = min(group_size, x.shape[0])
    size = x.shape[0]
    if x.shape[0] % group_size != 0:
        x = torch.cat([x, x[:(group_size - (x.shape[0] % group_size)) % group_size]])
    y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y**2).mean(dim=0) + 1e-8).mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y], dim=1)[:size]

class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        if last:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)    # 最后一层只有1个conv，加 minibatch_stddev,其他层有两个conv
        return x

class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                self.conv_1 = ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_1 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_1 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_2 = ln.Linear(latent_size, 2 * outputs, gain=1)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, s1, s2):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device)) #随机噪音编码，噪音变量1

        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device)) #随机噪音编码，噪音变量2

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x

    def forward_double(self, x, _x, s1, s2):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
                _x = upscale2d(_x)
            x = self.conv_1(x)
            _x = self.conv_1(_x)

            x = self.blur(x)
            _x = self.blur(_x)

        n1 = torch.randn([int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])])
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                          tensor2=n1)

        _x = torch.addcmul(_x, value=1.0, tensor1=self.noise_weight_1,
                          tensor2=n1)

        x = x + self.bias_1
        _x = _x + self.bias_1

        x = F.leaky_relu(x, 0.2)
        _x = F.leaky_relu(_x, 0.2)


        std = x.std(axis=[2, 3], keepdim=True)
        mean = x.mean(axis=[2, 3], keepdim=True)

        x = (x - mean) / std
        _x = (_x - mean) / std

        x = style_mod(x, self.style_1(s1))
        _x = style_mod(_x, self.style_1(s1))

        x = self.conv_2(x)
        _x = self.conv_2(_x)

        n2 = torch.randn([int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])])

        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                          tensor2=n2)

        _x = torch.addcmul(_x, value=1.0, tensor1=self.noise_weight_2,
                          tensor2=n2)

        x = x + self.bias_2
        _x = _x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        _x = F.leaky_relu(_x, 0.2)

        std = x.std(axis=[2, 3], keepdim=True)
        mean = x.mean(axis=[2, 3], keepdim=True)

        x = (x - mean) / std
        _x = (_x - mean) / std

        x = style_mod(x, self.style_2(s2))
        _x = style_mod(_x, self.style_2(s2))

        return x, _x

class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)
     
        return x

class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0, gain=1)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        mul = 2**(self.layer_count-1) # mul =  4, 8, 16, 32 ... | layer_count=6 -> 128*128

        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4)) #[1,512,4,4]
        self.zeros = torch.zeros(1, 1, 1, 1)
        init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0 # i=0 -> False i!= -> Ture
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale) 

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            #print("decode_block%d %s styles in: %dl out resolution: %d" % ((i + 1), millify(count_parameters(block)), outputs, resolution)) #输出参数大小
            self.decode_block.append(block)
            inputs = outputs
            mul //= 2 #逐渐递减到 1 

        self.to_rgb = to_rgb

    def decode3(self, styles, lod, noise, remove_blob=True): #这个模块步不更新梯度，仅能用于去除blob
        x = self.const
        _x = None
        for i in range(lod + 1):
            if i < 4 or not remove_blob:
                x = self.decode_block[i].forward(x, styles[:, 2 * i + 0], styles[:, 2 * i + 1])
                if remove_blob and i == 3:
                    _x = x.clone()
                    _x[x > 300.0] = 0

                # plt.hist((torch.max(torch.max(_x, dim=2)[0], dim=2)[0]).cpu().flatten().numpy(), bins=300)
                # plt.show()
                # exit()
            else:
                x, _x = self.decode_block[i].forward_double(x, _x, styles[:, 2 * i + 0], styles[:, 2 * i + 1])

        if _x is not None:
            x = _x # 大于300的值被清零
        if lod == 8:
            x = self.to_rgb[lod](x)
        else:
            x = x.max(dim=1, keepdim=True)[0]
            x = x - x.min()
            x = x / x.max()
            x = torch.pow(x, 1.0/2.2)
            x = x.repeat(1, 3, 1, 1)
        return x

    def decode(self, styles, lod, noise=0):
        x = self.const
        for i in range(lod+1):
            x = self.decode_block[i](x, styles[:, 2*i+0], styles[:,2*i+1])
        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise=0): #这步会完成图像混合style
        x = self.const

        for i in range(lod):
            x = self.decode_block[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1])

        x_prev = self.to_rgb[lod - 1](x)

        x = self.decode_block[lod](x, styles[:, 2 * lod + 0], styles[:, 2 * lod + 1])
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = torch.lerp(x_prev, x, blend)

        return x

    def forward(self, styles, lod, blend=1, remove_blob=False):
        if remove_blob == True:
            return self.decode3(styles, lod, 1)
        if blend == 1:
            return self.decode(styles, lod, 1)
        else: #blend<1,混合
            return self.decode2(styles, lod, blend)


class Discriminator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(inputs, outputs, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul=0.01):
        super(MappingBlock, self).__init__()
        self.fc = ln.Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x

class Mapping(nn.Module):
    def __init__(self, num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512, trunc_tensor=None):
        super(Mapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers # 8
        self.num_layers = num_layers #映射的扩充的层数，由1层 扩充到 2*9 = 18层, 256*256就是 2*7=14层
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.01)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)
        self.register_buffer('buffer1', trunc_tensor) #[18,512]

    def forward(self, z, coefs_m=0):
        x = pixel_norm(z)

        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)

        x = x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1) # [-1,1,512] -> [-1,18,512], 这里18层每层是一样的


        if self.buffer1 is not None:
            x = torch.lerp(self.buffer1.data, x, coefs_m) # avg + (styles-avg) * coefs

        return x


class Mapping2(nn.Module):
    def __init__(self, num_layers=18, mapping_layers=8, latent_size=512, trunc_tensor=None, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.mapping_layers = mapping_layers # 8
        self.num_layers = num_layers #映射的扩充的层数，由1层 扩充到 2*9 = 18层
        for i in range(mapping_layers-1):
            block = MappingBlock(latent_size, latent_size, lrmul=0.01)
            setattr(self, "block_%d" % (i + 1), block)
        if inverse == False:
            block = MappingBlock(latent_size, num_layers*latent_size, lrmul=0.01)
            setattr(self, "block_%d" % (mapping_layers), block)
        else:
            block = MappingBlock(num_layers*latent_size, latent_size, lrmul=0.01)
            setattr(self, "block_%d" % (mapping_layers), block)

    def forward(self, z, coefs_m=0):
        x = pixel_norm(z)

        if self.inverse ==False:
            for i in range(self.mapping_layers):
                x = getattr(self, "block_%d" % (i + 1))(x)
            x = x.view(-1,self.num_layers,512)
        else:
            x = x.view(-1,self.num_layers*512)
            for i in range(self.mapping_layers,0,-1):
                x = getattr(self, "block_%d" % (i))(x)

        return x


class Mapping3(nn.Module):
    def __init__(self, num_layers=18, mapping_layers=8, latent_size=512):
        super().__init__()
        self.mapping_layers = mapping_layers # 8
        self.num_layers = num_layers #映射的扩充的层数，由1层 扩充到 2*9 = 18层

        block = MappingBlock(512, 512*2, lrmul=0.01)
        setattr(self, "block_%d" % (1), block)
        block = MappingBlock(512*2, 512*4, lrmul=0.01)
        setattr(self, "block_%d" % (2), block)
        block = MappingBlock(512*4, 512*6, lrmul=0.01)
        setattr(self, "block_%d" % (3), block)
        block = MappingBlock(512*6, 512*8, lrmul=0.01)
        setattr(self, "block_%d" % (4), block)
        block = MappingBlock(512*8, 512*10, lrmul=0.01)
        setattr(self, "block_%d" % (5), block)
        block = MappingBlock(512*10, 512*12, lrmul=0.01)
        setattr(self, "block_%d" % (6), block)
        block = MappingBlock(512*12, 512*14, lrmul=0.01)
        setattr(self, "block_%d" % (7), block)
        block = MappingBlock(512*14, 512*18, lrmul=0.01)
        setattr(self, "block_%d" % (8), block)

    def forward(self, z, coefs_m=0):
        x = pixel_norm(z)
        for i in range(self.mapping_layers):
           x = getattr(self, "block_%d" % (i + 1))(x)
        x = x.view(-1,self.num_layers,512)
        return x

class Mapping4(nn.Module):
    def __init__(self, num_layers=18, mapping_layers=8, latent_size=512):
        super().__init__()
        self.mapping_layers = mapping_layers # 8
        self.num_layers = num_layers #映射的扩充的层数，由1层 扩充到 2*9 = 18层

        block = MappingBlock(512*18, 512*14, lrmul=0.01)
        setattr(self, "block_%d" % (1), block)
        block = MappingBlock(512*14, 512*12, lrmul=0.01)
        setattr(self, "block_%d" % (2), block)
        block = MappingBlock(512*12, 512*10, lrmul=0.01)
        setattr(self, "block_%d" % (3), block)
        block = MappingBlock(512*10, 512*8, lrmul=0.01)
        setattr(self, "block_%d" % (4), block)
        block = MappingBlock(512*8, 512*6, lrmul=0.01)
        setattr(self, "block_%d" % (5), block)
        block = MappingBlock(512*6, 512*4, lrmul=0.01)
        setattr(self, "block_%d" % (6), block)
        block = MappingBlock(512*4, 512*2, lrmul=0.01)
        setattr(self, "block_%d" % (7), block)
        block = MappingBlock(512*2, 512, lrmul=0.01)
        setattr(self, "block_%d" % (8), block)

    def forward(self, z, coefs_m=0):
        x = pixel_norm(z)
        x = x.view(-1,self.num_layers*512)
        for i in range(self.mapping_layers):
           x = getattr(self, "block_%d" % (i + 1))(x)
        return x

