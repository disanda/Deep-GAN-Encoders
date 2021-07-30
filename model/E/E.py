import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import sys
#sys.path.append('../')
import model.utils.lreq as ln
from model.utils.net import Blur,FromRGB,downscale2d
from torch.nn import functional as F

# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
# 比第0版多了残差, 每一层的两个(conv/line)输出的w1和w2合并为1个w
# 比第1版加了要学习的bias_1和bias_2，网络顺序和第1版有所不同(更对称)

class BEBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_last_conv=True, fused_scale=True): #分辨率大于128用fused_scale,即conv完成上采样
        super().__init__()
        self.has_last_conv = has_last_conv
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False, eps=1e-8)
        self.inver_mod1 = ln.Linear(2 * inputs, latent_size, gain=1) # [n, 2c] -> [n,512]
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)

        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(inputs, affine=False, eps=1e-8)
        self.inver_mod2 = ln.Linear(2 * inputs, latent_size, gain=1)
        #self.blur = Blur(inputs)
        if has_last_conv:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.fused_scale = fused_scale
        
        self.inputs = inputs
        self.outputs = outputs

        if self.inputs != self.outputs:
            self.conv_3 = ln.Conv2d(inputs, outputs, 1, 1, 0)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        mean1 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std1 = torch.sqrt(torch.mean((x - mean1) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style1 = torch.cat((mean1, std1), dim=1) # [b,2c,1,1]
        w1 = self.inver_mod1(style1.view(style1.shape[0],style1.shape[1])) # [b,2c]->[b,512]

        residual = x

        x = self.instance_norm_1(x)
        x = self.conv_1(x)
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))
        x = x + self.bias_1
        x = F.leaky_relu(x, 0.2)

        mean2 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std2 = torch.sqrt(torch.mean((x - mean2) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style2 = torch.cat((mean2, std2), dim=1) # [b,2c,1,1]
        w2 = self.inver_mod2(style2.view(style2.shape[0],style2.shape[1])) # [b,512] , 这里style2.view一直写错成style1.view

        x = self.instance_norm_2(x)
        if self.has_last_conv:
            #x = self.blur(x)
            x = self.conv_2(x)
            x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))
            x = x + self.bias_2
            x = F.leaky_relu(x, 0.2)
            if not self.fused_scale: #在新的一层起初 fused_scale = flase, 完成上采样
                x = downscale2d(x)
            residual = downscale2d(residual)


        if self.inputs != self.outputs: 
            residual = self.conv_3(residual)

        x = 0.111*x+0.889*residual #降低x的比例，可以将const的loss缩小！！0.7*residual： 10-11 >> 7 同时 c_s的loss扩大至3， ws的抖动提前, 效果更好
        return x, w1, w2


class BE(nn.Module):
    def __init__(self, startf=16, maxf=512, layer_count=9, latent_size=512, channels=3):
        super().__init__()
        self.maxf = maxf
        self.startf = startf
        self.latent_size = latent_size
        self.layer_to_resolution = [0 for _ in range(layer_count)]
        self.decode_block = nn.ModuleList()
        self.layer_count = layer_count
        inputs = startf # 16 
        outputs = startf*2
        resolution = 1024
        self.FromRGB = FromRGB(channels, inputs)
        #from_RGB = nn.ModuleList()
        for i in range(layer_count):

            has_last_conv = i+1 != layer_count
            #fused_scale = resolution >= 128 # 在新的一层起初 fused_scale = flase, 完成上采样
            fused_scale = False

            #from_RGB.append(FromRGB(channels, inputs))
            block = BEBlock(inputs, outputs, latent_size, has_last_conv, fused_scale=fused_scale)

            inputs = inputs*2
            outputs = outputs*2
            inputs = min(maxf, inputs) 
            outputs = min(maxf, outputs)
            self.layer_to_resolution[i] = resolution
            resolution /=2
            self.decode_block.append(block)

        #self.FromRGB = from_RGB

    #将w逆序，以保证和G的w顺序, block_num控制progressive
    def forward(self, x, block_num=9):
        #x = self.FromRGB[9-block_num](x) #不是progressive,去除多余的FromRGB
        x = self.FromRGB(x)
        #print(x.shape)
        w = torch.tensor(0)
        for i in range(9-block_num,self.layer_count):
            x,w1,w2 = self.decode_block[i](x)
            #print(x.shape)
            w_ = torch.cat((w2.view(x.shape[0],1,512),w1.view(x.shape[0],1,512)),dim=1) # [b,2,512]
            if i == (9-block_num):
                w = w_ # [b,n,512]
            else:
                w = torch.cat((w_,w),dim=1)
            #print(w.shape)
        return x, w