# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
# 比第0版多了残差, 每一层的两个(conv/line)输出的w1和w2合并为1个w
# 比第1版加了要学习的bias_1和bias_2，网络顺序和第1版有所不同(更对称)
# 比第2版，即可以使用到styleganv1,styleganv2, 不再使用带Equalize learning rate的Conv (这条已经废除). 以及Block第二层的blur操作
# 改变了上采样，不在conv中完成
# 改变了In,带参数的学习
# 改变了了residual,和残差网络一致，另外旁路多了conv1处理通道和In学习参数
# 经测试，不带Eq(Equalize Learning Rate)的参数层学习效果不好


import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
from torch.nn import functional as F
import model.utils.lreq as ln

# G 改 E, 实际上需要用G Block改出E block, 完成逆序对称，在同样位置还原style潜码
# 比第0版多了残差, 每一层的两个(conv/line)输出的w1和w2合并为1个w
# 比第1版加了要学习的bias_1和bias_2，网络顺序和第1版有所不同(更对称)

class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)
    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)
        return x

class BEBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_second_conv=True, fused_scale=True): #分辨率大于128用fused_scale,即conv完成上采样
        super().__init__()
        self.has_second_conv = has_second_conv
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False, eps=1e-8)
        self.inver_mod1 = ln.Linear(2 * inputs, latent_size) # [n, 2c] -> [n,512]
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)

        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.inver_mod2 = ln.Linear(2 * inputs, latent_size)
        if has_second_conv:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.fused_scale = fused_scale
        
        self.inputs = inputs
        self.outputs = outputs

        if self.inputs != self.outputs:
            self.conv_3 = ln.Conv2d(inputs, outputs, 1, 1, 0)
            self.instance_norm_3 = nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        residual = x
        mean1 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std1 = torch.sqrt(torch.mean((x - mean1) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style1 = torch.cat((mean1, std1), dim=1) # [b,2c,1,1]
        w1 = self.inver_mod1(style1.view(style1.shape[0],style1.shape[1])) # [b,2c]->[b,512]

        x = self.conv_1(x)
        x = self.instance_norm_1(x)
        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))
        x = x + self.bias_1
        x = F.leaky_relu(x, 0.2)


        mean2 = torch.mean(x, dim=[2, 3], keepdim=True) # [b, c, 1, 1]
        std2 = torch.sqrt(torch.mean((x - mean2) ** 2, dim=[2, 3], keepdim=True))  # [b, c, 1, 1]
        style2 = torch.cat((mean2, std2), dim=1) # [b,2c,1,1]
        w2 = self.inver_mod2(style2.view(style2.shape[0],style2.shape[1])) # [b,512] , 这里style2.view一直写错成style1.view

        if self.has_second_conv:
            x = self.conv_2(x)
            x = self.instance_norm_2(x)
            x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2, tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]).to(x.device))
            x = x + self.bias_2
            if self.inputs != self.outputs: 
                residual = self.conv_3(residual)
                residual = self.instance_norm_3(residual)
            x = x + residual
            x = F.leaky_relu(x, 0.2)
            if not self.fused_scale: #上采样
                x = F.avg_pool2d(x, 2, 2)

        #x = 0.111*x+0.889*residual #降低x的比例，可以将const的loss缩小！！0.7*residual： 10-11 >> 7 同时 c_s的loss扩大至3， ws的抖动提前, 效果更好
        return x, w1, w2


class BE(nn.Module):
    def __init__(self, startf=16, maxf=512, layer_count=9, latent_size=512, channels=3, pggan=False):
        super().__init__()
        self.maxf = maxf
        self.startf = startf
        self.latent_size = latent_size
        #self.layer_to_resolution = [0 for _ in range(layer_count)]
        self.decode_block = nn.ModuleList()
        self.layer_count = layer_count
        inputs = startf # 16 
        outputs = startf*2
        #resolution = 1024
        # from_RGB = nn.ModuleList()
        fused_scale = False
        self.FromRGB = FromRGB(channels, inputs)

        for i in range(layer_count):

            has_second_conv = i+1 != layer_count #普通的D最后一个块的第二层是 mini_batch_std
            #fused_scale = resolution >= 128 # 在新的一层起初 fused_scale = flase, 完成上采样
            #from_RGB.append(FromRGB(channels, inputs))

            block = BEBlock(inputs, outputs, latent_size, has_second_conv, fused_scale=fused_scale)

            inputs = inputs*2
            outputs = outputs*2
            inputs = min(maxf, inputs) 
            outputs = min(maxf, outputs)
            #self.layer_to_resolution[i] = resolution
            #resolution /=2
            self.decode_block.append(block)
        #self.FromRGB = from_RGB

        self.pggan = pggan
        if pggan:
            self.new_final = nn.Conv2d(512, 512, 4, 1, 0, bias=True)

    #将w逆序，以保证和G的w顺序, block_num控制progressive
    def forward(self, x, block_num=9):
        #x = self.FromRGB[9-block_num](x) #每个block一个
        x = self.FromRGB(x)
        w = torch.tensor(0)
        for i in range(9-block_num,self.layer_count):
            x,w1,w2 = self.decode_block[i](x)
            w_ = torch.cat((w2.view(x.shape[0],1,512),w1.view(x.shape[0],1,512)),dim=1) # [b,2,512]
            if i == (9-block_num):
                w = w_ # [b,n,512]
            else:
                w = torch.cat((w_,w),dim=1)
        if self.pggan:
            x = self.new_final(x)
        return x, w

#test
# E = BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
# imgs1 = torch.randn(2,3,256,256)
# const2,w2 = E(imgs1)
# print(const2.shape)
# print(w2.shape)
# print(E)
