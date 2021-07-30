#这个版本是最早在AAAI上试用的版本，其来自MSG-GAN的修改

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList, AvgPool2d
from model.utils.CustomLayers import DisGeneralConvBlock, DisFinalBlock, _equalized_conv2d

#和原网络中D对应的Encoder, 训练时G不变， v1只改了最后一层，v2是一个规模较小的网络

# in: [-1,512] ,    out: [-1,3,1024,1024]
class encoder_v1(torch.nn.Module):
    """ Discriminator of the GAN """
    def __init__(self, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"
        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size
        #self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)
        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list
        # create the fromRGB layers for various inputs:
        if self.use_eql:
            self.fromRGB = lambda out_channels: \
                _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=True)
        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])
        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(self.feature_size,
                                            self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)
            self.layers.append(layer)
            self.rgb_to_features.append(rgb)
        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)
        #new
        self.new_final = nn.Conv2d(512, 512, 4, 1, 0, bias=True)
    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """
        assert height < self.height, "Requested output depth cannot be produced"
        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        #out = self.final_block(y)
        out = self.new_final(y)
        return out

#in: [-1,3,1024,1024], out: [-1,512] 
class encoder_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
        nn.Conv2d(3,12,4,2,1,bias=False), # 1024->512
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(12,12,4,2,1,bias=False),# 512->256
        nn.BatchNorm2d(12),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(12,3,4,2,1,bias=False),# 256->128
        nn.BatchNorm2d(3),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(3,1,4,2,1,bias=False),# 128->64*64=4096
    )
        self.fc = nn.Linear(4096,512)
    def forward(self, x):
        y1 = self.main(x)
        y2 = y1.view(-1,4096)
        y3 = self.fc(y2)
        return y3