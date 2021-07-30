import torch
from torch import nn
import random
import module.losses as losses
from module.net import Generator, Mapping, Discriminator
import numpy as np


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)

class Model(nn.Module): #三个网络 Gp Gs D 的封装
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3):
        super(Model, self).__init__()

        self.mapping = Mapping(
            num_layers=2 * layer_count, # 2*9 = 18
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.generator = Generator(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.discriminator = Discriminator(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi       # 0.7
        self.style_mixing_prob = style_mixing_prob 
        self.truncation_cutoff = truncation_cutoff # 前8层

    def generate(self, lod, blend_factor, z=None, count=32, remove_blob=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping(z)
        if self.dlatent_avg_beta is not None: #让原向量以中心向量 dlatent_avg.buff.data 为中心，按比例self.dlatent_avg_beta=0.995围绕中心向量拉近，
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta) # y.lerp(x,a) = y - (y-x)*a

        if self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size) # z2 : [32, 512]
                styles2 = self.mapping(z2)

                layer_idx = torch.arange(self.mapping.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if self.truncation_psi is not None:     #让原向量以中心向量 dlatent_avg.buff.data 为中心，按比例truncation_psi围绕中心向量拉近，
            layer_idx = torch.arange(self.mapping.num_layers)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones) # 18个变量前8个裁剪比例truncation_psi
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs) # avg + (styles-avg) * 0.7

        rec = self.generator.forward(styles, lod, blend_factor, remove_blob) # styles:[-1 , 18, 512]
        return rec

    def forward(self, x, lod, blend_factor, d_train):
        if d_train:
            with torch.no_grad():
                rec = self.generate(lod, blend_factor, count=x.shape[0])
            self.discriminator.requires_grad_(True)
            d_result_real = self.discriminator(x, lod, blend_factor).squeeze()
            d_result_fake = self.discriminator(rec.detach(), lod, blend_factor).squeeze()

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            rec = self.generate(lod, blend_factor, count=x.shape[0])
            self.discriminator.requires_grad_(False)
            d_result_fake = self.discriminator(rec, lod, blend_factor).squeeze()
            loss_g = losses.generator_logistic_non_saturating(d_result_fake)
            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping.parameters()) + list(self.generator.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping.parameters()) + list(other.generator.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
