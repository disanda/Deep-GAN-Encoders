# make styleganv1 and BigGAN synthesized images for validation. seeds should be different(0,30000)

import os
import math
import torch
import torchvision
import model.E as BE #default styleganv1, if you will reconstruct other GANs, change corresponding E in here.
import lpips
import numpy as np
import argparse
import tensorboardX
from collections import OrderedDict
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN
from training_utils import *
from model.utils.biggan_config import BigGANConfig
import model.E.E_BIG as BE_BIG

def train(tensor_writer = None, args = None):
    type = args.mtype

    model_path = args.checkpoint_dir_GAN
    config_path = args.config_dir
    if type == 1: # StyleGAN1
        #model_path = './checkpoint/stylegan_v1/ffhq1024/'
        Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
        Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

        Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
        Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

        Gm.buffer1 = torch.load(model_path+'./center_tensor.pt')
        const_ = Gs.const
        const1 = const_.repeat(args.batch_size,1,1,1).cuda()
        layer_num = int(math.log(args.img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024 
        layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
        coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

        Gs.cuda()
        Gm.eval()

        E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)

    elif type == 2:  # StyleGAN2
        #model_path = './checkpoint/stylegan_v2/stylegan2_ffhq1024.pth'
        generator = model_v2.StyleGAN2Generator(resolution=args.img_size).to(device)
        checkpoint = torch.load(model_path) #map_location='cpu'
        if 'generator_smooth' in checkpoint: #default
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=8,randomize_noise=False)
        #Gs = generator.synthesis
        #Gs.cuda()
        #Gm = generator.mapping
        #truncation = generator.truncation
        const_r = torch.randn(args.batch_size)
        const1 = generator.synthesis.early_layer(const_r) #[n,512,4,4]

        E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) # layer_count: 7->256 8->512 9->1024

    elif type == 3:  # PGGAN
        #model_path = './checkpoint/PGGAN/pggan_horse256.pth'
        generator = model_pggan.PGGANGenerator(resolution=args.img_size).to(device)
        checkpoint = torch.load(model_path) #map_location='cpu'
        if 'generator_smooth' in checkpoint: #默认是这个
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        const1 = torch.tensor(0)

        E = BE_PG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, pggan=True)

    elif type == 4:
        model_path = './checkpoint/biggan/256/G-256.pt'
        config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
        config = BigGANConfig.from_json_file(config_file)
        generator = BigGAN(config).to(device)
        generator.load_state_dict(torch.load(model_path))

        E = BE_BIG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, biggan=True).to(device)

    else:
        print('error')
        return

#Load E
#     E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) .to(device)
    if args.checkpoint_dir_E is not None:
        E.load_state_dict(torch.load(args.checkpoint_dir_E, map_location=torch.device(device)))

    batch_size = args.batch_size
    it_d = 0
    for iteration in range(30000,30000+args.iterations):
        set_seed(iteration)
        z = torch.randn(batch_size, args.z_dim) #[32, 512]
        if type == 1:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = Gm(z,coefs_m=coefs).cuda() #[batch_size,18,512]
                imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
        elif type == 2:
            with torch.no_grad():
                #use generator
                result_all = generator(z.cuda(), **synthesis_kwargs)
                imgs1 = result_all['image']
                w1 = result_all['wp']

        elif type == 3:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = z.cuda()
                result_all = generator(w1)
                imgs1 = result_all['image']
        elif type == 4:
            z = truncated_noise_sample(truncation=0.4, batch_size=batch_size, seed=iteration%30000)
            #label = np.random.randint(1000,size=batch_size) # 生成标签
            flag = np.array(30)
            print(flag)
            label = np.ones(batch_size)
            label = flag * label
            label = one_hot(label)
            w1 = torch.tensor(z, dtype=torch.float).cuda()
            conditions = torch.tensor(label, dtype=torch.float).cuda() # as label
            truncation = torch.tensor(0.4, dtype=torch.float).cuda()
            with torch.no_grad(): #这里需要生成图片和变量
                imgs1, const1 = generator(w1, conditions, truncation) # const1 are conditional vectors in BigGAN

        if type != 4:
            const2,w2 = E(imgs1)
        else:
            const2,w2 = E(imgs1, const1)

        if type == 1:
            imgs2=Gs.forward(w2,int(math.log(args.img_size,2)-2))
        elif type == 2 or type == 3:
            imgs2=generator.synthesis(w2)['image']
        elif type == 4:
            imgs2, _=generator(w2, conditions, truncation)
        else:
            print('model type error')
            return

        imgs = torch.cat((imgs1,imgs2))
        torchvision.utils.save_image(imgs*0.5+0.5, resultPath1_1+'/id%s_%s.png'%(flag, str(iteration-30000).rjust(5,'0')),  nrow=10) # nrow=3
        #torchvision.utils.save_image(imgs2*0.5+0.5, resultPath1_2+'/%s_styleganv1_rec.png'%str(iteration-30000).rjust(5,'0')) # nrow=3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=30001) # training seeds: 0-30000; validated seeds > 30000
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v1/ffhq1024/') #None  ./checkpoint/stylegan_v1/ffhq1024/ or ./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth
    parser.add_argument('--config_dir', default=None) # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default='./result/BigGAN-256/models/E_model_ep30000.pth')
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--mtype', type=int, default=4) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=64)  # 16->1024 32->512 64->256
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/BigGAN256_inversion_v4"
        if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/generations"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/reconstructions"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    train(tensor_writer=writer, args = args)
