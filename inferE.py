# infer Encoder via pre-trained models, before use this code, set args from line 171 carefully (parameters for model type, path, image size and so on).
import os
import math
import torch
import torch.nn as nn
import torchvision
import model.E.E as BE
#import model.E.E_blur as BE  # if use case 2 change above line
import model.E.E_PG as BE_PG
import model.E.E_BIG as BE_BIG
from model.utils.custom_adam import LREQAdam
import lpips
from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import tensorboardX
import numpy as np
import argparse
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN
from model.utils.biggan_config import BigGANConfig
from training_utils import *

def train(tensor_writer = None, args = None):
    type = args.mtype

    model_path = args.checkpoint_dir_GAN
    if type == 1: # StyleGAN1
        Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
        Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

        Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
        Gm.load_state_dict(torch.load(model_path+'/Gm_dict.pth'))

        Gm.buffer1 = torch.load(model_path+'/center_tensor.pt')
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
        generator = model_v2.StyleGAN2Generator(resolution=args.img_size).to(device)
        checkpoint = torch.load(model_path) #map_location='cpu'
        if 'generator_smooth' in checkpoint: #default
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=8,randomize_noise=False)
        #Gs = generator.synthesis
        #Gm = generator.mapping
        const_r = torch.randn(args.batch_size)
        const1 = generator.synthesis.early_layer(const_r) #[n,512,4,4]
        #E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3) # 256
        E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) # layer_count: 7->256 8->512 9->1024

    elif type == 3:  # PGGAN
        generator = model_pggan.PGGANGenerator(resolution=args.img_size).to(device)
        checkpoint = torch.load(model_path) #map_location='cpu'
        if 'generator_smooth' in checkpoint: #默认是这个
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        const1 = torch.tensor(0)
        E = BE_PG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, pggan=True)

    elif type == 4:
        config = BigGANConfig.from_json_file(args.config_dir)
        generator = BigGAN(config).to(device)
        generator.load_state_dict(torch.load(model_path))
        
        E = BE_BIG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, biggan=True)

    else:
        print('error')
        return

    if args.checkpoint_dir_E != None:
        E.load_state_dict(torch.load(args.checkpoint_dir_E))
    E.cuda()
    writer = tensor_writer

    batch_size = args.batch_size

    #vgg16->Grad-CAM
    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)


    seed = 4 # 0, 1, 4
    set_seed(seed)
    z = torch.randn(batch_size, args.z_dim) #[32, 512]

    if type == 1:
        with torch.no_grad(): #这里需要生成图片和变量
            w1 = Gm(z,coefs_m=coefs).cuda() #[batch_size,18,512]
            imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
    elif type == 2:
        with torch.no_grad():
            result_all = generator(z.cuda(), **synthesis_kwargs)
            imgs1 = result_all['image']
            w1 = result_all['wp']
    elif type == 3:
        with torch.no_grad(): #这里需要生成图片和变量
            w1 = z.cuda()
            result_all = generator(w1)
            imgs1 = result_all['image']
    elif type == 4:
        z = truncated_noise_sample(truncation=0.4, batch_size=batch_size, seed=seed)
        #label = np.random.randint(1000,size=batch_size) # 生成标签
        flag = np.random.randint(1000)
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
        imgs2, _ = generator(w2, conditions, truncation)
    else:
        print('model type error')
        return

#Image Vectors
    mask_1 = grad_cam_plus_plus(imgs1.detach().clone(),None) #[c,1,h,w]
    mask_2 = grad_cam_plus_plus(imgs2.detach().clone(),None)
    # imgs1.retain_grad()
    # imgs2.retain_grad()
    imgs1_ = imgs1.detach().clone()
    imgs1_.requires_grad = True
    imgs2_ = imgs2.detach().clone()
    imgs2_.requires_grad = True
    grad_1 = gbp(imgs1_) # [n,c,h,w]
    grad_2 = gbp(imgs2_)
    heatmap_1,cam_1 = mask2cam(mask_1.detach().clone(),imgs1.detach().clone())
    heatmap_2,cam_2 = mask2cam(mask_2.detach().clone(),imgs2.detach().clone())

    for i,j in enumerate(imgs1):
        torchvision.utils.save_image(j.unsqueeze(0)*0.5+0.5, resultPath1_1+'/seed%d_iter%d.png'%(seed,i),nrow=1)
    for i,j in enumerate(imgs2):
        torchvision.utils.save_image(j.unsqueeze(0)*0.5+0.5, resultPath1_1+'/seed%d_iter%d-rc.png'%(seed,i),nrow=1)

    for i,j in enumerate(heatmap_1):
        torchvision.utils.save_image(j.unsqueeze(0), resultPath_grad_cam+'/seed%d_iter%d-heatmap.png'%(seed,i),nrow=1)
    for i,j in enumerate(heatmap_2):
        torchvision.utils.save_image(j.unsqueeze(0), resultPath_grad_cam+'/seed%d_iter%d-heatmap-rc.png'%(seed,i),nrow=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=210000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--experiment_dir', default=None)
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v2/stylegan2_cat256.pth')
    parser.add_argument('--config_dir', default='./checkpoint/biggan/256/biggan-deep-256-config.json') # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default='./checkpoint/E/E_Cat_styleganv2_ep6.pth')#'./result/StyleGAN1-car512-Aligned-modelV2/models/E_model_iter100000.pth'
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512) # BigGAN,z=128
    parser.add_argument('--mtype', type=int, default=2) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN00
    parser.add_argument('--start_features', type=int, default=64) # 16->1024 32->512 64->256 
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/infer_cat256_ep6_styleganv2"
        if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    resultPath_grad_cam = resultPath+"/grad_cam"
    if not os.path.exists(resultPath_grad_cam): os.mkdir(resultPath_grad_cam)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    train(tensor_writer=writer, args= args)