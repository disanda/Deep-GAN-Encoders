import os
import math
import torch
import argparse
import numpy as np
import torchvision
import model.E.E_Blur as BE
from collections import OrderedDict
from training_utils import *
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_gan', default='./checkpoint/stylegan_v1/ffhq1024/') # stylegan_v2/stylegan2_ffhq1024.pth 
    parser.add_argument('--checkpoint_dir_e', default='./checkpoint/E/styleGANv1_EAE_ep65000.pth') #None or E_ffhq_styleganv2_modelv2_ep110000.pth E_ffhq_styleganv2_modelv1_ep85000.pth
    parser.add_argument('--config_dir', default=None)
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)
    args = parser.parse_args()

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

#Load GANs
    type = args.mtype
    model_path = args.checkpoint_dir_gan
    config_path = args.config_dir

    if type == 1: # StyleGAN1, 1 diretory contains 3files(Gm, Gs, center-tensor)
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

    elif type == 2:  # StyleGAN2, a pth file.
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

        #E = BE.BE(startf=64, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
        E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) # layer_count: 7->256 8->512 9->1024

    elif type == 3:  # PGGAN, a pth file.
        #model_path = './checkpoint/PGGAN/pggan_horse256.pth'
        generator = model_pggan.PGGANGenerator(resolution=args.img_size).to(device)
        checkpoint = torch.load(model_path) #map_location='cpu'
        if 'generator_smooth' in checkpoint: #默认是这个
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        const1 = torch.tensor(0)

        E = BE_PG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, pggan=True)

    elif type == 4: # BigGAN, 2 files. G.pt and config.json
        #model_path = './checkpoint/biggan/256/G-256.pt'
        #config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
        config = BigGANConfig.from_json_file(config_file)
        generator = BigGAN(config)
        generator.load_state_dict(torch.load(model_path))
        E = BE_BIG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, biggan=True)
    else:
        print('error')

#Load E
    E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) .to(device)
    #E.load_state_dict(torch.load(args.checkpoint_dir_e, map_location=torch.device(device)))
    # omit RGB layers EAEv2->MSVv2:
    if args.checkpoint_dir_e != None:
        E_dict = torch.load(args.checkpoint_dir_e,map_location=torch.device(device))
        new_state_dict = OrderedDict()
        for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
                new_state_dict[i1] = j2 
        E.load_state_dict(new_state_dict)


    set_seed(args.seed)
    z = torch.randn(args.batch_size, args.z_dim) #[32, 512]

    type = args.mtype
    # if type == 1:
    #     with torch.no_grad(): #这里需要生成图片和变量
    #         w1 = Gm(z,coefs_m=coefs).to(device) #[batch_size,18,512]
    #         imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
    # elif type == 2:
    #     with torch.no_grad():
    #         #use generator
    #         result_all = generator(z.to(device), **synthesis_kwargs)
    #         imgs1 = result_all['image']
    #         w1 = result_all['wp']

    #os.listdir('C:/Users/Administrator/Desktop/TIP-Images/test_imgs')
    save_path = 'C:/Users/Administrator/Desktop/TIP-Images/test_imgs/'
    imgs_path = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".png")]
    img_size = 1024

    #PIL 2 Tensor
    transform = torchvision.transforms.Compose([
            #torchvision.transforms.CenterCrop(160),
            torchvision.transforms.Resize((img_size,img_size)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    images = []
    for idx, image_path in enumerate(imgs_path):
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)

    imgs_tensor = torch.stack(images, dim=0)

    for i, j  in enumerate(imgs_tensor):
        imgs1 = j.unsqueeze(0).cuda() * 2 -1
        if type != 4:
            const2,w2 = E(imgs1)
        else:
            const2,w2 = E(imgs1, cond_vector)


        if type == 1:
            imgs2=Gs.forward(w2,int(math.log(args.img_size,2)-2))
        elif type == 2 or type == 3:
            imgs2=generator.synthesis(w2)['image']
        elif type == 4:
            imgs2, _=G(w2, conditions, truncation)
        else:
            print('model type error')

    # n_row = args.batch_size
    # test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
    # torchvision.utils.save_image(test_img, './v2ep%d.jpg'%(args.seed),nrow=n_row) # nrow=3
        torchvision.utils.save_image(imgs1*0.5+0.5, './img%d.png'%i)
        torchvision.utils.save_image(imgs2*0.5+0.5, './%s_imgrc_v2.png'%str(i).rjust(3,'0'))