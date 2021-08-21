#reconstructing real_image by dirrectlly MTV E (no optimize E via StyleGANv1)
#set arg.realimg_dir for embedding real images to latent vectors
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

    if not os.path.exists('./images'): os.mkdir('./images')

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_gan', default='./checkpoint/stylegan_v1/ffhq1024/') # stylegan_v2/stylegan2_ffhq1024.pth 
    parser.add_argument('--checkpoint_dir_e', default='./checkpoint/E/E_styleganv1_state_dict.pth') #None or E_ffhq_styleganv2_modelv2_ep110000.pth E_ffhq_styleganv2_modelv1_ep85000.pth
    parser.add_argument('--config_dir', default=None)
    parser.add_argument('--realimg_dir', default='./images/real_images128/')
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)
    args = parser.parse_args()

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    resultPath1_1 = "./images/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = "./images/rec"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

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

    else:
        print('error')

    #Load E
    E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3) .to(device)
    if args.checkpoint_dir_e is not None:
        E.load_state_dict(torch.load(args.checkpoint_dir_e, map_location=torch.device(device)))

    type = args.mtype
    save_path = args.realimg_dir
    imgs_path = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".png") or f.endswith(".jpg")]
    img_size = args.img_size

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
        torchvision.utils.save_image(imgs1*0.5+0.5, resultPath1_1+'/%s_realimg.png'%str(i).rjust(5,'0'))
        torchvision.utils.save_image(imgs2*0.5+0.5, resultPath1_2+'/%s_mtv_rec.png'%str(i).rjust(5,'0'))
        print('doing:'+str(i).rjust(5,'0'))







