# Align image cropping

import os
import math
import torch
import torchvision
import model.E.E as BE
import model.E.E_PG as BE_PG
import model.E.E_BIG as BE_BIG
from model.utils.custom_adam import LREQAdam
import metric.pytorch_ssim as pytorch_ssim
import lpips
import numpy as np
import tensorboardX
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
        #model_path = './checkpoint/stylegan_v1/ffhq1024/'
        Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
        Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

        Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
        Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

        Gm.buffer1 = torch.load(model_path+'./center_tensor.pt')
        const_ = Gs.const
        const1 = const_.repeat(args.batch_size,1,1,1).detach().clone().cuda()
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
        const1 = generator.synthesis.early_layer(const_r).detach().clone() #[n,512,4,4]

        #E = BE.BE(startf=64, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
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
        #model_path = './checkpoint/biggan/256/G-256.pt'
        #config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
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

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = args.batch_size
    it_d = 0
    for iteration in range(0,args.iterations):
        set_seed(iteration%30000)
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

                ##use Gs and Gm independently
                # mapping_results = Gm(z)
                # w = mapping_results['w']
                # batch_w_avg = w.mean(dim=0)
                # truncation.w_avg.copy_(truncation.w_avg * 0.995 + batch_w_avg * (1 - 0.995))
                # new_z = torch.randn_like(z)
                # new_w = Gm(new_z)['w']

                # if np.random.uniform() < 0.9:
                #     mixing_cutoff = np.random.randint(1, int(np.log2(args.img_size // 4 * 2)) * 2)
                #     w = truncation(w)
                #     new_w = truncation(new_w)
                #     w[:, :mixing_cutoff] = new_w[:, :mixing_cutoff]

                # w1 = truncation(w,trunc_psi=0.7,trunc_layers=8).cuda()
                # imgs1 = Gs(w1)['image']

        elif type == 3:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = z.cuda()
                result_all = generator(w1)
                imgs1 = result_all['image']
        elif type == 4:
            z = truncated_noise_sample(truncation=0.4, batch_size=batch_size, seed=iteration%30000)
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
            imgs2 = Gs.forward(w2,int(math.log(args.img_size,2)-2))
        elif type == 2 or type == 3:
            imgs2 = generator.synthesis(w2)['image']
        elif type == 4:
            imgs2, _ = generator(w2, conditions, truncation)
        else:
            print('model type error')
            return

        E_optimizer.zero_grad()

#Image-Vectors 

# # Attention region for Aligned Images
# AT1 = imgs_torch[:,:,:,imgs_torch.shape[3]//8:-imgs_torch.shape[3]//8]
# torchvision.utils.save_image(AT1,'./img_torch_at1.png')

# AT2 = imgs_torch[:,:,\
# imgs_torch.shape[2]//8+imgs_torch.shape[2]//32:-imgs_torch.shape[2]//8-imgs_torch.shape[2]//32,\
# imgs_torch.shape[3]//8+imgs_torch.shape[3]//32:-imgs_torch.shape[3]//8-imgs_torch.shape[3]//32
# ]

# Case 1 (default): loss_tsa = loss_imgs + (loss_medium + loss_small)*0.1 
# Case 2: loss_tsa = loss_imgs + 5*loss_medium + 9*loss_small. 
# If using Case 2, chaning Encoder to E_Blur and remoing below loss-vectors' detach()&clone(), also upgrading different vectors separately (see ablation study)

#loss Images
        loss_imgs, loss_imgs_info = space_loss(imgs1.detach().clone(),imgs2.detach().clone(),lpips_model=loss_lpips)

#loss AT1
        imgs_medium_1 = imgs1[:,:,:,imgs1.shape[3]//8:-imgs1.shape[3]//8].detach().clone()
        imgs_medium_2 = imgs2[:,:,:,imgs2.shape[3]//8:-imgs2.shape[3]//8].detach().clone()
        loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)

#loss AT2
        imgs_small_1 = imgs1[:,:,\
        imgs1.shape[2]//8+imgs1.shape[2]//32:-imgs1.shape[2]//8-imgs1.shape[2]//32,\
        imgs1.shape[3]//8+imgs1.shape[3]//32:-imgs1.shape[3]//8-imgs1.shape[3]//32].detach().clone()

        imgs_small_2 = imgs2[:,:,\
        imgs2.shape[2]//8+imgs2.shape[2]//32:-imgs2.shape[2]//8-imgs2.shape[2]//32,\
        imgs2.shape[3]//8+imgs2.shape[3]//32:-imgs2.shape[3]//8-imgs2.shape[3]//32].detach().clone()

        loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)

        loss_tsa = loss_imgs + loss_medium + loss_small
        E_optimizer.zero_grad()
        loss_tsa.backward(retain_graph=True)
        E_optimizer.step()

#Latent-Vectors

## w
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)

## c
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)

        loss_mtv = loss_w*0.01 #+ loss_c*0.01
        E_optimizer.zero_grad()
        loss_mtv.backward()
        E_optimizer.step()

        print('ep_%d_iter_%d'%(iteration//30000,iteration%30000))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_small_info: %s'%loss_small_info)
        print('loss_medium_info: %s'%loss_medium_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)


        it_d += 1
        writer.add_scalar('loss_small_mse', loss_small_info[0][0], global_step=it_d)
        writer.add_scalar('loss_samll_mse_mean', loss_small_info[0][1], global_step=it_d)
        writer.add_scalar('loss_samll_mse_std', loss_small_info[0][2], global_step=it_d)
        writer.add_scalar('loss_samll_kl', loss_small_info[1], global_step=it_d)
        writer.add_scalar('loss_samll_cosine', loss_small_info[2], global_step=it_d)
        writer.add_scalar('loss_samll_ssim', loss_small_info[3], global_step=it_d)
        writer.add_scalar('loss_samll_lpips', loss_small_info[4], global_step=it_d)

        writer.add_scalar('loss_medium_mse', loss_medium_info[0][0], global_step=it_d)
        writer.add_scalar('loss_medium_mse_mean', loss_medium_info[0][1], global_step=it_d)
        writer.add_scalar('loss_medium_mse_std', loss_medium_info[0][2], global_step=it_d)
        writer.add_scalar('loss_medium_kl', loss_medium_info[1], global_step=it_d)
        writer.add_scalar('loss_medium_cosine', loss_medium_info[2], global_step=it_d)
        writer.add_scalar('loss_medium_ssim', loss_medium_info[3], global_step=it_d)
        writer.add_scalar('loss_medium_lpips', loss_medium_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_w_mse', loss_w_info[0][0], global_step=it_d)
        writer.add_scalar('loss_w_mse_mean', loss_w_info[0][1], global_step=it_d)
        writer.add_scalar('loss_w_mse_std', loss_w_info[0][2], global_step=it_d)
        writer.add_scalar('loss_w_kl', loss_w_info[1], global_step=it_d)
        writer.add_scalar('loss_w_cosine', loss_w_info[2], global_step=it_d)
        writer.add_scalar('loss_w_ssim', loss_w_info[3], global_step=it_d)
        writer.add_scalar('loss_w_lpips', loss_w_info[4], global_step=it_d)

        writer.add_scalar('loss_c_mse', loss_c_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c_mse_mean', loss_c_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c_mse_std', loss_c_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c_kl', loss_c_info[1], global_step=it_d)
        writer.add_scalar('loss_c_cosine', loss_c_info[2], global_step=it_d)
        writer.add_scalar('loss_c_ssim', loss_c_info[3], global_step=it_d)
        writer.add_scalar('loss_c_lpips', loss_c_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_small_mse':loss_small_info[0][0],'loss_medium_mse':loss_medium_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_small_kl':loss_small_info[1],'loss_medium_kl':loss_medium_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_samll_cosine':loss_small_info[2],'loss_medium_cosine':loss_medium_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_small_ssim':loss_small_info[3],'loss_medium_ssim':loss_medium_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_small_lpips':loss_small_info[4],'loss_medium_lpips':loss_medium_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_c_info[1],'loss_c_cosine':loss_c_info[2]}, global_step=it_d)


        if iteration % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d_iter%d.jpg'%(iteration//30000,iteration%30000),nrow=n_row) # nrow=3
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(iteration),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_small_info: %s'%loss_small_info,file=f)
                print('loss_medium_info: %s'%loss_medium_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
            if iteration % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d_iter%d.pth'%(iteration//30000,iteration%30000))
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_iter%d.pt'%iteration)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=210000) # epoch = iterations//30000
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth') #None  ./checkpoint/stylegan_v1/ffhq1024/ or ./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth or ./checkpoint/biggan/256/G-256.pt
    parser.add_argument('--config_dir', default='./checkpoint/biggan/256/biggan-deep-256-config.json') # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default=None)
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512) # PGGAN , StyleGANs are 512. BIGGAN is 128
    parser.add_argument('--mtype', type=int, default=2) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)  # 16->1024x1024 32->512x512 64->256x256 / 128->128x128
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/StyleGAN2-FFHQ1024-Aligned-ImgAT1AT2"
        if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    train(tensor_writer=writer, args = args)
