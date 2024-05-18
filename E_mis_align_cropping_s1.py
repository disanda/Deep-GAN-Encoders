# cropping mis_aligned_image via Gram-CAM

import os
import math
import torch
import torch.nn as nn
import torchvision
import model.E.E as BE
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

#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = False # faster

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
        const1 = const_.repeat(args.batch_size,1,1,1).detach().clone().cuda()
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
        const1 = generator.synthesis.early_layer(const_r).detach().clone() #[n,512,4,4]
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

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0) 
    #用这个adam不会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = args.batch_size
    it_d = 0

    #vgg16->Grad-CAM
    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)


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
        
        E_optimizer.zero_grad()

#Image Vectors
        mask_1 = grad_cam_plus_plus(imgs1,None) #[c,1,h,w]
        mask_2 = grad_cam_plus_plus(imgs2,None)
        # imgs1.retain_grad()
        # imgs2.retain_grad()
        imgs1_ = imgs1.detach().clone()
        imgs1_.requires_grad = True
        imgs2_ = imgs2.detach().clone()
        imgs2_.requires_grad = True
        grad_1 = gbp(imgs1_) # [n,c,h,w]
        grad_2 = gbp(imgs2_)
        heatmap_1,cam_1 = mask2cam(mask_1,imgs1)
        heatmap_2,cam_2 = mask2cam(mask_2,imgs2)

        loss_grad, loss_grad_info = space_loss(grad_1,grad_2,lpips_model=loss_lpips)

    ##--Image
        loss_imgs, loss_imgs_info = space_loss(imgs1.detach().clone(),imgs2.detach().clone(),lpips_model=loss_lpips)

    ##--Mask_Cam as AT1 (HeatMap from Mask)
        mask_1 = mask_1.float().to(device)
        mask_1.requires_grad=True
        mask_2 = mask_2.float().to(device)
        mask_2.requires_grad=True
        loss_mask, loss_mask_info = space_loss(mask_1.detach().clone(),mask_2.detach().clone(),lpips_model=loss_lpips)

    ##--Grad_CAM as AT2 (from mask with img)
        cam_1 = cam_1.float().to(device)
        cam_1.requires_grad=True
        cam_2 = cam_2.float().to(device)
        cam_2.requires_grad=True
        loss_Gcam, loss_Gcam_info = space_loss(cam_1.detach().clone(),cam_2.detach().clone(),lpips_model=loss_lpips)

        loss_tsa = loss_imgs + loss_mask + loss_Gcam
        E_optimizer.zero_grad()
        loss_tsa.backward(retain_graph=True)
        E_optimizer.step()

#Latent Vectors
    ##--C
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)

    ##--W
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)

        loss_mtv =  loss_w*0.01 #+ loss_c*0.01
        E_optimizer.zero_grad()
        loss_mtv.backward()
        E_optimizer.step()

        print('ep_%d_iter_%d'%(iteration//30000,iteration%30000))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_mask_info: %s'%loss_mask_info)
        print('loss_grad_info: %s'%loss_grad_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('loss_Gcam_info: %s'%loss_Gcam_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)

        it_d += 1
        writer.add_scalar('loss_mask_mse', loss_mask_info[0][0], global_step=it_d)
        writer.add_scalar('loss_mask_mse_mean', loss_mask_info[0][1], global_step=it_d)
        writer.add_scalar('loss_mask_mse_std', loss_mask_info[0][2], global_step=it_d)
        writer.add_scalar('loss_mask_kl', loss_mask_info[1], global_step=it_d)
        writer.add_scalar('loss_mask_cosine', loss_mask_info[2], global_step=it_d)
        writer.add_scalar('loss_mask_ssim', loss_mask_info[3], global_step=it_d)
        writer.add_scalar('loss_mask_lpips', loss_mask_info[4], global_step=it_d)

        writer.add_scalar('loss_grad_mse', loss_grad_info[0][0], global_step=it_d)
        writer.add_scalar('loss_grad_mse_mean', loss_grad_info[0][1], global_step=it_d)
        writer.add_scalar('loss_grad_mse_std', loss_grad_info[0][2], global_step=it_d)
        writer.add_scalar('loss_grad_kl', loss_grad_info[1], global_step=it_d)
        writer.add_scalar('loss_grad_cosine', loss_grad_info[2], global_step=it_d)
        writer.add_scalar('loss_grad_ssim', loss_grad_info[3], global_step=it_d)
        writer.add_scalar('loss_grad_lpips', loss_grad_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_Gcam', loss_Gcam_info[0][0], global_step=it_d)
        writer.add_scalar('loss_Gcam_mean', loss_Gcam_info[0][1], global_step=it_d)
        writer.add_scalar('loss_Gcam_std', loss_Gcam_info[0][2], global_step=it_d)
        writer.add_scalar('loss_Gcam_kl', loss_Gcam_info[1], global_step=it_d)
        writer.add_scalar('loss_Gcam_cosine', loss_Gcam_info[2], global_step=it_d)
        writer.add_scalar('loss_Gcam_ssim', loss_Gcam_info[3], global_step=it_d)
        writer.add_scalar('loss_Gcam_lpips', loss_Gcam_info[4], global_step=it_d)

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

        writer.add_scalars('Image_Space_MSE', {'loss_mask_mse':loss_mask_info[0][0],'loss_grad_mse':loss_grad_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_mask_kl':loss_mask_info[1],'loss_grad_kl':loss_grad_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_mask_cosine':loss_mask_info[2],'loss_grad_cosine':loss_grad_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_mask_ssim':loss_mask_info[3],'loss_grad_ssim':loss_grad_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_mask_lpips':loss_mask_info[4],'loss_grad_lpips':loss_grad_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_w_info[1],'loss_c_cosine':loss_w_info[2]}, global_step=it_d)

        if iteration % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d_iter%d.png'%(iteration//30000,iteration%30000),nrow=n_row) # nrow=3
            heatmap=torch.cat((heatmap_1,heatmap_2))
            cam=torch.cat((cam_1,cam_2))
            grads = torch.cat((grad_1,grad_2))
            grads = grads.data.cpu().numpy() # [n,c,h,w]
            grads -= np.max(np.min(grads), 0)
            grads /= np.max(grads)
            torchvision.utils.save_image(torch.tensor(heatmap),resultPath_grad_cam+'/heatmap_%d.png'%(iteration),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(cam),resultPath_grad_cam+'/cam_%d.png'%(iteration),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(grads),resultPath_grad_cam+'/gb_%d.png'%(iteration),nrow=n_row)
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(iteration),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_mask_info: %s'%loss_mask_info,file=f)
                print('loss_grad_info: %s'%loss_grad_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('loss_Gcam_info: %s'%loss_Gcam_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
            if iteration % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d_iter%d.pth'%(iteration//30000,iteration%30000))
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_iter%d.pt'%iteration)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=210000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--experiment_dir', default=None)
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v2/stylegan2_cat256.pth')
    parser.add_argument('--config_dir', default='./checkpoint/biggan/256/biggan-deep-256-config.json') # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default=None)#'./result/StyleGAN1-car512-Aligned-modelV2/models/E_model_iter100000.pth'
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512) # BigGAN,z=128, PGGAN and StyleGANs = 512
    parser.add_argument('--mtype', type=int, default=2) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN00
    parser.add_argument('--start_features', type=int, default=64) # 16->1024 32->512 64->256 
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/StyleGAN2-CAT256-MisAligned-solveDetach&Clone-FronterImageVecvtors"
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