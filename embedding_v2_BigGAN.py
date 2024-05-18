# The file optimize E for inverting real-img to latent space (getting Wy).
# Please refer to 186 below to set key args.
# Code-line: 205-206 for first step regularization parameters:  /beta and Norm_p

import os
import math
import torch
import lpips
import argparse
import collections
import torchvision
import numpy as np
import tensorboardX
from collections import OrderedDict
import metric.pytorch_ssim as pytorch_ssim
from training_utils import imgPath2loader, space_loss
from model.biggan_generator import BigGAN #BigGAN
from training_utils import *
from model.utils.biggan_config import BigGANConfig
import model.E.E_BIG as BE_BIG
from model.utils.custom_adam import LREQAdam
from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import torch.nn as nn

def train(tensor_writer = None, args = None, imgs_tensor = None):

    beta = args.beta
    rho  = args.norm_p

    model_path = './checkpoint/biggan/256/G-256.pt'
    config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
    config = BigGANConfig.from_json_file(config_file)
    generator = BigGAN(config).to(device)
    generator.load_state_dict(torch.load(model_path))
    
    # label/id/flag : bigGAN with imageNet_1K
    flag = np.array(30) # 30-frog, 22-eagle 7-cock 207-yellow dog 712- peiyangming
    print(flag)
    label = np.ones(args.batch_size)
    label = flag * label
    label = one_hot(label)
    conditions = torch.tensor(label, dtype=torch.float).cuda() # as label
    truncation = torch.tensor(0.4, dtype=torch.float).cuda()
    embed = generator.embeddings(conditions) # 1000 => z_dim: 128
    z = truncated_noise_sample(truncation=0.4, batch_size=args.batch_size, seed=args.iterations%30000)
    z = torch.tensor(z, dtype=torch.float).cuda()
    cond_vector = torch.cat((z, embed), dim=1) # 128->256
    #cond_vector.requires_grad=True
    
    #vgg16->Grad-CAM
    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)

    
    E = BE_BIG.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3, biggan=True).to(device)
    if args.checkpoint_dir_E is not None:
        E.load_state_dict(torch.load(args.checkpoint_dir_E, map_location=torch.device(device)))

    writer = tensor_writer
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    batch_size = args.batch_size
    it_d = 0

    #optimize E
    if args.optimizeE == True:
        E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0.0) #0.0003


    num = imgs_tensor.shape[0]
    interval = args.batch_size
    w_all = []
    img_all = [] 
    for g in range(0, num//interval): 
        imgs1 = imgs_tensor[g*interval : (g+1)*interval]
        if args.optimizeE == False:
            const1, w1_ = E(imgs1,cond_vector)
            w1 = w1_.detach()
            w1.requires_grad=True
            E_optimizer = LREQAdam([{'params': w1},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0)
        else:
            E.load_state_dict(torch.load(args.checkpoint_dir_E)) # if not this reload, the max num of optimizing images is about 5-6.
            E_optimizer.state = collections.defaultdict(dict) # Fresh the optimizer state. E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 
        loss_msiv_min = torch.tensor(0.)
        for iteration in range(0,args.iterations):
            if args.optimizeE == True:
                const1, w1 = E(imgs1,cond_vector)
            #imgs2 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
            imgs2, _=generator(w1, conditions, truncation)
            const2, w2 = E(imgs2,cond_vector)

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
            
            ##Image Vectors
            #Image
            loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)

#             #loss AT1
#             imgs_medium_1 = imgs1[:,:,:,imgs1.shape[3]//8:-imgs1.shape[3]//8]#.detach().clone()
#             imgs_medium_2 = imgs2[:,:,:,imgs2.shape[3]//8:-imgs2.shape[3]//8]#.detach().clone()
#             loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)
#             loss_medium, loss_medium_info = space_loss(mask_1.detach().clone(),mask_2.detach().clone(),lpips_model=loss_lpips)

#             #loss AT2
#             imgs_small_1 = imgs1[:,:,\
#             imgs1.shape[2]//8+imgs1.shape[2]//32:-imgs1.shape[2]//8-imgs1.shape[2]//32,\
#             imgs1.shape[3]//8+imgs1.shape[3]//32:-imgs1.shape[3]//8-imgs1.shape[3]//32]#.detach().clone()
#             imgs_small_2 = imgs2[:,:,\
#             imgs2.shape[2]//8+imgs2.shape[2]//32:-imgs2.shape[2]//8-imgs2.shape[2]//32,\
#             imgs2.shape[3]//8+imgs2.shape[3]//32:-imgs2.shape[3]//8-imgs2.shape[3]//32]#.detach().clone()
#             loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)

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

#             E_optimizer.zero_grad()
#             loss_msiv = loss_imgs + loss_medium*0 + loss_small*0
#             loss_msiv.backward(retain_graph=True) #retain_graph=True
#             E_optimizer.step()
            
            E_optimizer.zero_grad()
            loss_msiv = loss_imgs  + loss_mask + loss_Gcam
            loss_msiv.backward(retain_graph=True) # retain_graph=True
            E_optimizer.step()

            ##Latent-Vectors
            ## w
            loss_w, loss_w_info = space_loss(w1,w2,image_space = False)

            ## c1
            #loss_c1, loss_c1_info = space_loss(const2,const3,image_space = False)

            ## c2
            loss_c2, loss_c2_info = space_loss(const1,const2,image_space = False)

            E_optimizer.zero_grad()
            loss_msLv = loss_w*0.01 #  + loss_c2*0.01 #+ w1.norm(p=rho)*beta # 0.0003 0.0001 看要什么效果，重视重构效果就降低这个w1.norm(), 重视语意效果就提高
            loss_msLv.backward(retain_graph=True) # retain_graph=True
            E_optimizer.step()

            if iteration == args.iterations//2:
                loss_msiv_min = loss_msiv

            if loss_msiv_min > loss_msiv*1.05:
                loss_msiv_min = loss_msiv
                torch.save(w1,resultPath1_2+'/id%d-iter%d-norm%f-imgLoss-min%f.pt'%(g,iteration,w1.norm(),loss_msiv_min.item()))
                test_img_min1 = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                torchvision.utils.save_image(test_img_min1, resultPath1_1+'/id%d_ep%d-norm%.2f-imgLoss-min%f.jpg'%(g, iteration, w1.norm(), loss_msiv_min.item()),nrow=2)
                with open(resultPath+'/loss_min.txt','a+') as f:
                    print('ep%d_iter%d_minImg%.5f_wNorm%f'%(g,iteration,loss_msiv_min.item(),w1.norm()),file=f)

            # if w_norm_min > w1.norm()*1.05 :
            #     w_norm_min = w1.norm()
            #     torch.save(w1,resultPath1_2+'/id%d-iter%d-norm-min%f-imgLoss%f.pt'%(g,iteration,w1.norm(),loss_msiv_min.item()))
            #     test_img_min2 = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            #     torchvision.utils.save_image(test_img_min2, resultPath1_1+'/id%d_ep%d-norm-min%.2f-imgLoss%f.jpg'%(g, iteration, w1.norm(), loss_msiv_min.item()),nrow=n_row)
            #     with open(resultPath+'/loss_min.txt','a+') as f:
            #         print('ep%d_iter%d_Img%.5f_wNorm-min%f'%(g,iteration,loss_msiv_min.item(),w1.norm()),file=f)

            print('id_'+str(g)+'_____i_'+str(iteration))
            print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]')
            print('---------ImageSpace--------')
            print('loss_small_info: %s'%loss_mask_info)
            print('loss_medium_info: %s'%loss_Gcam_info)
            print('loss_imgs_info: %s'%loss_imgs_info)
            print('---------LatentSpace--------')
            print('loss_w_info: %s'%loss_w_info)
            # print('loss_c1_info: %s'%loss_c1_info)
            print('loss_c2_info: %s'%loss_c2_info)
            print('w_norm: %s'%w1.norm())
            print('Img_loss_min: %s'%loss_msiv_min.item())

            it_d += 1
            if iteration % 100 == 0:
                n_row = batch_size
                test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                torchvision.utils.save_image(test_img, resultPath1_1+'/id%d_ep%d-norm%.2f.jpg'%(g,iteration,w1.norm()),nrow=2) # nrow=3
                with open(resultPath+'/Loss.txt', 'a+') as f:
                    print('id_'+str(g)+'_____i_'+str(iteration),file=f)
                    print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                    print('---------ImageSpace--------',file=f)
                    print('loss_small_info: %s'%loss_mask_info,file=f)
                    print('loss_medium_info: %s'%loss_Gcam_info,file=f)
                    print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                    print('---------LatentSpace--------',file=f)
                    print('loss_w_info: %s'%loss_w_info,file=f)
                    # print('loss_c1_info: %s'%loss_c1_info,file=f)
                    print('loss_c2_info: %s'%loss_c2_info,file=f)
                    print('Img_loss: %s'%loss_msiv_min.item(),file=f)

                for i,j in enumerate(w1):
                    torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-w%d-norm%f.pt'%(g,i,iteration,w1.norm()))
                # for i,j in enumerate(imgs2):
                #     torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-img%d.pt'%(g,i,iteration))
                    #torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%iteration)

        torchvision.utils.save_image(imgs2*0.5+0.5,writer_path+'/%s_rec.png'%str(g).rjust(5,'0'))
        w_all.append(w1[0])
        img_all.append(imgs2[0])

    w_all_tensor = torch.stack(w_all, dim=0)
    img_all_tensor = torch.stack(img_all, dim=0)
    torch.save(w_all_tensor, resultPath1_2+'/w_all_%d.pt'%g)
    torch.save(img_all_tensor, resultPath1_2+'/img_all_%d.pt'%g)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=1501)
    parser.add_argument('--lr', type=float, default=0.0003) # better than 0.01 W:0.003, E:0.0003
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--img_dir', default='./bigGAN_inversion/id30/') # pt or directory
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--start_features', type=int, default=64)  # 16->1024 32->512 64->256
    parser.add_argument('--optimizeE', type=bool, default=True) # if not, optimize W directly True False
    parser.add_argument('--beta', type=float, default=10e-7)
    parser.add_argument('--norm_p', type=int, default=2)
    parser.add_argument('--checkpoint_dir_E', default='./result/BigGAN-256/models/E_model_ep30000.pth')
    args = parser.parse_args()

    result_path = './result_bigGAN_id30_GradCAM'
    if not os.path.exists(result_path): os.mkdir(result_path)
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = result_path+ "/mis_aligh_bigGAN_v1_opE/"
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

    if os.path.isdir(args.img_dir): # img_file
        img_list = os.listdir(args.img_dir)
        img_list.sort()
        img_tensor_list = [imgPath2loader(args.img_dir+i,size=args.img_size) for i in img_list \
        if i.endswith('jpg') or i.endswith('png')]
        imgs1 = torch.stack(img_tensor_list, dim = 0).to(device)
    else: # pt
        imgs1 = torch.load(args.img_dir)
    imgs1 = imgs1*2-1 # [0,1]->[-1,1]

    train(tensor_writer=writer, args = args, imgs_tensor = imgs1 )
