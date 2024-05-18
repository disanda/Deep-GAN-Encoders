# The file optimize E for inversion real-img to latent space W.
# You should set img_dir in args. Support img-file or img-tensor (W) dirrectlly
# one interation just optimize one img, you can run the file on multiple command-line.
# This type could make w.norm() to 1 (This is important to attribute edit)

import os
import math
import torch
import torchvision
import models.E.E_Blur as BE
from utils.custom_adam import LREQAdam
import utils.pytorch_ssim as pytorch_ssim
import lpips
import numpy as np
import tensorboardX
import argparse
from utils.training_utils import *
import collections
from collections import OrderedDict
# from model.stylegan1.net import Generator, Mapping #StyleGANv1
# import model.pggan.pggan_generator as model_pggan #PGGAN
# from model.biggan_generator import BigGAN #BigGAN
#from models.stylegan2.model import StyleGANv2_Generator_v1 # StyleGAN2v1
#import models.stylegan2_generator as model_v2 #StyleGANv2v3
from models.generators.stylegan2.stylegan2_wrap import StyleGAN2Generator # StyleGAN2v2


def train(tensor_writer = None, args = None, imgs_tensor = None):
    # type = args.mtype

    # model_path = args.checkpoint_dir_GAN
    # config_path = args.config_dir

    # Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
    # Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

    # Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    # Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

    # Gm.buffer1 = torch.load(model_path+'./center_tensor.pt')
    # const_ = Gs.const
    # const1_ = const_.repeat(args.batch_size,1,1,1).to(device)
    # const1 = const1_.detach().clone()
    # const1.requires_grad = False
    # layer_num = int(math.log(args.img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024 
    # layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
    # ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
    # coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

    # Gs.to(device)
    # Gm.eval()

    generator = StyleGAN2Generator(device, truncation= 0.7, use_w = True, checkpoint_path = args.checkpoint_dir_GAN)

    E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
    E.load_state_dict(torch.load(args.checkpoint_dir_E)) #strict=False
    #omit RGB layers EAEv2->MSVv2: 这个效果重构人脸要好一些(case1，即EAE)
    # if args.checkpoint_dir_E != None:
    #     E_dict = torch.load(args.checkpoint_dir_E,map_location=torch.device(device))
    #     new_state_dict = OrderedDict()
    #     for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
    #             new_state_dict[i1] = j2 
    #     E.load_state_dict(new_state_dict)
    E.to(device)

    writer = tensor_writer
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    batch_size = args.batch_size
    it_d = 0

    #optimize E
    if args.optimizeE == True:
        E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 


    num = imgs_tensor.shape[0]
    interval = args.batch_size
    w_all = []
    img_all = []
    loss_msiv_min = 100
    w_norm_min = 1000
    for g in range(0, num//interval): 
        imgs1 = imgs_tensor[g*interval : (g+1)*interval]
        if args.optimizeE == False:
            #const2, w1_ = E(imgs1)
            #w1 = w1_.detach()
            w1 = torch.randn(1,18,512).to(device)
            w1.requires_grad=True
            E_optimizer = LREQAdam([{'params': w1},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0)
            #E_optimizer  = torch.optim.Adam([{'params': w1}], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            E.eval()
        else:
            E.load_state_dict(torch.load(args.checkpoint_dir_E)) # if not this reload, the max num of optimizing images is about 5-6.
            # if args.checkpoint_dir_E != None:
            #     E_dict = torch.load(args.checkpoint_dir_E,map_location=torch.device(device))
            #     new_state_dict = OrderedDict()
            #     for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
            #             new_state_dict[i1] = j2 
            #     E.load_state_dict(new_state_dict)
            E_optimizer.state = collections.defaultdict(dict) # Fresh the optimizer state. E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 
        for iteration in range(0,args.iterations):
            if args.optimizeE == True:
                const2, w1 = E(imgs1)
            #imgs2 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
            imgs2  = generator(w1)

            ##Image Vectors
            #Image
            loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)

            #loss AT1
            imgs_medium_1 = imgs1[:,:,:,imgs1.shape[3]//8:-imgs1.shape[3]//8]#.detach().clone()
            imgs_medium_2 = imgs2[:,:,:,imgs2.shape[3]//8:-imgs2.shape[3]//8]#.detach().clone()
            loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)

            #loss AT2
            imgs_small_1 = imgs1[:,:,\
            imgs1.shape[2]//8+imgs1.shape[2]//32:-imgs1.shape[2]//8-imgs1.shape[2]//32,\
            imgs1.shape[3]//8+imgs1.shape[3]//32:-imgs1.shape[3]//8-imgs1.shape[3]//32]#.detach().clone()
            imgs_small_2 = imgs2[:,:,\
            imgs2.shape[2]//8+imgs2.shape[2]//32:-imgs2.shape[2]//8-imgs2.shape[2]//32,\
            imgs2.shape[3]//8+imgs2.shape[3]//32:-imgs2.shape[3]//8-imgs2.shape[3]//32]#.detach().clone()
            loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)

            E_optimizer.zero_grad()
            loss_msiv = loss_imgs + loss_medium*0.125*3 + loss_small*0.125*5# + w1.norm(p=2)*0.003 #这个比例看要优化的细节，dy的突出外围，眉毛的话突出中间
            #loss_msiv = loss_imgs #+ w1.norm(p=2)*0.003
            loss_msiv.backward(retain_graph=True) #retain_graph=True
            E_optimizer.step()

            ##Latent-Vectors
            const3, w2 = E(imgs2)
            #w
            loss_w, loss_w_info = space_loss(w1,w2,image_space = False)

            ## c1
            loss_c1, loss_c1_info = space_loss(const2,const3,image_space = False)
            # # ## c2
            # loss_c, loss_c2_info = space_loss(const1,const3,image_space = False)

            E_optimizer.zero_grad()
            loss_msLv = (loss_w+loss_c1)*0.01 +  w1.norm(p=2)*0.0003 # 0.0003 0.0001 看要什么效果，重视重构效果就降低这个w1.norm(), 重视语意效果就提高
            #loss_msLv = (loss_w +  w1.norm(p=2))*0.001
            loss_msLv.backward() # retain_graph=True
            E_optimizer.step()

            w1_flag = w1.norm().detach().clone()
            loss_msiv_flag = loss_msiv.detach().clone()
            if iteration>1000:
                if loss_msiv_min > loss_msiv_flag*1.03:
                    loss_msiv_min = loss_msiv_flag
                    torch.save(w1,resultPath1_2+'/id%d-iter%d-norm%f-imgLoss-min%f.pt'%(g, iteration, w1_flag, loss_msiv_flag))
                    test_img_min1 = torch.cat((imgs1[:batch_size],imgs2[:batch_size]))*0.5+0.5
                    torchvision.utils.save_image(test_img_min1, resultPath1_1+'/id%d_ep%d-norm%.2f-imgLoss-min%f.jpg'%(g, iteration, w1_flag, loss_msiv_flag),nrow=2)
                    with open(resultPath+'/loss_min.txt','a+') as f:
                        print('ep%d_iter%d_minImg%.5f_wNorm%f'%(g, iteration, loss_msiv_flag, w1_flag), file=f)

                if w_norm_min > w1_flag*1.05 :
                    w_norm_min = w1_flag
                    torch.save(w1,resultPath1_2+'/id%d-iter%d-norm-min%f-imgLoss%f.pt'%(g,iteration,w1.norm(),loss_msiv_min.item()))
                    test_img_min2 = torch.cat((imgs1[:batch_size],imgs2[:batch_size]))*0.5+0.5
                    torchvision.utils.save_image(test_img_min2, resultPath1_1+'/id%d_ep%d-norm-min%.2f-imgLoss%f.jpg'%(g, iteration, w1_flag, loss_msiv_flag),nrow=2)
                    with open(resultPath+'/loss_min.txt','a+') as f:
                        print('ep%d_iter%d_Img%.5f_wNorm-min%f'%(g, iteration, loss_msiv_flag, w1_flag), file=f)

            print('id_'+str(g)+'_____i_'+str(iteration))
            print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]')
            print('---------ImageSpace--------')
            print('loss_small_info: %s'%loss_small_info)
            print('loss_medium_info: %s'%loss_medium_info)
            print('loss_imgs_info: %s'%loss_imgs_info)
            print('---------LatentSpace--------')
            print('loss_w_info: %s'%loss_w_info)
            print('loss_c1_info: %s'%loss_c1_info)
            # print('loss_c2_info: %s'%loss_c2_info)
            print('w_norm: %s'%w1_flag)
            print('Img_loss: %s'%loss_msiv_flag)

            it_d += 1
            if iteration % 100 == 0:
                n_row = batch_size
                test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                torchvision.utils.save_image(test_img, resultPath1_1+'/id%d_ep%d-norm%.2f-imgLoss%f.jpg'%(g,iteration,w1_flag,loss_msiv_flag),nrow=2) # nrow=3
                with open(resultPath+'/Loss.txt', 'a+') as f:
                    print('id_'+str(g)+'_____i_'+str(iteration),file=f)
                    print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                    print('---------ImageSpace--------',file=f)
                    print('loss_small_info: %s'%loss_small_info,file=f)
                    print('loss_medium_info: %s'%loss_medium_info,file=f)
                    print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                    print('---------LatentSpace--------',file=f)
                    print('loss_w_info: %s'%loss_w_info,file=f)
                    print('loss_c1_info: %s'%loss_c1_info,file=f)
                    # print('loss_c2_info: %s'%loss_c2_info,file=f)
                    print('w_norm: %s'%w1_flag,file=f)
                    print('Img_loss: %s'%loss_msiv_flag,file=f)

                for i,j in enumerate(w1):
                    torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-w%d-norm%f-imgLoss%f.pt'%(g, i, iteration, w1_flag, loss_msiv_flag))
                # for i,j in enumerate(imgs2):
                #     torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-img%d.pt'%(g,i,iteration))
                    #torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%iteration)

        torchvision.utils.save_image(imgs2*0.5+0.5,writer_path+'/%s_rec.png'%str(g).rjust(5,'0'))
        w_all.append(w1[0])
        #img_all.append(imgs2[0])

    w_all_tensor = torch.stack(w_all, dim=0)
    #img_all_tensor = torch.stack(img_all, dim=0)
    torch.save(w_all_tensor, resultPath1_2+'/w_all_%d.pt'%g)
    #torch.save(img_all_tensor, resultPath1_2+'/img_all_%d.pt'%g)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=2001)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/generators/stylegan2-ffhq-config-f.pt') #None  ./checkpoint/stylegan_v1/ffhq1024/ or ./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth
    parser.add_argument('--config_dir', default=None) # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default='./checkpoint/encoder/E_blur_stylegan2_ufl.pth') # E_blur(case2)_styleganv1_FFHQ_state_dict.pth, styleGANv1_EAE_ep115000.pth
    parser.add_argument('--img_dir', default='./real_imgs/') # pt or directory
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)  # 16->1024 32->512 64->256
    parser.add_argument('--optimizeE', type=bool, default=True) # if not, optimize W directly
    args = parser.parse_args()

    if not os.path.exists('./realimg_embedding_result'): os.mkdir('./realimg_embedding_result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./realimg_embedding_result/7"
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
        img_tensor_list = [imgPath2loader(args.img_dir+i,size=args.img_size) for i in img_list]
        imgs1 = torch.stack(img_tensor_list, dim = 0).to(device)
    else: # pt
        imgs1 = torch.load(args.img_dir)
    imgs1 = imgs1*2-1 # [0,1]->[-1,1]

    train(tensor_writer=writer, args = args, imgs_tensor = imgs1 )
