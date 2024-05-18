 import os
import math
import torch
import torchvision
import model.E.E as BE
import model.E.E_PG as BE_PG
import model.E.E_BIG as BE_BIG
from model.utils.custom_adam import LREQAdam
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

in_nodes = 512*18
out_nodes = 512

from model.utils.net import MappingBlock
class MapModel(torch.nn.Module):
    def __init__(self, in_nodes=in_nodes, out_nodes=out_nodes, mapping_fmaps=512):
        super().__init__()
        self.block1 = MappingBlock(in_nodes, out_nodes, lrmul=0.01)
    def forward(self, z, coefs_m=0):
        x = self.block1(z)
        #if self.buffer1 is not None:
        #    x = torch.lerp(self.buffer1.data, x, coefs_m) # avg + (styles-avg) * coefs
        return x

def train(tensor_writer = None, args = None):
    type = args.mtype

    w1 = torch.load(args.img_dir)
    print(w1.shape)

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
        Gm.cuda()

        E = MapModel().to(device)
        z = torch.randn(args.batch_size,512).cuda()
        z.requires_grad = True
        w2 = torch.randn(args.batch_size,18,512).cuda()
        w2.requires_grad = True

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

    E_optimizer = LREQAdam([{'params': w2},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = args.batch_size
    it_d = 0
    for iteration in range(0,args.iterations):
        if type == 1:
                #w1 = Gm(z,coefs_m=coefs.cuda()) #[batch_size,18,512]
                imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2))
                imgs2 = Gs.forward(w2,int(math.log(args.img_size,2)-2)) # True

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
            z = truncated_noise_sample(truncation=0.4, batch_size=batch_size, seed=iteration%args.iterations)
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

#Latent-Vectors
## w
        loss_w, loss_w_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_w.backward()
        E_optimizer.step()

        print('ep_%d_iter_%d'%(iteration//args.iterations,iteration%args.iterations))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]')
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)

        if iteration % 100 == 0:
            #inference:
            # if type == 1:
            #     with torch.no_grad():
            #         imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2))
            #         imgs2 = Gs.forward(w2,int(math.log(args.img_size,2)-2)) 
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d_iter%d.jpg'%(iteration//args.iterations,iteration%args.iterations),nrow=n_row) # nrow=3
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(iteration),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
            if iteration % 500 == 0:
                torch.save(z, resultPath1_2+'/E_model_ep%d_iter%d.pth'%(iteration//args.iterations,iteration%args.iterations))
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_iter%d.pt'%iteration)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--iterations', type=int, default=3_0001) # epoch = iterations//30000
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v1/ffhq1024/') #None  ./checkpoint/stylegan_v1/ffhq1024/ or ./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth or ./checkpoint/biggan/256/G-256.pt
    parser.add_argument('--config_dir', default='./checkpoint/biggan/256/biggan-deep-256-config.json') # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default=None)
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512) # PGGAN , StyleGANs are 512. BIGGAN is 128
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)  # 16->1024 32->512 64->256
    parser.add_argument('--img_dir', default='./checkpoint/styleGAN_v1_w/face_w/00.pt') # pt or directory
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/w2z_styleganv1_img2w_id00"
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

    # if os.path.isdir(args.img_dir): # img_file
    #     img_list = os.listdir(args.img_dir)
    #     img_list.sort()
    #     img_tensor_list = [imgPath2loader(args.img_dir+i,size=args.img_size) for i in img_list]
    #     imgs1 = torch.stack(img_tensor_list, dim = 0).to(device)
    # else: # pt
    #     imgs1 = torch.load(args.img_dir)
    # #imgs1 = imgs1*2-1 # [0,1]->[-1,1]
    # w2 = imgs1
    # print(w2.shape)

    train(tensor_writer=writer, args = args)
