#Embedded_ImageProcessing, Just for StyleGAN_v1 FFHQ

import numpy as np
import math
import torch
import torchvision
import model.E.E_Blur as BE
from model.stylegan1.net import Generator, Mapping #StyleGANv1

#Params
use_gpu = False
device = torch.device("cuda" if use_gpu else "cpu")
img_size = 1024
GAN_path = './checkpoint/stylegan_v1/ffhq1024/'
direction = 'eyeglasses' #smile, eyeglasses, pose, age, gender
direction_path = './latentvectors/directions/stylegan_ffhq_%s_w_boundary.npy'%direction
w_path = './latentvectors/faces/i3_cxx2.pt'

#Loading Pre-trained Model, Directions
Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(img_size,2)-1), latent_size=512, channels=3)
Gs.load_state_dict(torch.load(GAN_path+'Gs_dict.pth', map_location=device))

# E = BE.BE()
# E.load_state_dict(torch.load('./checkpoint/E/styleganv1.pth',map_location=torch.device('cpu')))

direction = np.load(direction_path) #[[1, 512] interfaceGAN
direction = torch.tensor(direction).float()
direction = direction.expand(18,512) 
print(direction.shape)

w = torch.load(w_path, map_location=device).clone().squeeze(0)
print(w.shape)

# discovering face semantic attribute dirrections 
bonus= 70 #bonus   (-10) <- (-5) <- 0 ->5 ->10
start= 0 # default 0, if not 0, will be bed performance
end= 3 # default 3 or 4. if 3, it will keep face features (glasses). if 4, it will keep dirrection features (Smile).
w[start:start+end] = (w+bonus*direction)[start:start+end]
#w = w + bonus*direction
w = w.reshape(1,18,512)
with torch.no_grad():
  img = Gs.forward(w,8) # 8->1024
torchvision.utils.save_image(img*0.5+0.5, './img_bonus%d_start%d_end%d.png'%(bonus,start,end))

## end=3 人物ID的特征明显，end=4 direction的特征明显, end>4 空间纠缠严重
#smile: bonue*100, start=0, end=4(end不到4作用不大,end或bonus越大越猖狂）
#glass: bonue*200, start=0, end=4(end超过6开始崩,bonus也不宜过大)
#pose: bonue*5-10, start=0, end=3