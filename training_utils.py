import torch
import numpy as np
from scipy.stats import truncnorm
import metric.pytorch_ssim as pytorch_ssim
from torch.nn import functional as F
from PIL import Image
import torchvision

#img_path2tensor
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
def imgPath2loader(image_name,size):
    image = Image.open(image_name).convert('RGB')
    image = image.resize((size,size))
    image = loader(image)#.unsqueeze(0)
    return image.to(torch.float)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

def one_hot(x, class_count=1000):
    # 第一构造一个[class_count, class_count]的对角线为1的向量
    # 第二保留label对应的行并返回
    return torch.eye(class_count)[x,:]

def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def space_loss(imgs1,imgs2,image_space=True,lpips_model=None):
    loss_mse = torch.nn.MSELoss()
    loss_kl = torch.nn.KLDivLoss()
    ssim_loss = pytorch_ssim.SSIM()
    loss_lpips = lpips_model

    imgs1 = imgs1.contiguous()
    imgs2 = imgs2.contiguous()

    loss_imgs_mse_1 = loss_mse(imgs1,imgs2)
    loss_imgs_mse_2 = loss_mse(imgs1.mean(),imgs2.mean())
    loss_imgs_mse_3 = loss_mse(imgs1.std(),imgs2.std())
    loss_imgs_mse = loss_imgs_mse_1 #+ loss_imgs_mse_2 + loss_imgs_mse_3

    imgs1_kl, imgs2_kl = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
    loss_imgs_kl = loss_kl(torch.log(imgs2_kl),imgs1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
    loss_imgs_kl = torch.where(torch.isnan(loss_imgs_kl),torch.full_like(loss_imgs_kl,0), loss_imgs_kl)
    loss_imgs_kl = torch.where(torch.isinf(loss_imgs_kl),torch.full_like(loss_imgs_kl,1), loss_imgs_kl)

    imgs1_cos = imgs1.view(-1)
    imgs2_cos = imgs2.view(-1)
    loss_imgs_cosine = 1 - imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) #[-1,1],-1:反向相反，1:方向相同

    if imgs1.view(-1).shape[0] != imgs2.view(-1).shape[0]:
        print('error: vector1 dimentions are not equal to vector2 dimentions')
        return

    if image_space:
        while imgs1.shape[2] > 256:
            imgs1 = F.avg_pool2d(imgs1,2,2)
            imgs2 = F.avg_pool2d(imgs2,2,2)

    if  image_space:
        ssim_value = pytorch_ssim.ssim(imgs1, imgs2) # while ssim_value<0.999:
        loss_imgs_ssim = 1-ssim_loss(imgs1, imgs2)
    else:
        loss_imgs_ssim = torch.tensor(0)

    if image_space:
        loss_imgs_lpips = loss_lpips(imgs1,imgs2).mean()
    else:
        loss_imgs_lpips = torch.tensor(0)

    loss_imgs = 5*loss_imgs_mse + loss_imgs_kl + 3*loss_imgs_cosine + loss_imgs_ssim + 2*loss_imgs_lpips
    loss_info = [[loss_imgs_mse_1.item(),loss_imgs_mse_2.item(),loss_imgs_mse_3.item()], loss_imgs_kl.item(), loss_imgs_cosine.item(), loss_imgs_ssim.item(), loss_imgs_lpips.item()]
    return loss_imgs, loss_info