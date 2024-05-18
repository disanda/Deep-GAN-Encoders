# testing

import os
import skimage
import lpips
import torch
import torchvision
from PIL import Image


save_path1 = './styleganv1-generations/'
save_path2 = './MTV-rec/'

loss_mse = torch.nn.MSELoss()
loss_lpips = lpips.LPIPS(net='vgg')

def cosineSimilarty(imgs1_cos,imgs2_cos):
    values = imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) # [0,1]
    return values

def metrics(img_tensor1,img_tensor2):

    psnr = skimage.measure.compare_psnr(img_tensor1.float().numpy().transpose(1,2,0), img_tensor2.float().numpy().transpose(1,2,0), 255) #range:[0,255]

    ssim = skimage.measure.compare_ssim(img_tensor1.float().numpy().transpose(1,2,0), img_tensor2.float().numpy().transpose(1,2,0), data_range=255, multichannel=True)#[h,w,c] and range:[0,255]

    mse_value = loss_mse(img_tensor1,img_tensor2).numpy() #range:[0,255]

    lpips_value = loss_lpips(img_tensor1.unsqueeze(0)/255.0*2-1,img_tensor2.unsqueeze(0)/255.0*2-1).mean().detach().numpy() #range:[-1,1]

    cosine_value = cosineSimilarty(img_tensor1.view(-1)/255.0*2-1,img_tensor2.view(-1)/255.0*2-1).numpy() #range:[-1,1]

    print('-------------')
    print('psnr:',psnr)
    print('-------------')
    print('ssim:',ssim)
    print('-------------')
    print('lpips:',lpips_value)
    print('-------------')
    print('mse:',mse_value)
    print('-------------')
    print('cosine:',cosine_value)

    return psnr, ssim, lpips_value, mse_value, cosine_value

#--------文件夹内的图片转换为tensor:[n,c,h,w]------------------
img_size = 512
#PIL 2 Tensor
transform = torchvision.transforms.Compose([
        #torchvision.transforms.CenterCrop(160),
        torchvision.transforms.Resize((img_size,img_size)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#-----------------------------------------metric imgs_tensor-------------------------------
n = 0
psnr_values = 0
ssim_values = 0
lpips_values = 0
mse_values = 0
cosine_values = 0

imgs_path1 = [os.path.join(save_path1, f) for f in os.listdir(save_path1) if f.endswith(".png") or f.endswith(".jpg")]
imgs_path2 = [os.path.join(save_path2, f) for f in os.listdir(save_path2) if f.endswith(".png") or f.endswith(".jpg")]

for i,j in zip(imgs_path1,imgs_path2):
    print(i, 'vs.', j)
    img1 = Image.open(i).convert("RGB")
    img1 = transform(img1)
    img2 = Image.open(j).convert("RGB")
    img2 = transform(img2)

    img1 = img1*255.0
    img2 = img2*255.0

    n = n + 1

    psnr, ssim, lpips_value, mse_value, cosine_value = metrics(img1,img2)
    psnr_values +=psnr
    ssim_values +=ssim
    lpips_values +=lpips_value
    mse_values +=mse_value
    cosine_values +=cosine_value

    print('img_num:%d--psnr:%f--ssim:%f--mse_value:%f--lpips_value:%f--cosine_value:%f'\
        %(n,psnr_values/n, ssim_values/n, mse_values/n, lpips_values/n, cosine_values/n))
# if imgs_tensor1 = imgs_tensor2: -psnr: inf or 88.132626(with 1e-3) --ssim:1.000000--lpips_value:0.000000--mse_value:0.000000--cosine_value:1.000001

# imgs_path1 = [os.path.join(save_path1, f) for f in os.listdir(save_path1) if f.endswith(".png")]
# images1 = []
# for idx, image_path in enumerate(imgs_path1):
#     print(image_path)
#     img = Image.open(image_path).convert("RGB")
#     img = transform(img)
#     images1.append(img)
# imgs_tensor1 = torch.stack(images1, dim=0)


# imgs_path2 = [os.path.join(save_path2, f) for f in os.listdir(save_path2) if f.endswith(".png")]
# images2 = []
# for idx, image_path in enumerate(imgs_path2):
#     print(image_path)
#     img = Image.open(image_path).convert("RGB")
#     img = transform(img)
#     images2.append(img)
# imgs_tensor2 = torch.stack(images2, dim=0)

# if len(imgs_tensor1) != len(imgs_tensor2):
#     print('error: 2 comparing numbers are not equal!')

