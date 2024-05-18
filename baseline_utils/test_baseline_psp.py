from argparse import Namespace
import time
import sys
import os
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")


from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp


## 处理编码图像
save_path = "./real-128imgs"

## 配置
experiment_type = 'ffhq_encode' 
#@param ['ffhq_encode', 'ffhq_frontalize', 'celebs_sketch_to_face', 'celebs_seg_to_face', 'celebs_super_resolution', 'toonify']

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/psp_ffhq_encode.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
# pprint.pprint(opts)

# update the training options
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'output_size' not in opts:
    opts['output_size'] = 1024


opts = Namespace(**opts)
net = pSp(opts)
net.eval()
device = 'cuda' # 'cuda'
net.cuda()
print('Model successfully loaded!')

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to(device).float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to(device).float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

image_paths = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".png") or f.endswith(".jpg")]
n_images = len(image_paths)

images = []
n_cols = np.ceil(n_images / 2)
#fig = plt.figure(figsize=(20, 4))
for idx, image_path in enumerate(image_paths):
    #ax = fig.add_subplot(2, n_cols, idx + 1)
    img = Image.open(image_path).convert("RGB")
    images.append(img)
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.imshow(img)
#plt.show()

img_transforms = EXPERIMENT_ARGS['transform']
transformed_images = [img_transforms(image) for image in images]

batched_images = torch.stack(transformed_images, dim=0)

#batched_images = torch.load('real-30.pt')

with torch.no_grad():
    tic = time.time()
    for i,j in enumerate(batched_images):
        j = j.unsqueeze(0)
        result_images = run_on_batch(j, net, latent_mask=None)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
        torchvision.utils.save_image(result_images*0.5+0.5,'./r/%s_pSp_Rec.jpg'%str(i).rjust(5,'0'))

# from IPython.display import display

# couple_results = []
# for original_image, result_image in zip(images, result_images):
    # result_image = tensor2im(result_image)
    # res = np.concatenate([np.array(original_image.resize((256, 256))),
    #                       np.array(result_image.resize((256, 256)))], axis=1)
    # res_im = Image.fromarray(res)
    # couple_results.append(res_im)
    # display(res_im)
    # #import matplotlib.pyplot as plt
    # #img = plt.imread('1.jpg')#读取图片
    # plt.imshow(res_im)#展示图片
    # plt.show()
# result_images = result_images*0.5+0.5
# torch.save(result_images,'./psp_w30.pt')
# torchvision.utils.save_image(result_images,'./batched_images2.png',nrow=5)

