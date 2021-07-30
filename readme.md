

# MTV-TSA: Multi-Type Vectors Joint Two-Scale Attentions for Image Embedding with Latent Representation.

![Python 3.7.3](https://img.shields.io/badge/python-3.7.3-blue.svg?style=plastic)
![PyTorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-blue.svg?style=plastic) 
![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=plastic)

  <img src="./images/cxx1.gif" width = "128" height = "128" alt="cxx1"  />  <img src="./images/cxx2.gif" width = "128" height = "128" alt="cxx2"  />  <img src="./images/msk.gif" width = "128" height = "128" alt="msk" />   <img src="./images/dy.gif" width = "128" height = "128" alt="dy" />  <img src="./images/zy.gif" width = "128" height = "128" alt="zy" /> 


>This is the official code release for MTV-TSA: Multi-Type Vectors Joint Two-Scale Attentions for Image Embedding with Latent Representation. The code contains a set of encoders for matching  pre-trained GANs (PGGAN, StyleGANv1, StyleGANv2, BigGAN)  via multi-scale vectors with two-scale attentions.


##  Usage

- training encoder with center attentions (align image)

> python E_align.py

- training encoder with Gram-based attentions (mis-align image)

> python E_mis_align.py

- embedding real image to latent space (using StyleGANv1 and w)

> python embedding_img.py

- discovering attribute directions with latent space : embedded_img_processing.py

Note: Pre-trained Model should be download first !

## Metric

- validate performance (Pre-trained GANs and baseline)

  1. using generations.py to generate reconstructed images (generate GANs images if needed)
  2. Files in directory "./baseline/" could help you to quickly format images and latent vectors (w).
  3. Put  comparing images to different files, and run comparing-baseline.py


- ablation study : looking from  ./ablations-study/


## Setup

###   Encoders

- Case 1: Training most pre-trained GANs with encoders. 
at './model/E/E.py' (quickly converge for reconstructed GANs' image)
- Case 2: Training StyleGANv1 on FFHQ for ablation study and real face image process
at './model/E/E_Blur.py'  (margin blur and more GPU memory)

###   Pre-Trained GANs
> note: put pre-trained GANs weight file at checkpoint directory
- StyleGAN_V1 (should contain 3 files: Gm, Gs, center-tensor):
  - Cat 256:
    - ./checkpoint/stylegan_V1/cat/cat256_Gs_dict.pth
    - ./checkpoint/stylegan_V1/cat/cat256_Gm_dict.pth
    - ./checkpoint/stylegan_V1/cat/cat256_tensor.pt
  - Car 256: same above
  - Bedroom 256:
- StyleGAN_V2 (Only one files : pth):
  - FFHQ 1024:
    - ./checkpoint/stylegan_V2/stylegan2_ffhq1024.pth
- PGGAN ((Only one files : pth)): 
  - Horse 256:
    - ./checkpoint/PGGAN/
- BigGAN (Two files : model as .pt and config as .json ):
  - Image-Net 256:
    - ./checkpoint/biggan/256/G-256.pt
    - ./checkpoint/biggan/256/biggan-deep-256-config.json

###  Options and Setting

> note: different GANs  should choose different pre-trained modle path 

-  choose --mtype for StyleGANv1=1, StyleGANv2=2, PGGAN=3, BIGGAN=4
-  choose Encoder start_features (--z_dim) carefully, the value is 16->1024x1024, 32->512x512, 64->256x256
-  if go on training, set --checkpoint_dir_E which path save pre-trained Encoder model
-  --checkpoint_dir_GAN is needed, StyleGANv1 is a directory(contains 3 filers: Gm, Gs, center-tensor) , others are file path (.pth or .pt)
```python
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
    parser.add_argument('--start_features', type=int, default=16)  # 16->1024 32->512 64->256
```

## Pre-trained Model

- We offered StyleGANv1-(FFHQ1024,  Cars512, Cats256) models (contain 3 files Gm, Gs, center-tensor), and our corresponding encoer here:

- We  offered BigGAN 256 encoder here:
- We  offered StyleGANv1 FFHQ encoder here (for real-image embedding and process):
- We  offered Cat256 StyleGANv2 (Grad-Cam based Case1 and Case2) here:

- If needed, we will offer other pre-trained model in future. You can also get pre-trianed GANs model in below  Acknowledgements. (The code could Directly adapt PGGAN, StyleGANv2 and BigGAN-pytorch pre-trained models)


##  Acknowledgements

Pre-trained GANs:

> StyleGANv1: https://github.com/podgorskiy/StyleGan.git, 
> ( Converting  code for official pre-trained model  is here: https://github.com/podgorskiy/StyleGAN_Blobless.git)
> StyleGANv2 and PGGAN: https://github.com/genforce/genforce.git
> BigGAN: https://github.com/huggingface/pytorch-pretrained-BigGAN

Comparing Works:

> In-Domain GAN: https://github.com/genforce/idinvert_pytorch
> pSp: https://github.com/eladrich/pixel2style2pixel
> ALAE: https://github.com/podgorskiy/ALAE.git

Ratelted Works:

> Grad-CAM & Grad-CAM++: https://github.com/yizt/Grad-CAM.pytorch
> SSIM Index: https://github.com/Po-Hsun-Su/pytorch-ssim

We express our thanks to above authors.

## License

The code of this repository is released under the [Apache 2.0](LICENSE) license.<br>
The directory `netdissect` is a derivative of the [GAN Dissection][gandissect] project, and is provided under the MIT license.<br>
The directories `models/biggan` and `models/stylegan2` are provided under the MIT license.


## BibTeX