#test_baseline_alae.py


import sys
sys.path.append(".")
sys.path.append("..")

import torch.utils.data
import torchvision
import random
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from skimage.transform import resize
from PIL import Image
import logging

config_file='./configs/ffhq.yaml'
cfg=get_cfg_defaults()
cfg.merge_from_file(config_file)
cfg.freeze()

torch.cuda.set_device(0)
model = Model(
    startf=cfg.MODEL.START_CHANNEL_COUNT,
    layer_count=cfg.MODEL.LAYER_COUNT,
    maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
    latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
    truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
    truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
    mapping_layers=cfg.MODEL.MAPPING_LAYERS,
    channels=cfg.MODEL.CHANNELS,
    generator=cfg.MODEL.GENERATOR,
    encoder=cfg.MODEL.ENCODER)
model.cuda(0)
model.eval()
model.requires_grad_(False)

decoder = model.decoder
encoder = model.encoder
mapping_tl = model.mapping_d
mapping_fl = model.mapping_f
dlatent_avg = model.dlatent_avg

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

logger.info("Trainable parameters generator:")
count_parameters(decoder)

logger.info("Trainable parameters discriminator:")
count_parameters(encoder)

arguments = dict()
arguments["iteration"] = 0

model_dict = {
    'discriminator_s': encoder,
    'generator_s': decoder,
    'mapping_tl_s': mapping_tl,
    'mapping_fl_s': mapping_fl,
    'dlatent_avg': dlatent_avg
}

checkpointer = Checkpointer(cfg,
                            model_dict,
                            {},
                            logger=logger,
                            save=False)

extra_checkpoint_data = checkpointer.load()

model.eval()

layer_count = cfg.MODEL.LAYER_COUNT
im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
print(im_size)

def encode(x):
    Z, _ = model.encode(x, layer_count - 1, 1)
    Z = Z.repeat(1, model.mapping_f.num_layers, 1)
    return Z

def decode(x):
    layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32)
    coefs = torch.where(layer_idx < model.truncation_cutoff, 1.0 * ones, ones)
    # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
    return model.decoder(x, layer_count - 1, 1, noise=True)

#传入路径，输出图片及其重构
def make(paths):
    src = []
    for i,filename in enumerate(paths):
        img = Image.open(path + '/' + filename)
        img = img.resize((im_size,im_size))
        img = np.asarray(img)
        print(i,img.shape)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        factor = x.shape[2] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[2] == im_size
        src.append(x)

    with torch.no_grad():
        reconstructions = []
        for s in src:
            latents = encode(s[None, ...])
            reconstructions.append(decode(latents).cpu().detach().numpy())
    # print(len(src))
    # print(src[0].shape)
    # print(reconstructions[0].shape)
    return src, reconstructions


path = './styleganv1-generations'
paths = list(os.listdir(path))

# paths = paths[:256]
# chuncker_id = 0 # 0, 1, 2, 3

paths = paths[256:]
chuncker_id = 1 # 0, 1, 2, 3

src0, rec0 = make(paths)
src0 = [torch.tensor(array_np) for array_np in src0]
rec0 = [torch.tensor(array_np[0]) for array_np in rec0]
batched_images1 = torch.stack(src0, dim=0)
batched_images2 = torch.stack(rec0, dim=0)
print(batched_images1.shape)
print(batched_images2.shape)

for i,j in enumerate(batched_images2):
    j = j.unsqueeze(0)
    torchvision.utils.save_image(j*0.5+0.5,'./%s_rec.png'%str(i+256*chuncker_id).rjust(3,'0'))

#torch.save(batched_images1*0.5+0.5,'./ALAE-real-30.pt')
#torch.save(batched_images2*0.5+0.5,'./ALAE-rec-30.pt')