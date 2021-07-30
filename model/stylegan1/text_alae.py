from net import *
from model import Model
from launcher import run
from dataloader import *
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from PIL import Image
import PIL
from torchvision.utils import save_image

def draw_uncurated_result_figure(cfg, png, model, cx, cy, cw, ch, rows, lods, seed):
    print(png)
    N = sum(rows * 2**lod for lod in lods)
    images = []

    rnd = np.random.RandomState(5)
    for i in range(N):
        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
        samplez = torch.tensor(latents).float().cuda()
        image = model.generate(cfg.DATASET.MAX_RESOLUTION_LEVEL-2, 1, samplez, 1, mixing=True)
        images.append(image[0])

    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
    image_iter = iter(list(images))
    for col, lod in enumerate(lods):
        for row in range(rows * 2**lod):
            im = next(image_iter).cpu().numpy()
            im = im.transpose(1, 2, 0)
            im = im * 0.5 + 0.5
            image = PIL.Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8), 'RGB')
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
    canvas.save(png)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
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

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    decoder = nn.DataParallel(decoder)

    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
    with torch.no_grad():
        rnd = np.random.RandomState(0)
        latents = rnd.randn(1, 512)
        samplez = torch.tensor(latents).float().cuda()
        image = model.generate(8, 1, samplez, 1, mixing=True)
         ls   = model.encode(image,8,1)
         x    = model.decoder(x, 8, 1, noise=True)
        save_image(samplez,'1.png')
        save_image(x,'2.png')

if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-generations', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
