# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""
# invert.py
# python test-baseline-indomainG.py 'styleganinv_ffhq256' './styleganv1-generations-512/'
import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image

import torch
import torchvision

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('image_list', type=str,
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      logger=logger)
  image_size = inverter.G.resolution

  # Load image list.
  # logger.info(f'Loading image list.')
  # image_list = []
  # with open(args.image_list, 'r') as f:
  #   for line in f:
  #     image_list.append(line.strip())

  path = args.image_list # 文件路径
  paths = list(os.listdir(path))
  image_list = []
  for i in paths:
    image_list.append(path + '/' + i)


  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Invert images.
  logger.info(f'Start inversion.')
  latent_codes = []
  codes_rec = []
  codes_inv = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    image_path = image_list[img_idx]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = resize_image(load_image(image_path), (image_size, image_size))
    code, viz_results = inverter.easy_invert(image, num_viz=args.num_results)
    latent_codes.append(code)
    #save_image(f'{output_dir}/{image_name}_ori.png', image)
    save_image(f'{output_dir}/r/{image_name}_enc.png', viz_results[1])
    save_image(f'{output_dir}/i/{image_name}_inv.png', viz_results[-1])
    visualizer.set_cell(img_idx, 0, text=image_name)
    visualizer.set_cell(img_idx, 1, image=image)
    codes_rec.append(torch.tensor(viz_results[1],dtype=float)/255.0) # cv2->pytorch
    codes_inv.append(torch.tensor(viz_results[-1],dtype=float)/255.0)
    for viz_idx, viz_img in enumerate(viz_results[1:]):
      visualizer.set_cell(img_idx, viz_idx + 2, image=viz_img)

  # Save results.
  #os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/inverted_codes.npy', np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')

  # codes_rec_tensor_ = torch.stack(codes_rec, dim=0)
  # codes_rec_tensor = codes_rec_tensor_.permute(0,3,1,2) # cv2->pytorch
  # codes_inv_tensor_ = torch.stack(codes_inv, dim=0)
  # codes_inv_tensor = codes_inv_tensor_.permute(0,3,1,2)
  # torch.save(codes_rec_tensor,'./indomain-rec32.pt')
  # torch.save(codes_inv_tensor,'./indomain-inv32.pt')
  # torchvision.utils.save_image(codes_rec_tensor,'./indomain_images_rec.png',nrow=5)
  # torchvision.utils.save_image(codes_inv_tensor,'./indomain_images_inv.png',nrow=5)

if __name__ == '__main__':
  main()
