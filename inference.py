import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

def remove_background(image, matte):
  # obtain predicted foreground
  image = np.asarray(image)
  if len(image.shape) == 2:
    image = image[:, :, None]
  if image.shape[2] == 1:
    image = np.repeat(image, 3, axis=2)
  elif image.shape[2] == 4:
    image = image[:, :, 0:3]
  matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
  foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

  return Image.fromarray(np.uint8(foreground))

def get_background(path_to_background, image_size=(1440,960)):
    background = Image.open(path_to_background)
    if not background.size == image_size:
        background = background.resize(image_size)
    return background

def are_images_same_shape(image1, image2):
    return image1.shape == image2.shape

def combine_images(img1, img2, mask, mode='RGB'):
    if img1.mode != mode:
        img1 = img1.convert(mode)
    if img2.mode != mode:
        img2 = img2.convert(mode)
    if mask.mode != 'L':
        mask = mask.convert('L')

    # Combine the images using the mask
    comb_img = Image.composite(img1, img2, mask)
    return comb_img    


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path of input images')
    parser.add_argument('--output_path', type=str, help='path of output images')
    parser.add_argument('--ckpt_path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()
    
# define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(args.input_path, im_name))
        #print("read image from input_path")
        #print(args.input_path)

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        foreground_name = im_name.split('.')[0] + '.foreground.png'
        matte_array = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        matte_array.save(os.path.join(args.output_path, matte_name))
        print("saved matte")

        foreground = remove_background(Image.open(os.path.join(args.input_path, im_name)), matte_array)
        foreground.save(os.path.join(args.output_path, foreground_name))
        print("saved background")

    foreground_im_names = glob.glob(args.output_path + '/*.foreground.png')
    mask_im_names = []
    for im_name in os.listdir(args.output_path):
        if not im_name.startswith('.') and im_name not in foreground_im_names:
            mask_im_names.append(os.path.join(args.output_path, im_name.split('.')[0] + '.png'))
    backgrounds = [os.path.join(args.input_path, 'backgrounds', img_name.split('.')[0] + '.jpeg') for img_name in os.listdir(os.path.join(args.input_path, 'backgrounds'))]
    img_and_mask = list(zip(foreground_im_names, mask_im_names))
    print("background files: ", backgrounds)
    print("img_and_mask_files: ", img_and_mask)
    for for_img, mask_img in img_and_mask:
        fg_img = Image.open(for_img)
        mask = Image.open(mask_img)

        for each_background in backgrounds:
            out_file_name = for_img.split('.')[0].split('/')[-1] + '_' + each_background.split('.')[0].split('/')[-1] + '.combined.png'
            bg_img = Image.open(each_background)
            combined_image = combine_images(fg_img, bg_img, mask)
            combined_image.save(os.path.join(args.output_path, 'combined_images', out_file_name))
            print("saved combined image: ", out_file_name)
