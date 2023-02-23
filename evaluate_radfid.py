"""
Performs six perturbations on data so that the correlation between RadFID and human perceptual judgment could be evaluated.
"""
# Imports
import argparse
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import transform
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, help='Image directory to be changed', required=True)
parser.add_argument('--out_dir', type=str, help='Directory to save images to', required=True)
parser.add_argument('--alpha', type=float, help='Parameter for perturbations', required=True)
parser.add_argument('--perturbation', type=str, help='Type of perturbation: gn for Gaussian noise, gb for Gaussian blur,'
                                                    'bb for black boxes, s for swirl, spn for salt & pepper noise,'
                                                    '& c for contamination')
parser.add_argument('--cont_dir', type=str,
                    help='Directory that contains the images to contaminate the distribution with')
args = parser.parse_args()
input_dir = args.in_dir
output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'class0'))
alpha = args.alpha
p = args.perturbation
cont_dir = args.cont_dir

# Functions
def gaussian_noise(img, a):
    """
    Adds gaussian noise to image with strength a.
    Inputs:
      img (numpy array): image to which Gaussian noise will be added
      a (float): amount of noise to add according the linear combination
                 (1-a)*image + a*noise
    Returns:
      perturbed image
    """
    N = np.random.normal(size=img.shape)
    N += abs(N.min())
    N /= N.max()
    N *= 255.
    out_img = (1-a)*img + a*N
    out_img = np.uint8(np.clip(out_img, 0, 255))
    return out_img

def black_boxes(img, a):
    """
    Adds five random black boxes to and image.
    Inputs:
      img (numpy array): image to which the black boxes will be added
      a (float): the porportion of the image that a box should take up
    Returns:
      perturbed image
    """
    height, width = img.shape[0], img.shape[1]
    x = np.random.randint(0, height-int(a*height), 5)
    y = np.random.randint(0, width-int(a*width), 5)
    for i in range(5):
        img[x[i]:x[i]+size,y[i]:y[i]+size] = 0.
    return img

def salt_pepper_noise(img, a):
    """
    Adds salt and pepper noise to an image.
    Inputs:
      img (numpy array): image to which the noise will be added
      a (float): porportion of pixels to be affected
    Returns:
      perturbed image
    """
    height, width = img.shape[0], img.shape[1]
    num_pixels = int(height*width*a)
    x = np.random.randint(0, height, num_pixels)
    y = np.random.randint(0, width, num_pixels)
    for i in range(num_pixels):
        color = np.random.choice([0., 255.])
        img[x[i], y[i]] = color
    return img

def image_contamination(files, a):
    """
    Replaces images in one distribution with images from another distribution.
    Inputs:
      files (list): list of files in the original distribution
      a (float): porportion of images to be changed
    """
    num_imgs = int(len(files)*a)
    to_change = np.random.choice(files, num_imgs)
    msd_files = os.listdir(cont_dir)
    to_cont = np.random.choice(msd_files, num_imgs)
    count = 0
    for i, file in tqdm(enumerate(files)):
        if file in to_change:
            Image.open(os.path.join(cont_dir, to_cont[count])).save(os.path.join(output_dir, 'class0', file))
            count += 1
        else:
            Image.open(os.path.join(input_dir, file)).save(os.path.join(output_dir, 'class0', file))

# Main code.
if p == 'c':
    image_contamination(os.listdir(input_dir), alpha)
else:
    for file in tqdm(os.listdir(input_dir)):
        img = np.array(Image.open(os.path.join(input_dir, file)))
        if p == 'gn':
            Image.fromarray(gaussian_noise(img, alpha)).save(os.path.join(output_dir, 'class0', file))
        elif p == 'gb':
            Image.fromarray(ndimage.gaussian_filter(img, alpha)).save(os.path.join(output_dir, 'class0', file))
        elif p == 'bb':
            Image.fromarray(black_boxes(img, alpha)).save(os.path.join(output_dir, 'class0', file))
        elif p == 's':
            out_img = transform.swirl(img, strength=alpha, radius=256) * 255.
            Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(output_dir, 'class0', file))
        elif p == 'spn':
            Image.fromarray(salt_pepper_noise(img, alpha)).save(os.path.join(output_dir, 'class0', file))
        elif p == 'flip':
            if np.random.random() <  0.5:
                img = np.flip(img, 1)
            Image.fromarray(img).save(os.path.join(output_dir, 'class0', file))
