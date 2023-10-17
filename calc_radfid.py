"""
Calculates the RadFID.
"""
# Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import InceptionV3, inception_v3
import os
import argparse
import numpy as np
from tqdm import tqdm
from fid import calculate_frechet_distance

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_node', type=str, help='GPU node to utilize')
parser.add_argument('--image_size', type=int, help='image size')
parser.add_argument('--img_dir_gen', type=str, help='Generated image directory')
parser.add_argument('--img_dir_real', type=str, help='Real image directory')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--metric', type=str, help='radfid or fid', default='radfid')
parser.add_argument('--out_path', type=str, help='txt file to write out results to')
args = parser.parse_args()
gpu_node = args.gpu_node
image_size = args.image_size
img_dir_gen = args.img_dir_gen
img_dir_real = args.img_dir_real
batch_size = args.batch_size
metric = args.metric
out_path = args.out_path

# Environment
if gpu_node:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_node
    
# Functions
def calculate_activation_statistics(img_dir):
    """
    Calculates the mean and standard deviation of a set of images.
    Code mirrors that of the original implementation: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    Inputs:
        img_dir (str): the directory that contains the 'class0' directory. The 'class0' directory contains
                       the images to be evaluated.
    Returns:
        mu (float): mean
        sigma (float): standard deviation
    """
    n_imgs = len(os.listdir(os.path.join(img_dir, 'class0/')))
    n_batches = n_imgs // batch_size + 1
    generator = datagen.flow_from_directory(img_dir, target_size=(image_size, image_size),
                                            batch_size=batch_size, class_mode=None)
    pred_arr = np.empty((n_imgs, 2048))
    for i in tqdm(range(n_batches)):
        start = i*batch_size
        if start + batch_size < n_imgs:
            end = start + batch_size
        else:
            end = n_imgs
        batch = generator.next()
        act = model(batch)
        pred_arr[start:end] = act
        del batch
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

# Main code
if metric == 'radfid':
    model_file = 'RadImageNet-InceptionV3_notop.h5'
    model = InceptionV3(weights=model_file, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
    datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
elif metric == 'fid':
    model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
    datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)
mu1, sigma1 = calculate_activation_statistics(img_dir_gen)
mu2, sigma2 = calculate_activation_statistics(img_dir_real)
fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print(fid)
if out_path:
    with open(out_path, 'w') as f:
        f.write(f"The FID is {fid}")
