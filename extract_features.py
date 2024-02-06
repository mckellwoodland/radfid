"""
Extracts and saves features from the penultimate layer
"""

# Imports
import argparse
import os
import tensorflow
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-m', '--model_dir', type=str, help='Specify the path to the folder that contains the TensorFlow RadImageNet pretrained models.')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-a', '--architecture', type=str, default='InceptionV3', help='Specify which feature extraction architecture to use. \
                                                                                  Options: "IRV2", "ResNet50", "DenseNet121", "InceptionV3". \
                                                                                  Defaults to "InceptionV3".')
optional.add_argument('-d', '--dataset', type=str, default='RadImageNet', help='Specify which dataset the feature extractor should be trained on. \
                                                                                Options: "RadImageNet", "ImageNet". \
                                                                                Defaults to "RadImageNet".')
optional.add_argument('-g', '--gpu_node', type=int, default=0, help="Specify the GPU node. \
                                                                     Defaults to 0.")
optional.add_argument('-s', '--img_size', type=int, default=512, help='Specify the height/width of the images. Defaults to 512.')
args = parser.parse_args()

# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_node

# Variables
model_paths = {"IRV2": "RadImageNet-IRV2-notop.h5",
               "ResNet50": "RadImageNet-ResNet50-notop.h5",
               "DenseNet121": "RadImageNet-DenseNet121-notop.h5",
               "InceptionV3": "RadImageNet-InceptionV3-notop.h5"}

# Functions
def get_compiled_model(model_dir, arch, data, size):
    """
    Compiles model in TensorFlow with given weights.
    Modified from https://github.com/BMEII-AI/RadImageNet/blob/main/acl/acl_train.py.
    Inputs:
        model_dir (str): Folder that contains the RadImageNet pretrained models.
        arch (str): Which architecture to use for the feature extractor.
                    Options: "IRV2", "ResNet50", "DenseNet121", "InceptionV3".
        data (str): Which dataset to use for the feature extractor.
                    Options: "RadImageNet", "ImageNet".
        size (int): Height/width of the images.

    Returns:
        Compiled model.
    """
    if data == "RadImageNet":
        weights = os.path.join(model_dir, model_paths[arch])
    else:
        weights = "imagenet"
    if arch == "IRV2":
        pass
    if args.metric == 'radfid':
        model_file = 'RadImageNet-InceptionV3_notop.h5'
        # Model setup and preprocessing follows the official RadImageNet GitHub: https://github.com/BMEII-AI/RadImageNet/blob/main/acl/acl_train.py (lines 84 and 160)
        model = applications.InceptionV3(weights=model_file, input_shape=(args.img_size, args.img_size, 3), include_top=False, pooling='avg')
        datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=applications.imagenet_utils.preprocess_input)
    elif args.metric == 'fid':
        model = applications.InceptionV3(input_shape=(args.img_size, args.img_size, 3), include_top=False, pooling='avg')
        datagen = image.ImageDataGenerator(preprocessing_function=applications.inception_v3.preprocess_input)

# Main code
model = get_compiled_model(args.model_dir, args.architecture, args.dataset, args.img_size)