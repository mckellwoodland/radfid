"""
Extracts and saves features from the penultimate layer of a Keras pre-trained model.
"""

# Imports
import argparse
import numpy as np
import os
import tqdm
from tensorflow.keras import applications
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-i', '--img_dir', type=str, required=True, help='Specify the path to the folder that contains the images to be embedded within a folder labeled class0.')
required.add_argument('-f', '--feature_dir', type=str, required=True, help='Specify the path to the folder where the features should be saved. \
                                                                            This folder will be further subdivided by feature extraction architecture and dataset automatically.')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-a', '--architecture', type=str, default='InceptionV3', help='Specify which feature extraction architecture to use. \
                                                                                  Options: "IRV2", "ResNet50", "DenseNet121", "InceptionV3". \
                                                                                  Defaults to "InceptionV3".')
optional.add_argument('-d', '--dataset', type=str, default='ImageNet', help='Specify which dataset the feature extractor should be trained on. \
                                                                                Options: "RadImageNet", "ImageNet". \
                                                                                Defaults to "ImageNet".')
optional.add_argument('-m', '--model_dir', type=str, help='Specify the path to the folder that contains the RadImageNet-pretrained models in TensorFlow. \
                                                           Required if the dataset to be evaluated is RadImageNet.')
optional.add_argument('-g', '--gpu_node', type=str, default="0", help="Specify the GPU node. \
                                                                     Defaults to 0.")
optional.add_argument('-s', '--img_size', type=int, default=512, help='Specify the height/width of the images. Defaults to 512.')
optional.add_argument('-b', '--batch_size', type=int, default=32, help='Specify the batch size for inference.')
args = parser.parse_args()
if not os.path.exists(os.path.join(args.feature_dir, f'{args.architecture}')):
    os.mkdir(os.path.join(args.feature_dir, f'{args.architecture}'))
if not os.path.exists(os.path.join(args.feature_dir, f'{args.architecture}', f'{args.dataset}')):
    os.mkdir(os.path.join(args.feature_dir, f'{args.architecture}', f'{args.dataset}'))

# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_node

# Variables
model_paths = {"IRV2": "RadImageNet-IRV2_notop.h5",
               "ResNet50": "RadImageNet-ResNet50_notop.h5",
               "DenseNet121": "RadImageNet-DenseNet121_notop.h5",
               "InceptionV3": "RadImageNet-InceptionV3_notop.h5"}
models = {"IRV2": applications.InceptionResNetV2,
          "ResNet50": applications.ResNet50,
          "DenseNet121": applications.DenseNet121,
          "InceptionV3": applications.InceptionV3}

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
        assert os.path.exists(weights), f"{model_paths[arch]} does not exist at the specified path: {model_dir}."
    else:
        weights = "imagenet"
    return models[arch](weights=weights, 
                        input_shape=(size, size, 3), 
                        include_top=False, 
                        pooling='avg')

# Main code
if __name__ == "__main__":
    model = get_compiled_model(args.model_dir, 
                               args.architecture, 
                               args.dataset, 
                               args.img_size)
    datagen = image.ImageDataGenerator(rescale=1./255,
                                       preprocessing_function=imagenet_utils.preprocess_input)
    extract_gen = datagen.flow_from_directory(args.img_dir,
                                              batch_size=args.batch_size,
                                              class_mode=None,
                                              shuffle=False)
    n_imgs = len(os.listdir(os.path.join(args.img_dir, 'class0')))
    n_batches = n_imgs // args.batch_size + 1
    filenames = extract_gen.filenames
    for i in tqdm.tqdm(range(n_batches)):
        batch = extract_gen.next()
        features = model.predict(batch)
        for j in range(len(features)):
            feature = features[j]
            feature_file_path = os.path.join(args.feature_dir, f'{args.architecture}', f'{args.dataset}', filenames[j][7:-3] + 'npy')
            np.save(feature_file_path, features)