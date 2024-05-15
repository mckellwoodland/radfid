"""
Extracts and saves features from the penultimate layer of a PyTorch pre-trained model.
"""

#Imports
import argparse
import cv2
import os
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from Swin-Transformer-main.models.swin_transformer import SwinTransformer

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-i', '--img_dir', type=str, required=True, help='Specify the path to the folder that contains the images to be embedded.')
required.add_argument('-f', '--feature_dir', type=str, required=True, help='Specify the path to the folder where the features should be saved. \
                                                                            This folder will be further subdivided by feature extraction architecture automatically.')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-a', '--architecture', type=str, default='InceptionV3', help='Specify which feature extraction architecture to use. \
                                                                                     Options: DenseNet121, InceptionV3, ResNet50, SwinT_rin, SwinT_img2rin. \
                                                                                     Defaults to InceptionV3.')
optional.add_argument('-g', '--gpu_node', type=str, default="0", help='Specify the GPU node. \
                                                                       Defaults to 0.')
optional.add_argument('-m', '--model_dir', type=str, default="models/RadImageNet_pytorch/", help="Specify the path to the folder that contains the RadImageNet pre-trained models in PyTorch. \
                                                                                                  Defaults to models/RadImageNet_pytorch")
optional.add_argument('-b', '--batch_size', type=int, default=32, help='Specify the batch size for inference.')
args = parser.parse_args()
if not os.path.exists(os.path.join(args.feature_dir, f'{args.architecture}_PyTorch')):
    os.mkdir(os.path.join(args.feature_dir, f'{args.architecture}_PyTorch'))

# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_node

# Variables
model_paths = {"DenseNet121": "DenseNet121.pt",
               "InceptionV3": "InceptionV3.pt",
               "ResNet50": "ResNet50.pt",
               "SwinT_rin": "rin_swintf.pth",
               "SwinT_img2rin": "img2rin_swintf.pth"}
if args.architecture == "SwinT_rin":
    with open(os.path.join(args.model_dir, "rin_config.json"),"r") as f:
        config = yaml.safe_load(f)
elif args.architecture == "SwinT_img2rin":
    with open(os.path.join(args.model_dir, "img2rin_config.json"),"r") as f:
        config = yaml.safe_load(f)
models_init = {"DenseNet121": models.densenet121(weights=None),
          "InceptionV3": models.inception_v3(init_weights=False, aux_logits=False),
          "ResNet50": models.resnet50(weights=None),          
          "SwinT_rin": SwinTransformer(**config),
          "SwinT_img2rin": SwinTransformer(**config)}

# Classes
class Backbone(nn.Module):
    def __init__(self, arch):
        super().__init__()
        base_model = models_init[arch]
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])

    def forward(self, x):
        return self.backbone(x)

class createDataset(Dataset):
    def __init__(self, img_dir):
        self.files = os.listdir(img_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.files[index]
        image = cv2.imread(image)
        image = (image-127.5)*2 / 255
        image = cv2.resize(image,(224,224))
        #image = np.transpose(image,(2,0,1))
        if self.transform is not None:
            image = self.transform(image)
        filename = self.files[index]
        return {"image": image , "filename": filename}

# Main code.
if __name__ == "__main__":
    #backbone = Backbone(args.architecture)
    backbone = models.swin_t()
    backbone.load_state_dict(torch.load(os.path.join(args.model_dir, model_paths[args.architecture])))


