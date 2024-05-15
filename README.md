# RadFID

This repository calculates Fréchet distances (FDs) with [RadImageNet](https://www.radimagenet.com/) [1].

# Docker

Build Docker container
```
docker build -t radfid .
```

Run the docker container.
```
docker run --gpus all -it -v $(pwd):/workspace radfid
```

# Extract features

Before extracting features, you'll need to download the TensorFlow models [here](https://github.com/BMEII-AI/RadImageNet). You can then extract the features for the real and generated images. We extracted features for 50,000 generated images and the full real dataset.

```
usage: extract_features.py [-h] -i IMG_DIR -f FEATURE_DIR [-a ARCHITECTURE] [-d DATASET]
                           [-m MODEL_DIR] [-g GPU_NODE] [-s IMG_SIZE] [-b BATCH_SIZE]

Required Arguments:
  -i IMG_DIR, --img_dir IMG_DIR
                        Specify the path to the folder that contains the images to be embedded within
                        a folder labeled class0.
  -f FEATURE_DIR, --feature_dir FEATURE_DIR
                        Specify the path to the folder where the features should be saved. This folder
                        will be further subdivided by feature extraction architecture and dataset
                        automatically.

Optional Arguments:
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Specify which feature extraction architecture to use. Options: "IRV2",
                        "ResNet50", "DenseNet121", "InceptionV3". Defaults to "InceptionV3".
  -d DATASET, --dataset DATASET
                        Specify which dataset the feature extractor should be trained on. Options:
                        "RadImageNet", "ImageNet". Defaults to "ImageNet".
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Specify the path to the folder that contains the RadImageNet-pretrained models
                        in TensorFlow. Required if the dataset to be evaluated is RadImageNet.
  -g GPU_NODE, --gpu_node GPU_NODE
                        Specify the GPU node. Defaults to 0.
  -s IMG_SIZE, --img_size IMG_SIZE
                        Specify the height/width of the images. Defaults to 512.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Specify the batch size for inference.
```

# Calculate FDs

```
usage: fd.py [-h] -f1 FEAT_DIR1 -f2 FEAT_DIR2

Required Arguments:
  -f1 FEAT_DIR1, --feat_dir1 FEAT_DIR1
                        Specify the path to the folder that contains the first group of
                        embeddings.
  -f2 FEAT_DIR2, --feat_dir2 FEAT_DIR2
                        Specify the path to the folder that contains the second group of
                        embeddings.
```

# Citation

If you have found our work useful, we would appreciate a citation of our MICCAI 2024 early accept preprint.

```
@misc{woodland2024_fid_med,
      title={Feature Extraction for Generative Medical Imaging Evaluation: New Evidence Against an Evolving Trend}, 
      author={McKell Woodland and Austin Castelo and Mais Al Taie and Jessica Albuquerque Marques Silva and Mohamed Eltaher and Frank Mohn and Alexander Shieh and Suprateek Kundu and Joshua P. Yung and Ankit B. Patel and Kristy K. Brock},
      year={2024},
      eprint={2311.13717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# References

[1] X. Mei *et al.*, "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning," *Radiol. Artif. Intell.*, vol. 4, no. 5, pp. e210315, Jul. 2022, doi: 10.1148/ryai.210315.
