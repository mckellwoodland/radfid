# RadFID

This repository provides an implementation of the Radiologic Fréchet Inception Distance (RadFID).

RadFID is the same as the Fréchet Inception Distance (FID) [1] except that the Inception network was pretrained on [RadImageNet](https://www.radimagenet.com/) [2].

This repository contains two main Python files:

1. `calc_radfid.py` can be used to compute the RadFID. 
It requires the [`fid.py`](https://github.com/bioinf-jku/TTUR/blob/master/fid.py) file from [1] and the `RadImageNet-InceptionV3_notop.h5` model from this [Google Drive](https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc/view) [2] to be in the same directory.
To use this code, your model must first generate 50,000 images and put them in a directory named `class0`.
`class0` must be the only directory in the encompassing directory.
All of the images for the real distribution should also be in a standalone `class0` repository.
The file contains the following arguments: `--gpu_node` to specify a GPU node to use, `--image_size` a single integer to denote the size of the images, `--img_dir_gen` the filepath to directory that contains the `class0` directory with 50,000 generated images in it, `--img_dir_real` the filepath to the directory that contains the `class0` directory with all real images in it, `--batch-size` the largest batch size possible according to computational limitations, and `--metric`.
`--metric` has two options: `radfid` or `fid` and defaults to `radfid`.

2. `evaluate_radfid.py` was the code used to test the correlation of RadFID with human perceptual judgment.
It has six possible pertubations under the argument `--pertubation`: `gn` for Gaussian noise, `gb` for Gaussian blur, `bb` for black boxes, `s` for swirl, `spn` for salt and pepper noise, and `c` for contamination.
Other arguments include: `--in_dir` the directory that contains the images to be perturbed, `--out_dir` the directory to save the perturbed images to, `--alpha` the strength of each pertubation, and `--cont_dir` the directory that contains the "contaminated" images in it.

# Docker

Build Docker container
```
docker build -t radfid .
```

Run the docker container.
```
docker run -it radfid /bin/bash
```

If you need access to files outside of the preprocessing folder, you can mount a directory.
```
docker run -it -v {DIR}:/workspace radfid /bin/bash
```
## References

[1] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017, [Online] Available: https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf

[2] X. Mei *et al.*, "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning," *Radiol. Artif. Intell.*, vol. 4, no. 5, pp. e210315, Jul. 2022, doi: 10.1148/ryai.210315.
