<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

# Wavelet-Packet Powered Deepfake Image Detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the supplementary source code for our paper
Wavelet-Packet Powered Deepfake Image Detection,
which is currently under review.


## Installation
The latest code can be installed in development mode with:
```shell
$ pip install -e .
```

## Data sets:
We utilize three datasets which commonly appeard in previous work:
-  [FFHQ](https://github.com/NVlabs/ffhq-dataset)
-  [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
-  [LSUN bedroom](https://github.com/fyu/lsun)

## GAN Architectures:
We utilize pre-trained models from the following repositories:
-  [StyleGAN](https://github.com/NVlabs/stylegan)
-  [GANFingerprints](https://github.com/ningyu1991/GANFingerprints)

## Getting the FFHQ experiment to run:
Download the 128x128 pixel version of the FFHQ data set.
Insert a foor loop a random seed and code to resize i.e. 
``` PIL.Image.fromarray(images[0], 'RGB').resize((128, 128)).save(png_filename)```
into 
[ffhq-stylegan](https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/pretrained_example.py)
generate 70k images .
Store all images in a `data` folder. Use i.e
```
./data/source_data/A_ffhq
./data/source_data/B_stylegan
```

Afterwards run:
```shell
$ python -m freqdect.prepare_dataset ./data/source_data/ --packets
$ python -m freqdect.prepare_dataset ./data/source_data/ --raw
```
Now you should be able to train a classifier using
```shell
$ python -m freqdect.train_classifier --data-prefix ./data/source_data/ --calc-normalization
```

### Dataset preparation
We work with images of the size 128x128 pixels. Hence, the raw images from the LSUN/CelebA data set have to be cropped and/or resized to this size. To do this, run `freqdect.crop_celeba` or `freqdect.crop_lsun`, depending on the dataset. This will create a new folder with the transformed images. The FFHQ dataset is already distributed in the required image size.

Store all images (cropped original and GAN-generated) in a separate subdirectories of a directory, i.e. the directory structure should look like this
```
lsun_bedroom_200k_png
 ├── A_lsun
 ├── B_CramerGAN
 ├── C_MMDGAN
 ├── D_ProGAN
 └── E_SNGAN
```
For the FFHQ case, we have only two subdirectories: `ffhq/A_ffhq` and `ffhq/B_stylegan`. The prefixes of the folders are important, since the directories get the labels in lexicographic order of their prefix, i.e. directory `A_...` gets label 0, `B_...` label 1, etc.

Now, to prepare the data sets run `freqdect.prepare_dataset`. It reads in the data set, splits them into a training, validation and test set, applies the specified transformation (to wavelet packets, log-scaled wavelet packets or just the raw image data) and stores the result as numpy arrays.

```shell
usage: prepare_dataset.py [-h] [--train-size TRAIN_SIZE] [--test-size TEST_SIZE] [--val-size VAL_SIZE] [--batch-size BATCH_SIZE] [--packets] [--log-packets] directory

positional arguments:
  directory             The folder with the real and gan generated image folders.

optional arguments:
  -h, --help            show this help message and exit
  --train-size TRAIN_SIZE
                        Desired size of the training subset of each folder. (default: 63_000).
  --test-size TEST_SIZE
                        Desired size of the test subset of each folder. (default: 5_000).
  --val-size VAL_SIZE   Desired size of the validation subset of each folder. (default: 2_000).
  --batch-size BATCH_SIZE
                        The batch_size used for image conversion. (default: 2048).
  --packets, -p         Save image data as wavelet packets.
  --log-packets, -lp    Save image data as log-scaled wavelet packets.
```
