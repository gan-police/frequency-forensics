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

## Dataset preparation
We work with images of the size 128x128 pixels. Hence, the raw images from the LSUN/CelebA data set have to be cropped and/or resized to this size. To do this, run `freqdect.crop_celeba` or `freqdect.crop_lsun`, depending on the dataset. This will create a new folder with the transformed images. The FFHQ dataset is already distributed in the required image size.

Use the pretrained GAN-models to generate images.
In case of StyleGAN, there is only a pre-trained model generating images of size 1024x1024, so one has to resize the GAN-generated images to size 128x128 pixels, e.g. by inserting
```
PIL.Image.fromarray(images[0], 'RGB').resize((128, 128)).save(png_filename)
```
into the [ffhq-stylegan](https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/pretrained_example.py)

Store all images (cropped original and GAN-generated) in a separate subdirectories of a directory, i.e. the directory structure should look like this
```
source_data
 ├── A_original
 ├── B_CramerGAN
 ├── C_MMDGAN
 ├── D_ProGAN
 └── E_SNGAN
```
For the FFHQ case, we have only two subdirectories: `ffhq_stylegan/A_ffhq` and `ffhq_stylegan/B_stylegan`. The prefixes of the folders are important, since the directories get the labels in lexicographic order of their prefix, i.e. directory `A_...` gets label 0, `B_...` label 1, etc.

Now, to prepare the data sets run `freqdect.prepare_dataset` . It reads in the data set, splits them into a training, validation and test set, applies the specified transformation (to wavelet packets, log-scaled wavelet packets or just the raw image data) and stores the result as numpy arrays.

Afterwards run e.g.:
```shell
$ python -m freqdect.prepare_dataset ./data/ffhq_stylegan/ --log-packets
$ python -m freqdect.prepare_dataset ./data/ffhq_stylegan/
```
The data-set preperation script accepts additional arguments. For example it is possible
to change the sizes of the train, test or validation sets. For a list of all optional arguments, open the help page via the `-h` argument.

## Training the classifier
Now you should be able to train a classifier using for example:
```shell
$ python -m freqdect.train_classifier \
    --data-prefix ./data/source_data_packets \
    --calc-normalization \
    --features packets
```
This trains a regression classifier using default hyperparameters. The training, validation and test accuracy and loss values are stored in a file placed in a `log` folder. The state dict of the trained model is stored there as well. For a list of all optional arguments, open the help page via the `-h` argument.

## Evaluating the classifier
### Plotting the accuracies

To plot the accuracy results, run
```shell
$ python -m freqdect.plot_accuracy_results {shared, lsun, celeba} {regression, CNN} ...
```
For a list of all optional arguments, open the help page via the `-h` argument.

### Calculating the confusion matrix

To calculate the confusion matrix, run `freqdect.confusion_matrix`. For a list of all arguments, open the help page via the `-h` argument.

## ⚖️ Licensing

This project is licensed under the [GNU GPLv3 license](LICENSE)

## Acknowledgements

### 📖 Citation

> TBD

### 🙏 Support

This project has been supported by the following organizations (in alphabetical order):

- [Fraunhofer Institute for Algorithms and Scientific Computing (SCAI)](https://www.scai.fraunhofer.de)
- [Fraunhofer Cluster of Excellence Cognitive Internet Technologies (CCIT)](https://www.cit.fraunhofer.de/en.html)
- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

### 💰 Funding

This project has been funded by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | Young Faculty Award (PI: Benjamin M. Gyori) | W911NF2010255   |
