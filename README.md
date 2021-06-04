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
``` PIL.Image.fromarray(images[0], 'RGB').resize((128, 128)).save(png_filename)```
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

Afterwards run i.e.:
```shell
$ python -m freqdect.prepare_dataset ./data/ffhq_stylegan/ --log-packets
$ python -m freqdect.prepare_dataset ./data/ffhq_stylegan/
```
The data-set preperation script accepts additional arguments. For example it is possible
to change the sizes of the train, test or validation sets. For all options see:
```
usage: prepare_dataset.py [-h] [--train-size TRAIN_SIZE] [--test-size TEST_SIZE]
    [--val-size VAL_SIZE] [--batch-size BATCH_SIZE] [--packets]
    [--log-packets] directory

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

  Example: python -m freqdect.prepare_dataset ./data/source_data/ --packets
```

## Training the classifier
Now you should be able to train a classifier using for example:
```shell
$ python -m freqdect.train_classifier \
    --data-prefix ./data/source_data_packets \
    --calc-normalization \
    --features packets
```
This trains a regression classifier using default hyperparameters. The training, validation and test accuracy and loss values are stored in a file placed in a `log` folder. The state dict of the trained model is stored there as well.
For reference other options are:
```
usage: train_classifier.py [-h] [--features {raw,packets}]
                           [--batch-size BATCH_SIZE]
                           [--learning-rate LEARNING_RATE]
                           [--weight-decay WEIGHT_DECAY] [--epochs EPOCHS]
                           [--validation-interval VALIDATION_INTERVAL]
                           [--data-prefix DATA_PREFIX] [--nclasses NCLASSES]
                           [--seed SEED] [--model {regression,cnn,mlp}]
                           [--tensorboard] [--normalize MEAN [STD ...] |
                           --calc-normalization]

Train an image classifier

optional arguments:
  -h, --help            show this help message and exit
  --features {raw,packets}
                        the representation type
  --batch-size BATCH_SIZE
                        input batch size for testing (default: 512)
  --learning-rate LEARNING_RATE
                        learning rate for optimizer (default: 1e-3)
  --weight-decay WEIGHT_DECAY
                        weight decay for optimizer (default: 0)
  --epochs EPOCHS       number of epochs (default: 10)
  --validation-interval VALIDATION_INTERVAL
                        number of training steps after which the model is
                        tested on the validation data set (default: 200)
  --data-prefix DATA_PREFIX
                        shared prefix of the data paths (default:
                        ./data/source_data_packets)
  --nclasses NCLASSES   number of classes (default: 2)
  --seed SEED           the random seed pytorch works with.
  --model {regression,cnn,mlp}
                        The model type chosse regression or CNN. Default:
                        Regression.
  --tensorboard         enables a tensorboard visualization.
  --normalize MEAN [STD ...]
                        normalize with specified values for mean and standard
                        deviation (either 2 or 6 values are accepted)
  --calc-normalization  calculates mean and standard deviation used in
                        normalizationfrom the training data
```

## Evaluating the classifier
### Plotting the accuracies

To plot the accuracy results, run
```shell
$ python -m freqdect.plot_accuracy_results {shared, lsun, celeba} {regression, CNN} ...
```
For a list of all optional arguments, open the help page via the `-h` argument.

### Calculating the confusion matrix

To calculate the confusion matrix, run `freqdect.confusion_matrix`.

```
usage: confusion_matrix.py [-h] [--classifier-path CLASSIFIER_PATH] [--data DATA] [--model {regression,CNN}] [--features {raw,packets}] [--batch-size BATCH_SIZE] [--normalize MEAN [STD ...]] [--label-names LABEL_NAMES [LABEL_NAMES ...]]
                           [--plot] [--nclasses NCLASSES] [--generalized]

Calculate the confusion matrix

optional arguments:
  -h, --help            show this help message and exit
  --classifier-path CLASSIFIER_PATH
                        path to classifier model file
  --data DATA           path of folder containing the test data
  --model {regression,CNN}
                        The model type. Choose regression or CNN.
  --features {raw,packets}
                        the representation type
  --batch-size BATCH_SIZE
                        input batch size for testing (default: 512)
  --normalize MEAN [STD ...]
                        normalize with specified values for mean and standard deviation (either 2 or 6 values are accepted)
  --label-names LABEL_NAMES [LABEL_NAMES ...]
                        string representation of the class labels. Only used when '--generalized' is not selected.
  --plot                plot the confusion matrix and store the plot as png. Does only have an effect when '--generalized' is not selected.
  --nclasses NCLASSES   number of classes (default: 2)
  --generalized         Calculates a generalized confusion matrix for the binary classification task differentiating fake from real images.
```
