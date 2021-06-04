<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

# frequency-detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

The latest code can be installed directly from GitHub with:

```shell
$ pip install git+https://github.com/gan-police/frequency-forensics
```

The latest code can be installed in development mode with:

```shell
$ git clone https://github.com/gan-police/frequency-forensics
$ cd frequency-forensics
$ pip install -e .
```

Where <kbd>-e</kbd> means "editable" mode.

Periodically run `tox -e black` to blackify the code when in development mode.

## Getting a minimal example to run:

Download FFHQ-Style-Gan examples from
https://drive.google.com/file/d/1pKmmRtRCtFqs-FuwmToXEYeZFaXk98Kw/view?usp=sharing

and extract these into a `data` folder.

Afterwards run:

```shell
$ CUDA_VISIBLE_DEVICES=0 python -m freqdect.prepare_dataset ./data/source_data/ --packets
$ python -m freqdect.prepare_dataset ./data/source_data/ --raw
```

afterwards you should be able to train a classifier using

```shell
$ CUDA_VISIBLE_DEVICES=0 python -m freqdect.train_classifier
```

## Plotting the accuracy

To plot the accuracy results, run
```shell
$ python -m freqdect.plot_accuracy_results {shared, lsun, celeba} {regression, CNN} ...
```
For a list of all optional arguments, open the help page via the `-h` argument.
