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

## whats the plan?

- I think it may be a good idea to use the fwt instead of the fft to find GAN generated conted, because of Gibbs
  phenomenon:
  https://en.wikipedia.org/wiki/Gibbs_phenomenon
- Faces have sharp edges i.e. at the forehead - hair transition. I expect this to happen with the method above. => See
  if we can find a wavelet that is better suited to GAN-content dectection and try to write a paper about it.

## GAN Review List

https://docs.google.com/spreadsheets/d/1uWQBdcbQIOcomveN912X8m3U0W_wORY1ZF5tMvGmmSA/edit?usp=sharing

## Motivation List

Distopian future where you dont know what's real and what's not. Thus, we have to prepare beforehand for this race by
having machines that are able to recognize the difference between fake and real content.

For example:

- People already started using GANs to fabricate fake content (e.g., images, audios, etc.).
    - Example: Biden's video
    - Example: Homer's voice
- There are existing fake versions of Copyright work that is being.
    - Youtube videos that have been copied/modified (e.g., 2x speed video and you want to recognize)
    - Recognizing real paintings

## Goals

- Flag such content so that people is aware that the image/audio is fake.
- Having an open source software available for the community (e.g., media) to check the vericity of content.
- Benchmark for the community and extensions (e.g., paper comes and they have to be able to avoid being recognized by
  our software but at the same time new comes).
- If easily generated content is everywhere, the source is key to determine the vericity of the content.
-

## Loading the data

FFHQ:

``` shell
$ python download_ffhq.py -t
```

LSUN:
The lsun repository has python 2 to 3 conversion problems, the solution proposed in
https://github.com/fyu/lsun/issues/11#issuecomment-567390245
works.

## Plotting the accuracy

To plot the accuracy results, run
```shell
$ python -m freqdect.plot_accuracy_results {shared, lsun, celeba} {regression, CNN} ...
```
For a list of all optional arguments, open the help page via the `-h` argument.
