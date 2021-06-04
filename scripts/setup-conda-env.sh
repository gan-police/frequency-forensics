conda create \
  python=3.8 \ # use python 3.8
  scikit-learn-intelex \ #intel sklearn extension
  tqdm \
  matplotlib \
  seaborn \
  PyWavelets \
  scipy \
  pillow \
  opencv \
  scikit-learn \
  -c intel \ # use intel channel
  -c conda-forge \ # use conda-forge channel
  --prefix=~/env/intel38 # install environment in ~/env with the name intel38
