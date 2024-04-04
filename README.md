# Rotating Features for Object Discovery

This repository contains an implementation of the concepts described in the paper "[*Rotating Features for Object Discovery*](https://arxiv.org/abs/2306.00600)".

## Usage

### Installation
Run the executable `setup.sh` in order to install the conda environment, `pytorch`, `cudatoolkit` and `einops`. The script also installs the datasets (*2shapes*, *3shapes* and *MNIST_shapes*).

### Training
To train the network run the following command:
```
python ./autoencoder.py --task=train --epochs=20
```
The above command launches the training on the dataset "*4Shapes*" for 20 epochs.
