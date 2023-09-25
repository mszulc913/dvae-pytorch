# Discrete Variational Autoencoder in PyTorch

Experiments with Discrete Variational Autoencoder in PyTorch.

## Installation
To install the pinned versions of the dependencies (versions I used to conduct the experiments)
run the following:
```shell
pip install --upgrade pip
pip install -r requirements.txt -r dev-requirements.txt
```

If you want to install newer versions of the dependencies you can just run
```shell
pip install ".[dev]"
```

**Note**: If you want to use a GPU/TPU acceleration, make sure you have a compatible
version of CUDA installed on your system.

## Run
You can run an example experiment with CIFAR10 using the [train.py](dvae_pytorch/training/train.py)
script:
```shell
python -m dvae_pytorch.training.train --config configs/dvae_cifar10.yaml
```
You can track your experiment with tensorboard:
```shell
tensorboard --logdir tb_logs
```

### Tests
To run the tests execute the following command:
```shell
python -m pytest
```