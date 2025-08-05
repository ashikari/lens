# mnist_warmup
Brushing off the rust on my bare bones ML modeling with some MNIST experiments.

# Experiment Plan
 - MNIST trained with Neural Network to find the best configuration of a CNN
 - MNIST Trained w/ Partial Dataset Labels
 - MNIST trained w/ Parital Dataset Labels + Psuedo Labels
 - MNIST VAE trained on Full Dataset (Frozen) + Fine Tune on Parital Dataset Labels
 - MNIST VAE Trained on Full Dataset + Fine Tune on Partial Dataset Labels

# TODO:
- [x] Load MNIST
- [x] Visualize MNIST
- [x] Train CNN on MNIST Full
- [x] Incorporate W&B
- [x] Partition MNIST Training (static)
- [] Build multi-training loop for psuedo labels
- [] Build VAE for MNIST reconstruction
- [] Build Encoder fine tuning pipeline
- [] Finalize results in W&B

## 🛠️ Project Tooling

- **Dependency Management:** [Poetry](https://python-poetry.org/)
- **Task Runner:** [Poe the Poet](https://github.com/nat-n/poethepoet)


# How to use
 - all commands can be found in pyproject.toml
 - visualize data: 
 - train a model:
 - Compare multiple models:
 