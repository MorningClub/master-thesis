# Master's Thesis

This repository was built upon the work of ["Learning to Drive from a World on Rails" paper](https://arxiv.org/abs/2105.00636), and was forked from the [World on Rails repository](https://github.com/dotchen/WorldOnRails).

## Introduction

Building on the work of ["Learning to Drive from a World on Rails" paper](https://arxiv.org/abs/2105.00636), we introduce an autoencoder training step to pre-train a visual encoder module in two different implementations of a visuomotor agent:

* **Implementation A:** Visuomotor agent using a pre-trained visual encoder module with frozen weights, using an additional trainable latent space head.
* **Implementation B:** Visuomotor agent using a pre-trained visual encoder module with unfrozen weights.

## Getting Started
In order to setup the CARLA simulator, and the World on Rails environment, please follow [INSTALL.md](docs/INSTALL.md).

## Training

To use any of our implementations, please run the `rails.autoencoder_train_phase.py` script prior to `rails.train_phase2` or `rails.train_phase2A`, and set the flag `use_trained_encoder` in `config_nocrash.yaml`. To train the visuomotor agent using **Implementation A**, use the `rails.train_phase2A` instead of `rails.train_phase2`. Refer to [RAILS.md](docs/RAILS.md) for the rest on how to train a World on Rails agent.