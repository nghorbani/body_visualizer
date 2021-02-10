# Body Visualizer
![alt text](support_data/vposer_samples.png "Novel Human Poses Sampled From the VPoser.")
## Description
Set of tools to visualize and render SMPL family body parameters.

## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Tutorials](#tutorials)
  * [Contact](#contact)

## Installation
**Requirements**
- For Python 2.7 install: pip install -r requirements27.txt
- For Python 3.7 install: pip install -r requirements.txt

Install from this repository for the latest developments:
```bash
pip install git+https://github.com/nghorbani/body_visualizer
```

## Tutorials
![alt text](support_data/latent_interpolation_1.gif "Interpolation of novel poses on the smoother VPoser latent space.")
![alt text](support_data/latent_interpolation_2.gif "Interpolation of novel poses on the smoother VPoser latent space.")

* [VPoser PoZ Space for Body Models](notebooks/vposer_poZ.ipynb)
* [Sampling Novel Body Poses from VPoser](notebooks/vposer_sampling.ipynb)
* [Preparing VPoser Training Dataset](src/human_body_prior/data/README.md)
* [Train VPoser from Scratch](src/human_body_prior/train/README.md)

## Contact
The code in this repository is developed by [Nima Ghorbani](https://nghorbani.github.io/).
