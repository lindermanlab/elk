# Scalable and Stable Parallelization of Nonlinear RNNs

This repository contains code for the paper "Towards Scalable and Stable Parallelization of Nonlinear RNNs," published in NeurIPS 2024. The paper is available on ArXiv [here](https://arxiv.org/abs/2407.19115). The primary contributions of our paper are the quasi-DEER and ELK algorithms. quasi-DEER is a scalable algorithm. ELK, which stands for "**E**valuating **L**evenberg-Marquardt with **K**alman, is a stable algorithm. We also contribute quasi-ELK, which is both scalable and stable.

![Figure 1](./ims/banner_schematic.png)

Our paper and codebase build on the work of YH Lim, *et al.*, in "Parallelizing non-linear sequential models over the sequence length", ([paper](https://arxiv.org/abs/2309.12252), [codebase](https://github.com/machine-discovery/deer)) published in ICLR 2024. 

## Installation Instructions

We recommend using a virtual environment. **Use python 3.12.1**

Within that virtual environment, first install JAX with
```
pip install --upgrade pip
pip install -U "jax[cuda12]"
```

After installing JAX, pip install the package with
```
pip install --upgrade -e .
```

We originally wrote the paper in Python 3.9, but this required running in a singularity container and using old verisons of JAX. Using the code is much easier in Python 3.12.1 and the provided `setup.py`.

## Citation
```
@inproceedings{gonzalez2024scalable,
  title={Towards Scalable and Stable Parallelization of Nonlinear RNNs},
  author={Xavier Gonzalez and Andrew Warrington and Jimmy T. H. Smith and Scott W. Linderman},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://doi.org/10.48550/arXiv.2407.19115},
}
```
