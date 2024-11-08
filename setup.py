from setuptools import setup, find_packages

setup(
    name="elk",
    version="0.1",
    description="Scalable and Stable Parallelization of RNNs",
    python_requires="==3.12.1",
    install_requires=[
        "diffrax==0.6.0",
        "dynamax==0.1.4",
        "equinox==0.11.8",
        "flax==0.10.0",
        "GPUtil==1.4.0",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "optax==0.2.3",
        "pytorch-lightning==2.4.0",
        "tensorflow-probability==0.24.0",
        "torch==2.5.0",
        "tqdm==4.66.5",
        "wandb==0.18.5",
        "pandas==2.2.3",
    ],
    packages=find_packages()
)
