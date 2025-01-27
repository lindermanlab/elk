from setuptools import setup, find_packages

# Define the base requirements with specific versions
base_requirements = [
    "diffrax==0.6.0",
    "dynamax==0.1.5",
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
]

# Create flexible requirements by removing version specifiers
flex_requirements = [req.split("==")[0] for req in base_requirements]

setup(
    name="elk",
    version="0.1",
    description="Scalable and Stable Parallelization of RNNs",
    packages=find_packages(),
    install_requires=base_requirements,  # Use specific versions by default
    extras_require={
        "cr": ["python==3.12.1"],  # Python 3.12.1 for v1.0.0, commit 458ad76
        "flex": flex_requirements,  # Flexible versions without specifiers
    },
)
