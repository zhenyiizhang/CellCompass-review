from setuptools import setup, find_packages

setup(
    name="CytoBridge",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy<2",
        "torch",
        "torchdiffeq",
        "torchsde",
        "scipy",
        "scikit-learn",
        "pot",
        "phate",
        "pyyaml",
        "tqdm",
        "seaborn",
        "pandas",
        "ipywidgets",
        "scanpy",
        "geomloss",
        "scvelo",
    ],
)
