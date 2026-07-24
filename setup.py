from setuptools import setup, find_packages

setup(
    name="pikv",
    version="3.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "scipy>=1.10.0",
        "numpy<2.0.0",
        "psutil>=5.9.0",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.11",
) 