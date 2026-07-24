from setuptools import setup, find_packages

# Keep in sync with requirements.txt (core runtime deps).
install_requires = [
    "torch>=2.2.0,<2.7",
    "torchvision>=0.17.0",
    "torchaudio>=2.2.0",
    "transformers>=4.40.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.23.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.2.0",
    "tokenizers>=0.19.0",
    "numpy>=1.26.0,<2.0.0",
    "scipy>=1.11.0",
    "pandas>=2.1.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.66.0",
    "wandb>=0.17.0",
    "tensorboard>=2.14.0",
    "psutil>=5.9.0",
    "matplotlib>=3.8.0",
    "evaluate>=0.4.0",
]

setup(
    name="pikv",
    version="3.2.0",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    extras_require={
        "vllm": ["vllm>=0.5.0"],
        "deepspeed": ["deepspeed"],
        "peft": ["peft>=0.11.0"],
    },
)
