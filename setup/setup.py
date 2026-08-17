"""
PiKV packaging metadata (lives under setup/).

Prefer installing from the repo root:
  pip install -e .
  pip install -r setup/requirements.txt
"""

from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent
ROOT = SETUP_DIR.parent


def _read_requirements(name: str = "requirements.txt"):
    reqs = []
    path = SETUP_DIR / name
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="pikv",
    version="3.3.0",
    description="PiKV: Parallel Distributed Key-Value Cache with Routing",
    packages=find_packages(
        where=str(ROOT),
        include=["core*", "data*"],
        exclude=[
            "setup*",
            "scripts*",
            "examples*",
            "docs*",
            "assets*",
            "downstream_tasks*",
        ],
    ),
    package_dir={"": str(ROOT)},
    install_requires=_read_requirements("requirements.txt"),
    python_requires=">=3.11",
    extras_require={
        "vllm": ["vllm>=0.5.0"],
        "deepspeed": ["deepspeed"],
        "peft": ["peft>=0.11.0"],
    },
)
