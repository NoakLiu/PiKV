# Packaging & environment files

| File | Role |
|------|------|
| `setup.py` | Package metadata (also invoked via root `../setup.py`) |
| `requirements.txt` | Full pip deps (incl. torch) |
| `requirements-conda.txt` | Pip deps without torch (after conda CUDA PyTorch) |
| `environment.yml` | Conda env `pikv` (Python 3.11 + CUDA 12.4) |

```bash
# From repo root
./scripts/install_pikv.sh
conda env create -f setup/environment.yml
pip install -r setup/requirements.txt && pip install -e .
```
