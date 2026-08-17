# Shell scripts for PiKV (canonical location)

| Script | Purpose |
|--------|---------|
| `install_pikv.sh` | Conda/venv + deps + editable install |
| `build_cuda.sh` | Build CUDA kernels (routing + **full compression** + scheduling) |
| `build_fpga.sh` | FPGA host / RTL sim / Vivado bitstream |
| `run_distributed_training.sh` | torchrun DeepSpeed / MoE training modes |
| `run_ntp_distributed.sh` | Next-token-prediction multi-GPU train |

```bash
./scripts/install_pikv.sh
./scripts/build_cuda.sh release      # full compression suite
./scripts/build_cuda.sh test
./scripts/build_fpga.sh bitstream
./scripts/run_distributed_training.sh moe
./scripts/run_ntp_distributed.sh 10 5 pikv
```
