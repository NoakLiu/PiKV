# PiKV `data/` — download, corpora, and dataloaders

Canonical home for **all** evaluation / training text data and the shared
PyTorch dataloaders. Prefer this over any nested `downstream_tasks/**/data`.

## Layout

```
data/
  download_data.py     # HF corpus download CLI
  dataloader.py        # TextWindowDataset, PromptEvalDataset, helpers
  __init__.py
  README.md            # this file
  train.txt            # training corpus (from download)
  test.txt             # held-out text
  prompts_eval.txt     # frozen one-prompt-per-line eval set
  manifest.json        # dataset id, seed, sizes
  test_sample_legacy.txt  # tiny legacy sample (optional)
```

## Quick start

```bash
# From repo root
pip install datasets transformers

python -m data.download_data
python -m data.download_data --max-prompts 256 --prompt-chars 800 --seed 42

# Eval on frozen prompts
python -m downstream_tasks.eval.eval_with_data --max-prompts 32

# Ablation with identical inputs from data/prompts_eval.txt
python -m downstream_tasks.eval.ablation_study --preset factor --from-data
```

## API

```python
from data import (
    create_train_dataloader,
    create_eval_dataloader,
    iter_eval_prompts,
    prompts_to_hidden,
)

train_loader = create_train_dataloader(max_length=512, batch_size=4)
eval_loader = create_eval_dataloader(prefill_tokens=512, batch_size=1, max_prompts=64)

prompts = list(iter_eval_prompts(max_prompts=8))
x = prompts_to_hidden(prompts, hidden_size=512, seq_len=512, seed=42)
```

## Who consumes this

| Consumer | Usage |
|----------|--------|
| `downstream_tasks/eval/eval_with_data.py` | CE / token accuracy on `create_eval_dataloader` |
| `downstream_tasks/eval/ablation_study.py --from-data` | `prompts_to_hidden` for fair module isolation |
| `downstream_tasks/llm/next_tok_pred/*` | Prefer `data/train.txt` / `data/test.txt` |
| `EXPERIMENTAL_PROTOCOL.md` | Mandates `data/prompts_eval.txt` as fairness control |

Legacy shim: `downstream_tasks.data` re-exports this package. Prefer `import data` / `python -m data.download_data`.
