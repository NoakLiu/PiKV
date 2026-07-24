"""
PiKV data package — corpus download and shared dataloaders.

Canonical location: repo-root ``data/``
  python -m data.download_data
"""

from .dataloader import (
    PromptEvalDataset,
    TextWindowDataset,
    create_eval_dataloader,
    create_train_dataloader,
    default_data_dir,
    iter_eval_prompts,
    prompts_to_hidden,
    resolve_data_path,
)

__all__ = [
    "TextWindowDataset",
    "PromptEvalDataset",
    "create_train_dataloader",
    "create_eval_dataloader",
    "iter_eval_prompts",
    "prompts_to_hidden",
    "default_data_dir",
    "resolve_data_path",
]
