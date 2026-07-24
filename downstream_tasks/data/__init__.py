"""
Backward-compatible shim — prefer ``import data`` / ``python -m data.download_data``.

All logic lives in the repo-root ``data/`` package.
"""

from data import (  # noqa: F401
    PromptEvalDataset,
    TextWindowDataset,
    create_eval_dataloader,
    create_train_dataloader,
    default_data_dir,
    iter_eval_prompts,
    prompts_to_hidden,
    resolve_data_path,
)
from data.download_data import download  # noqa: F401

__all__ = [
    "download",
    "TextWindowDataset",
    "PromptEvalDataset",
    "create_train_dataloader",
    "create_eval_dataloader",
    "iter_eval_prompts",
    "prompts_to_hidden",
    "default_data_dir",
    "resolve_data_path",
]
