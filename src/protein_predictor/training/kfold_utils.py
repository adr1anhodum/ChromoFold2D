"""
training/kfold_utils.py
-----------------------
K-fold splitting and random seed management for reproducible training.
"""

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def split_kfold(n_samples: int, n_splits: int = 5, random_state: int = 42):
    """
    Generate k-fold split indices.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility

    Yields
    ------
    tuple
        (train_idx, val_idx) for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(n_samples)
    for train_idx, val_idx in kf.split(indices):
        yield train_idx, val_idx
