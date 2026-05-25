"""
data/dataset.py
---------------
Load, embed, normalise, and split the Fluorescent Protein Database.

Expected CSV columns
--------------------
    Protein sequence | Chromophore/ligand | Stokes shift | kDa |
    Emission wavelength | Brightness

Main entry point
----------------
    from protein_predictor.data import prepare_datasets

    splits = prepare_datasets(df)
    # splits["esm"]["train"]  →  FPDataset
    # splits["tscales"]["val"]  →  FPDataset
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from torch.utils.data        import Dataset

from protein_predictor.config import (
    TARGET_COLS, TEST_FRAC, VAL_FRAC, RANDOM_SEED,
)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class FPDataset(Dataset):
    """Minimal Dataset wrapping (X, y) numpy arrays as float32 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Feature concatenation ────────────────────────────────────────────────────

def _concat_esm(row) -> np.ndarray:
    """ESM(1280) ‖ ChemBERTa(768) ‖ Stokes_shift + kDa  →  2050-d"""
    return np.concatenate([
        row["esm"],
        row["smiles_vectors"],
        np.array([row["Stokes shift"], row["kDa"]], dtype=np.float32),
    ])


def _concat_tscales(row) -> np.ndarray:
    """T-scales(256) ‖ ChemBERTa(768) ‖ Stokes_shift + kDa  →  1026-d"""
    return np.concatenate([
        row["tscales_cls"],
        row["smiles_vectors"],
        np.array([row["Stokes shift"], row["kDa"]], dtype=np.float32),
    ])


# ── Main preparation function ─────────────────────────────────────────────────

def prepare_datasets(df: pd.DataFrame) -> dict:
    """
    Build train / val / test FPDatasets from an embedded DataFrame.

    The DataFrame must already contain columns:
        "esm"           — np.ndarray of shape (1280,)
        "tscales_cls"   — np.ndarray of shape (256,)
        "smiles_vectors"— np.ndarray of shape (768,)
        "Stokes shift"  — float
        "kDa"           — float
        "Emission wavelength" — float  (target)
        "Brightness"          — float  (target)

    Returns
    -------
    dict with keys "esm" and "tscales", each containing:
        "train"        : FPDataset  (X normalised, y normalised)
        "val"          : FPDataset
        "test"         : FPDataset
        "y_raw_test"   : np.ndarray  shape (n_test, 2) in original units
        "y_scaler"     : fitted StandardScaler  (use to invert-transform predictions)
        "input_dim"    : int
    """
    # Drop rows with missing embeddings (e.g. unknown chromophore SMILES)
    df = df.dropna(subset=["esm", "tscales_cls", "smiles_vectors"]).reset_index(drop=True)

    X_esm     = np.vstack(df.apply(_concat_esm,     axis=1)).astype(np.float32)
    X_tscales = np.vstack(df.apply(_concat_tscales, axis=1)).astype(np.float32)
    y         = df[TARGET_COLS].values.astype(np.float32)

    # ── Train / val / test indices ────────────────────────────────────────────
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,  random_state=RANDOM_SEED)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=VAL_FRAC, random_state=RANDOM_SEED)

    def _split(X):
        return X[tr_idx], X[va_idx], X[te_idx]

    X1_tr, X1_va, X1_te = _split(X_esm)
    X2_tr, X2_va, X2_te = _split(X_tscales)
    y_tr,  y_va,  y_te  = _split(y)

    # ── Input scalers (fit on train only) ─────────────────────────────────────
    sx1 = StandardScaler()
    X1_tr = sx1.fit_transform(X1_tr)
    X1_va = sx1.transform(X1_va)
    X1_te = sx1.transform(X1_te)

    sx2 = StandardScaler()
    X2_tr = sx2.fit_transform(X2_tr)
    X2_va = sx2.transform(X2_va)
    X2_te = sx2.transform(X2_te)

    # ── Target scaler (shared between both models) ────────────────────────────
    sy = StandardScaler()
    y_tr_s = sy.fit_transform(y_tr)
    y_va_s = sy.transform(y_va)
    y_te_s = sy.transform(y_te)

    return {
        "esm": {
            "train":      FPDataset(X1_tr, y_tr_s),
            "val":        FPDataset(X1_va, y_va_s),
            "test":       FPDataset(X1_te, y_te_s),
            "y_raw_test": y_te,
            "y_scaler":   sy,
            "x_scaler":   sx1,
            "input_dim":  X1_tr.shape[1],
        },
        "tscales": {
            "train":      FPDataset(X2_tr, y_tr_s),
            "val":        FPDataset(X2_va, y_va_s),
            "test":       FPDataset(X2_te, y_te_s),
            "y_raw_test": y_te,
            "y_scaler":   sy,
            "x_scaler":   sx2,
            "input_dim":  X2_tr.shape[1],
        },
    }


# ── K-fold support ─────────────────────────────────────────────────────────

def prepare_kfold_fold(df: pd.DataFrame, fold_train_idx, fold_val_idx,
                       test_idx, random_state: int = 42) -> dict:
    """
    Prepare a single k-fold fold for training.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with embeddings
    fold_train_idx : np.ndarray
        Training indices for this fold
    fold_val_idx : np.ndarray
        Validation indices for this fold
    test_idx : np.ndarray
        Test indices (fixed across all folds)
    random_state : int
        Unused (for API compatibility)

    Returns
    -------
    dict with keys "esm" and "tscales", same structure as prepare_datasets()
    """
    df_clean = df.dropna(subset=["esm", "tscales_cls", "smiles_vectors"]).reset_index(drop=True)

    X_esm     = np.vstack(df_clean.apply(_concat_esm,     axis=1)).astype(np.float32)
    X_tscales = np.vstack(df_clean.apply(_concat_tscales, axis=1)).astype(np.float32)
    y         = df_clean[TARGET_COLS].values.astype(np.float32)

    X1_tr, X1_va, X1_te = X_esm[fold_train_idx], X_esm[fold_val_idx], X_esm[test_idx]
    X2_tr, X2_va, X2_te = X_tscales[fold_train_idx], X_tscales[fold_val_idx], X_tscales[test_idx]
    y_tr,  y_va,  y_te  = y[fold_train_idx], y[fold_val_idx], y[test_idx]

    sx1 = StandardScaler()
    X1_tr = sx1.fit_transform(X1_tr)
    X1_va = sx1.transform(X1_va)
    X1_te = sx1.transform(X1_te) if len(X1_te) > 0 else X1_te

    sx2 = StandardScaler()
    X2_tr = sx2.fit_transform(X2_tr)
    X2_va = sx2.transform(X2_va)
    X2_te = sx2.transform(X2_te) if len(X2_te) > 0 else X2_te

    sy = StandardScaler()
    y_tr_s = sy.fit_transform(y_tr)
    y_va_s = sy.transform(y_va)
    y_te_s = sy.transform(y_te) if len(y_te) > 0 else y_te

    return {
        "esm": {
            "train":      FPDataset(X1_tr, y_tr_s),
            "val":        FPDataset(X1_va, y_va_s),
            "test":       FPDataset(X1_te, y_te_s),
            "y_raw_val":  y_va,
            "y_raw_test": y_te,
            "y_scaler":   sy,
            "input_dim":  X1_tr.shape[1],
        },
        "tscales": {
            "train":      FPDataset(X2_tr, y_tr_s),
            "val":        FPDataset(X2_va, y_va_s),
            "test":       FPDataset(X2_te, y_te_s),
            "y_raw_val":  y_va,
            "y_raw_test": y_te,
            "y_scaler":   sy,
            "input_dim":  X2_tr.shape[1],
        },
    }
