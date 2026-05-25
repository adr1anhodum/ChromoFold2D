"""
training/kfold.py
-----------------
K-fold cross-validation training with multiple runs for reproducibility.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from protein_predictor.config import (
    ARCH_GRID, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, DROPOUT, PATIENCE,
)
from protein_predictor.data import prepare_kfold_fold
from protein_predictor.models import FluorescenceNet
from protein_predictor.training.callbacks import MetricsCallback
from protein_predictor.training.datamodule import FPDataModule
from protein_predictor.training.kfold_utils import set_seeds, split_kfold


def _silence_loggers():
    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run_kfold_training(df, n_folds: int = 5, n_runs: int = 3,
                       results_dir: Path = None) -> tuple:
    """
    Run k-fold cross-validation with multiple runs for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with embeddings and targets
    n_folds : int
        Number of folds (default 5)
    n_runs : int
        Number of independent runs with different random seeds (default 3)
    results_dir : Path
        Directory to save results JSON

    Returns
    -------
    tuple
        (all_results, oof_data)

        all_results : {run_id: {fold_id: {(model_key, n_layers, n_neurons): val_rmse}}}

        oof_data : {
            "esm":     {"pred": ndarray (n, 2), "true": ndarray (n, 2)},
            "tscales": {"pred": ndarray (n, 2), "true": ndarray (n, 2)},
        }
        Predictions are in ORIGINAL (non-normalised) units.
        Collected from run 0 using the best architecture per fold.
    """
    _silence_loggers()

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    df_clean = df.dropna(
        subset=["esm", "tscales_cls", "smiles_vectors"]
    ).reset_index(drop=True)
    n_samples = len(df_clean)

    all_results = {}
    oof_data = {
        "esm":     {"pred": [], "true": []},
        "tscales": {"pred": [], "true": []},
        "idx":     [],   # row positions in df_clean for each OOF sample (run 0 only)
    }

    for run_id in range(n_runs):
        seed = 42 + run_id
        set_seeds(seed)
        print(f"\n{'='*70}")
        print(f"RUN {run_id + 1}/{n_runs}  (seed={seed})")
        print(f"{'='*70}")

        fold_results = {}
        fold_gen = split_kfold(n_samples, n_splits=n_folds, random_state=seed)

        for fold_id, (train_idx, val_idx) in enumerate(fold_gen):
            test_idx = np.array([], dtype=int)
            splits   = prepare_kfold_fold(df, train_idx, val_idx, test_idx)

            print(f"\n  [Fold {fold_id + 1}/{n_folds}]")
            arch_results  = {}
            best_models   = {}  # model_key -> best (model, val_rmse) so far this fold

            for n_layers, n_neurons in ARCH_GRID:
                for model_key in ["esm", "tscales"]:
                    s  = splits[model_key]
                    cb = MetricsCallback()
                    dm = FPDataModule(
                        s["train"], s["val"], s["test"], batch_size=BATCH_SIZE
                    )

                    model = FluorescenceNet(
                        input_dim    = s["input_dim"],
                        hidden_sz    = n_neurons,
                        n_layers     = n_layers,
                        dropout      = DROPOUT,
                        lr           = LEARNING_RATE,
                        weight_decay = WEIGHT_DECAY,
                    )

                    trainer = pl.Trainer(
                        max_epochs           = MAX_EPOCHS,
                        callbacks            = [
                            cb,
                            EarlyStopping(
                                monitor="val_loss", patience=PATIENCE, mode="min"
                            ),
                        ],
                        logger               = False,
                        enable_checkpointing = False,
                        enable_progress_bar  = False,
                        enable_model_summary = False,
                    )
                    trainer.fit(model, datamodule=dm)

                    val_rmse = trainer.callback_metrics.get("val_rmse", float("nan"))
                    if hasattr(val_rmse, "item"):
                        val_rmse = val_rmse.item()
                    val_rmse = float(val_rmse)

                    arch_key = (model_key, n_layers, n_neurons)
                    arch_results[arch_key] = val_rmse

                    # Track best model for this fold (run 0 OOF collection)
                    if run_id == 0:
                        prev_best = best_models.get(model_key)
                        if prev_best is None or val_rmse < prev_best[1]:
                            best_models[model_key] = (model, val_rmse, splits[model_key])

            # Print best this fold
            best_esm = min(
                ((k, v) for k, v in arch_results.items() if k[0] == "esm"),
                key=lambda x: x[1],
            )
            best_tsc = min(
                ((k, v) for k, v in arch_results.items() if k[0] == "tscales"),
                key=lambda x: x[1],
            )
            print(
                f"    Best ESM     : {best_esm[0][1]}L × {best_esm[0][2]:3d}N"
                f"  val_rmse={best_esm[1]:.4f}"
            )
            print(
                f"    Best T-scales: {best_tsc[0][1]}L × {best_tsc[0][2]:3d}N"
                f"  val_rmse={best_tsc[1]:.4f}"
            )

            fold_results[fold_id] = arch_results

            # Collect OOF predictions from best architecture (run 0 only)
            if run_id == 0:
                oof_data["idx"].extend(val_idx.tolist())
                for model_key, (model, _, s) in best_models.items():
                    model.eval()
                    val_loader = DataLoader(
                        s["val"], batch_size=BATCH_SIZE, shuffle=False
                    )
                    preds_norm = []
                    with torch.no_grad():
                        for x, _ in val_loader:
                            preds_norm.append(model(x).cpu().numpy())

                    if preds_norm:
                        preds_norm = np.vstack(preds_norm)
                        # Inverse-transform to original units
                        preds_orig = s["y_scaler"].inverse_transform(preds_norm)
                        true_orig  = s["y_raw_val"]

                        oof_data[model_key]["pred"].append(preds_orig)
                        oof_data[model_key]["true"].append(true_orig)

        all_results[run_id] = fold_results

    # Concatenate OOF across folds
    oof_final = {"idx": np.array(oof_data["idx"])}
    for model_key in ["esm", "tscales"]:
        preds_list = oof_data[model_key]["pred"]
        trues_list = oof_data[model_key]["true"]
        if preds_list:
            oof_final[model_key] = {
                "pred": np.vstack(preds_list),
                "true": np.vstack(trues_list),
            }
        else:
            oof_final[model_key] = {"pred": np.empty((0, 2)), "true": np.empty((0, 2))}

    # Save results JSON
    results_file = results_dir / "kfold_results.json"
    with open(results_file, "w") as f:
        serializable = {}
        for run_k, fold_dict in all_results.items():
            serializable[str(run_k)] = {}
            for fold_k, arch_dict in fold_dict.items():
                serializable[str(run_k)][str(fold_k)] = {
                    f"{mk}_{nl}L_{nn}N": v
                    for (mk, nl, nn), v in arch_dict.items()
                }
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*70}")
    print(f"K-fold results saved to {results_file}")
    print(f"{'='*70}\n")

    return all_results, oof_final
