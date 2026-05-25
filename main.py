"""
main.py
=======
Entry point for the protein_predictor package.

HOW TO RUN
----------
    python main.py                              # 5 folds × 10 runs (default)
    python main.py --n_folds 3 --n_runs 2      # quick smoke-test (~5 min)

Workflow
--------
1. Load CSV from data/Fluorescent-Protein-Database.csv
2. Compute embeddings (ESM-2, T-scales, ChemBERTa) — cached after first run
3. K-fold cross-validation with multiple seeds for reproducibility
4. Save 12 figures and results/kfold_results.json
5. Print summary RMSE across all folds and runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import logging
import os

import numpy as np
import pandas as pd

from protein_predictor.config import CSV_PATH, DATA_DIR, FIGURES_DIR, CHECKPOINTS
from protein_predictor.embeddings import ESMEncoder, TScalesEncoder, ChemBERTaEncoder, SMILES_DICT
from protein_predictor.training import run_kfold_training
from protein_predictor.utils import (
    plot_kfold_consistency,
    plot_kfold_arch_heatmap,
    plot_kfold_model_comparison,
    plot_kfold_pred_wavelength,
    plot_kfold_pred_vs_actual,
    plot_kfold_esm_targets,
    plot_kfold_tsc_targets,
    plot_kfold_protein_errors,
    plot_kfold_protein_errors_esm,
    plot_kfold_protein_errors_tsc,
    plot_kfold_protein_errors_grid,
    plot_kfold_protein_errors_brightness,
)

os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
(Path(__file__).resolve().parent / "results").mkdir(parents=True, exist_ok=True)

_CACHE = DATA_DIR / "embeddings_cache.pkl"


def load_csv() -> pd.DataFrame:
    print(f"\n[1/3] Loading dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"      {len(df)} rows, columns: {list(df.columns)}")
    return df


def compute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    if _CACHE.exists():
        print(f"\n[2/3] Loading cached embeddings from {_CACHE}")
        cached = pd.read_pickle(_CACHE)
        for col in ["esm", "tscales_cls", "smiles_vectors"]:
            df[col] = cached[col].values
        return df

    print("\n[2/3] Computing embeddings (slow on first run, cached afterwards)...")

    print("      -> ESM-2 ...")
    esm_enc = ESMEncoder()
    df["esm"] = df["Protein sequence"].apply(esm_enc.encode)

    print("      -> T-scales BERT ...")
    tsc_enc = TScalesEncoder()
    df["tscales_cls"] = tsc_enc.encode_series(df["Protein sequence"])

    print("      -> ChemBERTa ...")
    df["smiles"] = df["Chromophore/ligand"].str.strip().str.upper().map(SMILES_DICT)
    chem_enc = ChemBERTaEncoder()
    df["smiles_vectors"] = df["smiles"].apply(chem_enc.encode)

    df[["esm", "tscales_cls", "smiles_vectors"]].to_pickle(_CACHE)
    print(f"      Cached -> {_CACHE}")
    return df


def build_oof_dataframe(df: pd.DataFrame, oof_data: dict) -> pd.DataFrame:
    """
    Combine OOF predictions with protein metadata from the original dataframe.

    Returns a DataFrame with one row per OOF sample (run 0, best architecture
    per fold), containing protein identity columns alongside predicted and true
    values for both targets and their absolute errors.
    """
    df_clean = df.dropna(
        subset=["esm", "tscales_cls", "smiles_vectors"]
    ).reset_index(drop=True)

    idx = oof_data["idx"]   # row positions in df_clean

    meta = df_clean.loc[idx, ["Protein Name", "PDB code", "Chromophore/ligand"]].reset_index(drop=True)

    esm_pred = oof_data["esm"]["pred"]
    esm_true = oof_data["esm"]["true"]
    tsc_pred = oof_data["tscales"]["pred"]
    tsc_true = oof_data["tscales"]["true"]

    oof_df = pd.DataFrame({
        "Protein Name":          meta["Protein Name"].values,
        "PDB code":              meta["PDB code"].values,
        "Chromophore":           meta["Chromophore/ligand"].values,
        "True Wavelength":       esm_true[:, 0],
        "True Brightness":       esm_true[:, 1],
        "ESM Pred Wavelength":   esm_pred[:, 0],
        "ESM Pred Brightness":   esm_pred[:, 1],
        "TSC Pred Wavelength":   tsc_pred[:, 0],
        "TSC Pred Brightness":   tsc_pred[:, 1],
        "ESM Abs Err Wavelength": np.abs(esm_pred[:, 0] - esm_true[:, 0]),
        "ESM Abs Err Brightness": np.abs(esm_pred[:, 1] - esm_true[:, 1]),
        "TSC Abs Err Wavelength": np.abs(tsc_pred[:, 0] - tsc_true[:, 0]),
        "TSC Abs Err Brightness": np.abs(tsc_pred[:, 1] - tsc_true[:, 1]),
    })
    return oof_df


def run_kfold_pipeline(df: pd.DataFrame, n_folds: int, n_runs: int):
    print(f"\n[3/3] K-fold CV: {n_folds} folds × {n_runs} runs ...")

    kfold_results, oof_data = run_kfold_training(df, n_folds=n_folds, n_runs=n_runs)

    plot_kfold_consistency(kfold_results, save=True)
    plot_kfold_arch_heatmap(kfold_results, save=True)
    plot_kfold_model_comparison(kfold_results, save=True)
    plot_kfold_pred_wavelength(oof_data, save=True)
    plot_kfold_pred_vs_actual(oof_data, save=True)
    plot_kfold_esm_targets(oof_data, save=True)
    plot_kfold_tsc_targets(oof_data, save=True)

    # ── Per-protein OOF DataFrame + CSV + annotation figure ───────────────────
    oof_df = build_oof_dataframe(df, oof_data)
    results_dir = Path(__file__).resolve().parent / "results"
    csv_path = results_dir / "oof_predictions.csv"
    oof_df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")
    plot_kfold_protein_errors(oof_df, save=True)
    plot_kfold_protein_errors_esm(oof_df, save=True)
    plot_kfold_protein_errors_tsc(oof_df, save=True)
    plot_kfold_protein_errors_grid(oof_df, save=True)
    plot_kfold_protein_errors_brightness(oof_df, save=True)

    # ── Summary table in original units (from OOF predictions) ────────────────
    print(f"\n{'='*65}")
    print(f"PERFORMANCE SUMMARY  ({n_folds} folds x {n_runs} runs, original units)")
    print(f"{'='*65}")
    print(f"{'':22s}  {'ESM-2':>18s}  {'T-scales + BERT':>18s}")
    print(f"{'':22s}  {'RMSE':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAE':>8s}")
    print(f"{'-'*65}")

    for col, (tname, unit) in enumerate([("Emission Wavelength", "nm"), ("Brightness", "")]):
        u = f" {unit}" if unit else ""
        row_parts = []
        for mkey in ["esm", "tscales"]:
            p = oof_data[mkey]["pred"][:, col]
            t = oof_data[mkey]["true"][:, col]
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            mae  = float(np.mean(np.abs(p - t)))
            row_parts.append((rmse, mae, u))
        esm_rmse, esm_mae, u = row_parts[0]
        tsc_rmse, tsc_mae, _ = row_parts[1]
        print(
            f"  {tname + u:20s}  "
            f"{esm_rmse:>8.3f}  {esm_mae:>8.3f}  "
            f"{tsc_rmse:>8.3f}  {tsc_mae:>8.3f}"
        )

    print(f"{'='*65}\n")
    print(f"  Figures -> {FIGURES_DIR}/")
    print(f"  Results -> results/kfold_results.json")
    print(f"  OOF CSV -> results/oof_predictions.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation for fluorescent protein property prediction"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds (default: 5)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of independent runs with different seeds (default: 10)"
    )
    args = parser.parse_args()

    df = load_csv()
    df = compute_embeddings(df)
    run_kfold_pipeline(df, n_folds=args.n_folds, n_runs=args.n_runs)
