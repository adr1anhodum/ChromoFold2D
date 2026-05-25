"""
Streamlit web interface for fluorescent protein property prediction.

Run with:
    streamlit run app.py

First run `python main.py` to generate the k-fold results and figures.
On first launch, a quick demo model is trained (~2 min on CPU).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from protein_predictor.config import (
    CSV_PATH, DATA_DIR, FIGURES_DIR, BATCH_SIZE, MAX_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT, PATIENCE,
)
from protein_predictor.embeddings import ESMEncoder, TScalesEncoder, ChemBERTaEncoder, SMILES_DICT
from protein_predictor.data import prepare_datasets
from protein_predictor.models import FluorescenceNet
from protein_predictor.training import FPDataModule, MetricsCallback


# ── Load data + train quick demo models ──────────────────────────────────────

@st.cache_resource
def load_models_and_data():
    pl.seed_everything(42)
    torch.manual_seed(42)

    with st.spinner("Loading data and embeddings..."):
        df = pd.read_csv(CSV_PATH)
        cache_file = DATA_DIR / "embeddings_cache.pkl"

        if cache_file.exists():
            cached = pd.read_pickle(cache_file)
            for col in ["esm", "tscales_cls", "smiles_vectors"]:
                df[col] = cached[col].values
        else:
            esm_enc = ESMEncoder()
            df["esm"] = df["Protein sequence"].apply(esm_enc.encode)

            tsc_enc = TScalesEncoder()
            df["tscales_cls"] = tsc_enc.encode_series(df["Protein sequence"])

            df["smiles"] = df["Chromophore/ligand"].str.strip().str.upper().map(SMILES_DICT)
            chem_enc = ChemBERTaEncoder()
            df["smiles_vectors"] = df["smiles"].apply(chem_enc.encode)

            df[["esm", "tscales_cls", "smiles_vectors"]].to_pickle(cache_file)

        # Always keep encoder instances alive for inference
        esm_enc  = ESMEncoder()
        tsc_enc  = TScalesEncoder()
        chem_enc = ChemBERTaEncoder()

    splits = prepare_datasets(df)

    with st.spinner("Training demo models (first launch only, ~2 min)..."):
        models = {}
        for model_key in ["esm", "tscales"]:
            s = splits[model_key]
            cb = MetricsCallback()
            dm = FPDataModule(s["train"], s["val"], s["test"], batch_size=BATCH_SIZE)

            model = FluorescenceNet(
                input_dim=s["input_dim"],
                hidden_sz=64,
                n_layers=2,
                dropout=DROPOUT,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            trainer = pl.Trainer(
                accelerator="cpu",
                max_epochs=MAX_EPOCHS,
                callbacks=[
                    cb,
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
                ],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, datamodule=dm)
            model.eval()
            models[model_key] = {"model": model, "split": s, "cb": cb, "dm": dm}

    return df, splits, models, esm_enc, tsc_enc, chem_enc


def _run_inference(model, dataloader):
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            preds.append(model(x).cpu().numpy())
    return np.vstack(preds) if preds else np.empty((0, 2))


def make_prediction(sequence, chromophore_input, stokes_shift, mw,
                    model_key, models, splits, esm_enc, tsc_enc, chem_enc):
    chromophore_upper = chromophore_input.strip().upper()
    smiles = SMILES_DICT.get(chromophore_upper, chromophore_upper)

    smiles_vec = chem_enc.encode(smiles)
    if smiles_vec is None:
        return None, "Could not encode chromophore — try a SMILES string directly"

    if model_key == "esm":
        prot_vec = esm_enc.encode(sequence)
    else:
        tsc_result = tsc_enc.encode_series(pd.Series([sequence]))
        prot_vec = tsc_result[0] if isinstance(tsc_result, list) else tsc_result.tolist()[0]

    X_raw    = np.concatenate([prot_vec, smiles_vec, [stokes_shift, mw]], dtype=np.float32)
    s        = splits[model_key]
    X_scaled = s["x_scaler"].transform(X_raw.reshape(1, -1)).astype(np.float32)

    model = models[model_key]["model"]
    with torch.no_grad():
        pred_norm = model(torch.tensor(X_scaled)).cpu().numpy()[0]

    pred_original = s["y_scaler"].inverse_transform([pred_norm])[0]
    wavelength, brightness = pred_original
    return {"wavelength": float(wavelength), "brightness": float(brightness)}, None


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fluorescent Protein Predictor", layout="wide")
st.title("Fluorescent Protein Property Predictor")
st.markdown(
    "Predict **emission wavelength** and **brightness** from protein sequence and chromophore data."
)

df, splits, models, esm_enc, tsc_enc, chem_enc = load_models_and_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_kfold = st.tabs(["Prediction", "K-Fold Results"])

with tab_pred:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        model_choice = st.selectbox("Model", ["ESM-2", "T-scales"])
        model_key    = "esm" if model_choice == "ESM-2" else "tscales"

        sequence = st.text_area(
            "Protein Sequence (single-letter amino acids)",
            height=150,
            value="MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVWPTL",
        )
        chromophore = st.text_input("Chromophore (name or SMILES, e.g. CR2, CRO)")

        c1, c2 = st.columns(2)
        with c1:
            stokes_shift = st.number_input("Stokes Shift (nm)", 0.0, 100.0, 21.0)
        with c2:
            mw = st.number_input("Molecular Weight (kDa)", 1.0, 100.0, 27.0)

        if st.button("Predict"):
            if not sequence.strip():
                st.error("Please enter a protein sequence.")
            elif not chromophore.strip():
                st.error("Please enter a chromophore name or SMILES.")
            else:
                with st.spinner("Running prediction..."):
                    pred, error = make_prediction(
                        sequence, chromophore, stokes_shift, mw,
                        model_key, models, splits, esm_enc, tsc_enc, chem_enc,
                    )
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Prediction complete")
                    m1, m2 = st.columns(2)
                    m1.metric("Emission Wavelength", f"{pred['wavelength']:.1f} nm")
                    m2.metric("Brightness",          f"{pred['brightness']:.3f}")

    with col2:
        st.subheader("Demo Model — Test-Set Performance")
        entry = models[model_key]
        y_pred_norm = _run_inference(entry["model"], entry["dm"].test_dataloader())
        y_pred = entry["split"]["y_scaler"].inverse_transform(y_pred_norm)
        y_true = entry["split"]["y_raw_test"]

        if len(y_true) > 0:
            rmse_wl = float(np.sqrt(((y_pred[:, 0] - y_true[:, 0]) ** 2).mean()))
            rmse_br = float(np.sqrt(((y_pred[:, 1] - y_true[:, 1]) ** 2).mean()))
            m1, m2 = st.columns(2)
            m1.metric("RMSE Emission Wavelength", f"{rmse_wl:.2f} nm")
            m2.metric("RMSE Brightness",          f"{rmse_br:.4f}")
            st.caption(
                "Note: these metrics come from a single quick demo model (2L × 64N). "
                "Run `python main.py` for full k-fold evaluation."
            )
        else:
            st.info("No test samples available.")

with tab_kfold:
    st.subheader("K-Fold Cross-Validation Results")
    st.markdown(
        "Generated by running `python main.py`. "
        "Run it first to populate the figures and results below."
    )

    results_path = Path(__file__).resolve().parent / "results" / "kfold_results.json"
    if results_path.exists():
        with open(results_path) as f:
            raw = json.load(f)
        n_runs  = len(raw)
        n_folds = len(next(iter(raw.values())))
        st.success(f"Found results: {n_folds} folds × {n_runs} runs")

    fig_paths = [
        ("K-Fold Consistency",                          FIGURES_DIR / "kfold_consistency.png"),
        ("Architecture Heatmap",                        FIGURES_DIR / "kfold_arch_heatmap.png"),
        ("Model Comparison",                            FIGURES_DIR / "kfold_model_comparison.png"),
        ("Predicted vs. True — Wavelength",             FIGURES_DIR / "kfold_pred_wavelength.png"),
        ("Predicted vs. True — All Targets",            FIGURES_DIR / "kfold_pred_vs_actual.png"),
        ("ESM-2 — Wavelength and Brightness",           FIGURES_DIR / "kfold_esm_targets.png"),
        ("T-scales + BERT — Wavelength and Brightness", FIGURES_DIR / "kfold_tsc_targets.png"),
        ("Per-Protein View — Wavelength (both models)",              FIGURES_DIR / "kfold_protein_errors.png"),
        ("Per-Protein View — ESM-2 (wavelength + brightness)",        FIGURES_DIR / "kfold_protein_errors_esm.png"),
        ("Per-Protein View — T-scales + BERT (wavelength + brightness)", FIGURES_DIR / "kfold_protein_errors_tsc.png"),
        ("Per-Protein View — 2×2 grid (all models & targets)",        FIGURES_DIR / "kfold_protein_errors_grid.png"),
        ("Per-Protein View — Brightness (both models)",               FIGURES_DIR / "kfold_protein_errors_brightness.png"),
    ]
    any_found = False
    for title, path in fig_paths:
        if path.exists():
            any_found = True
            st.markdown(f"**{title}**")
            st.image(str(path))
    if not any_found:
        st.info("No figures found yet. Run `python main.py` to generate them.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Help")
st.sidebar.markdown("""
**Protein Sequence** — paste amino acid sequence (single letter code)

**Chromophore** — use abbreviations (GFP, RFP, BFP) or a SMILES string directly

**Stokes Shift** — difference between excitation and emission peak (nm)

**Molecular Weight** — approximate protein mass in kDa

**Models**
- ESM-2: 1280-d protein language model embeddings (650M params)
- T-scales: 256-d physicochemical descriptor embeddings (faster)

**K-Fold Results tab** requires running `python main.py` first.
""")
