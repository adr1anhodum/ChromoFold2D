"""
utils/figures.py
----------------
Publication-quality figure generation for the Chromofold 2D k-fold pipeline.

Figures produced
----------------
 1. plot_kfold_consistency            - per-fold RMSE across runs (mean +/- std + individual points)
 2. plot_kfold_arch_heatmap           - mean RMSE heatmap over all architectures (layers x neurons)
 3. plot_kfold_model_comparison       - violin + box plot: ESM-2 vs T-scales + BERT
 4. plot_kfold_pred_wavelength        - 1x2 OOF predicted vs true emission wavelength (nm)
 5. plot_kfold_pred_vs_actual         - 2x2 OOF predicted vs true for both targets
 6. plot_kfold_esm_targets            - 1x2 ESM-2 predicted vs true: wavelength and brightness
 7. plot_kfold_tsc_targets            - 1x2 T-scales predicted vs true: wavelength and brightness
 8. plot_kfold_protein_errors         - 1x2 per-protein wavelength: both models annotated
 9. plot_kfold_protein_errors_esm     - 1x2 per-protein ESM-2: wavelength and brightness annotated
10. plot_kfold_protein_errors_tsc     - 1x2 per-protein T-scales: wavelength and brightness annotated
11. plot_kfold_protein_errors_grid    - 2x2 per-protein: both models x both targets annotated
12. plot_kfold_protein_errors_brightness - 1x2 per-protein brightness: both models annotated
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

from protein_predictor.config import (
    FIGURES_DIR, FIG_DPI, LAYER_OPTIONS, NEURON_OPTIONS,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
_ESM_COLOR = "#2ecc71"
_TSC_COLOR = "#e74c3c"

_FS_TITLE  = 15
_FS_LABEL  = 13
_FS_TICK   = 12
_FS_LEGEND = 12
_FS_ANNOT  = 11

_STYLE = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
}

# ── Wavelength color classes ──────────────────────────────────────────────────
_WL_BINS   = [470, 530, 580]
_WL_COLORS = ["#5B7FD4", "#3DBE6C", "#F0C030", "#E05040"]   # blue/green/yellow/red
_WL_LABELS = ["< 470 nm", "470–530 nm", "530–580 nm", "> 580 nm"]


def _wl_point_colors(wavelengths: np.ndarray) -> list[str]:
    """Map each emission wavelength to its spectral color class."""
    return [
        _WL_COLORS[0] if w < 470
        else _WL_COLORS[1] if w < 530
        else _WL_COLORS[2] if w < 580
        else _WL_COLORS[3]
        for w in wavelengths
    ]


def _color_legend_handles(markersize: int = 6) -> list:
    return [
        mlines.Line2D(
            [], [], marker="o", color="w", markerfacecolor=c,
            markersize=markersize, label=lbl,
        )
        for c, lbl in zip(_WL_COLORS, _WL_LABELS)
    ]


def _stat_box(ax, rmse: float, mae: float, r2: float, unit: str = ""):
    """Add RMSE / MAE / R² annotation in the top-left corner."""
    u = f" {unit}" if unit else ""
    ax.text(
        0.04, 0.96,
        f"RMSE = {rmse:.2f}{u}\nMAE  = {mae:.2f}{u}\n$R^2$   = {r2:.3f}",
        transform=ax.transAxes, fontsize=_FS_ANNOT, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.90, ec="lightgray"),
    )


def _apply_style():
    plt.rcParams.update(_STYLE)


def _oof_stats(pred: np.ndarray, true: np.ndarray, col: int):
    """Return (rmse, mae, r2) for one target column."""
    p, t = pred[:, col], true[:, col]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    ss_r = np.sum((p - t) ** 2)
    ss_t = np.sum((t - t.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-12))
    return rmse, mae, r2


def _extract_best_per_run_fold(kfold_results: dict) -> tuple[dict, dict]:
    n_folds = max(
        fold_id
        for fold_dict in kfold_results.values()
        for fold_id in fold_dict
    ) + 1
    best_esm: dict[int, list] = {f: [] for f in range(n_folds)}
    best_tsc: dict[int, list] = {f: [] for f in range(n_folds)}
    for fold_dict in kfold_results.values():
        for fold_id, arch_dict in fold_dict.items():
            esm_vals = [v for (mk, *_), v in arch_dict.items() if mk == "esm"]
            tsc_vals = [v for (mk, *_), v in arch_dict.items() if mk == "tscales"]
            if esm_vals:
                best_esm[fold_id].append(min(esm_vals))
            if tsc_vals:
                best_tsc[fold_id].append(min(tsc_vals))
    return best_esm, best_tsc


# ─────────────────────────────────────────────────────────────────────────────
# 1. K-fold consistency
# ─────────────────────────────────────────────────────────────────────────────

def plot_kfold_consistency(kfold_results: dict, save: bool = True) -> plt.Figure:
    """
    Bar chart (mean +/- 1 std) with individual run points overlaid.

    Use in Results: "Performance is stable across folds (std <= X), confirming
    that the small dataset does not introduce a highly fold-dependent bias."
    """
    _apply_style()
    rng = np.random.default_rng(0)

    best_esm, best_tsc = _extract_best_per_run_fold(kfold_results)
    n_folds = len(best_esm)
    n_runs  = len(kfold_results)
    folds   = np.arange(n_folds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"K-Fold Reproducibility  ({n_folds} folds x {n_runs} runs)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    for ax, best, label, color in [
        (axes[0], best_esm, "ESM-2",          _ESM_COLOR),
        (axes[1], best_tsc, "T-scales + BERT", _TSC_COLOR),
    ]:
        means = np.array([np.mean(best[f]) for f in range(n_folds)])
        stds  = np.array([
            np.std(best[f], ddof=1) if len(best[f]) > 1 else 0.0
            for f in range(n_folds)
        ])

        ax.bar(
            folds, means, yerr=stds, capsize=6, alpha=0.55,
            color=color, edgecolor="black", linewidth=1.1,
            error_kw={"linewidth": 1.8, "ecolor": "#333333", "capthick": 1.8},
            zorder=2,
        )
        for f in range(n_folds):
            jitter = rng.uniform(-0.12, 0.12, len(best[f]))
            ax.scatter(
                [f + j for j in jitter], best[f],
                color="black", s=30, zorder=4, alpha=0.65, linewidths=0,
            )

        overall = np.mean([v for vals in best.values() for v in vals])
        ax.axhline(
            overall, color="black", ls="--", lw=1.5, alpha=0.8, zorder=3,
            label=f"Overall mean = {overall:.3f}",
        )

        ax.set_xlabel("Fold", fontsize=_FS_LABEL)
        if ax is axes[0]:
            ax.set_ylabel("Best val RMSE (normalized)", fontsize=_FS_LABEL)
        ax.set_title(label, fontsize=_FS_LABEL, fontweight="bold", pad=8)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f + 1}" for f in folds], fontsize=_FS_TICK)
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.legend(fontsize=_FS_LEGEND, framealpha=0.9)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "kfold_consistency.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_kfold_arch_heatmap(kfold_results: dict, save: bool = True) -> plt.Figure:
    """
    Heatmap: rows = hidden layers, columns = neurons per layer.
    Cell value = mean RMSE averaged over every run and fold.

    Use in Discussion: "Shallow networks (1-2 layers, 32-64 neurons) generalize
    best on our ~100-sample dataset — deeper models overfit."
    """
    _apply_style()

    esm_acc: dict[tuple, list] = {}
    tsc_acc: dict[tuple, list] = {}

    for fold_dict in kfold_results.values():
        for arch_dict in fold_dict.values():
            for (mk, nl, nn), rmse in arch_dict.items():
                key = (nl, nn)
                if mk == "esm":
                    esm_acc.setdefault(key, []).append(rmse)
                else:
                    tsc_acc.setdefault(key, []).append(rmse)

    nr, nc = len(LAYER_OPTIONS), len(NEURON_OPTIONS)

    def _build_matrix(acc):
        M = np.full((nr, nc), np.nan)
        for r, nl in enumerate(LAYER_OPTIONS):
            for c, nn in enumerate(NEURON_OPTIONS):
                vals = acc.get((nl, nn))
                if vals:
                    M[r, c] = np.mean(vals)
        return M

    M_esm = _build_matrix(esm_acc)
    M_tsc = _build_matrix(tsc_acc)
    vmin  = np.nanmin([M_esm, M_tsc])
    vmax  = np.nanmax([M_esm, M_tsc])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Mean Val RMSE by Architecture  (averaged over all runs and folds)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    for ax, M, title in zip(axes, [M_esm, M_tsc], ["ESM-2", "T-scales + BERT"]):
        im = ax.imshow(
            M, aspect="auto", cmap="RdYlGn_r",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        for r in range(nr):
            for c in range(nc):
                val = M[r, c]
                if not np.isnan(val):
                    brightness = (val - vmin) / (vmax - vmin + 1e-9)
                    tcolor = "white" if brightness > 0.58 else "black"
                    ax.text(
                        c, r, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=_FS_TICK, fontweight="bold", color=tcolor,
                    )
        br, bc = np.unravel_index(np.nanargmin(M), M.shape)
        ax.add_patch(plt.Rectangle(
            (bc - 0.49, br - 0.49), 0.98, 0.98,
            fill=False, edgecolor="gold", linewidth=3.5, zorder=5,
        ))
        ax.set_xticks(range(nc))
        ax.set_xticklabels([str(n) for n in NEURON_OPTIONS], fontsize=_FS_TICK)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(
            [f"{l} layer{'s' if l > 1 else ''}" for l in LAYER_OPTIONS],
            fontsize=_FS_TICK,
        )
        ax.set_xlabel("Neurons per hidden layer", fontsize=_FS_LABEL)
        ax.set_title(title, fontsize=_FS_LABEL, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.025, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean val RMSE (normalized)", fontsize=_FS_LABEL)
    cbar.ax.tick_params(labelsize=_FS_TICK)

    if save:
        path = FIGURES_DIR / "kfold_arch_heatmap.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model comparison violin + box
# ─────────────────────────────────────────────────────────────────────────────

def plot_kfold_model_comparison(kfold_results: dict, save: bool = True) -> plt.Figure:
    """
    Violin + box plot comparing the full RMSE distribution of the two models.

    Use in Results: "ESM-2 achieves a median RMSE of X vs Y for T-scales."
    Use in Discussion: relate embedding dimensionality (1280 vs 256) to the
    observed performance gap and the risk of overfitting with few samples.
    """
    _apply_style()
    rng = np.random.default_rng(1)

    best_esm, best_tsc = _extract_best_per_run_fold(kfold_results)
    esm_all = [v for vals in best_esm.values() for v in vals]
    tsc_all = [v for vals in best_tsc.values() for v in vals]

    fig, ax = plt.subplots(figsize=(7, 6))
    data   = [esm_all, tsc_all]
    labels = ["ESM-2", "T-scales\n+ BERT"]
    colors = [_ESM_COLOR, _TSC_COLOR]
    pos    = [1, 2]

    parts = ax.violinplot(
        data, positions=pos, widths=0.55,
        showmedians=False, showextrema=False,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)

    bp = ax.boxplot(
        data, positions=pos, widths=0.22, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.5},
        whiskerprops={"linewidth": 1.6},
        capprops={"linewidth": 1.6},
        flierprops={"marker": "", "linestyle": "none"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for xi, (vals, color) in zip(pos, zip(data, colors)):
        jitter = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(
            [xi + j for j in jitter], vals,
            color="black", s=25, zorder=5, alpha=0.65, linewidths=0,
        )

    for xi, vals in zip(pos, data):
        mu  = np.mean(vals)
        sig = np.std(vals, ddof=1)
        med = np.median(vals)
        ypos = max(vals) + 0.015 * (
            max(max(esm_all), max(tsc_all)) - min(min(esm_all), min(tsc_all))
        )
        ax.text(
            xi, ypos,
            f"median={med:.3f}\nmu={mu:.3f}  sigma={sig:.3f}",
            ha="center", va="bottom", fontsize=_FS_ANNOT,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"),
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=_FS_LABEL)
    ax.set_ylabel("Best val RMSE (normalized)", fontsize=_FS_LABEL)
    ax.set_title(
        "Model Performance Distribution\n(best architecture per run x fold)",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "kfold_model_comparison.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Predicted vs. measured — emission wavelength only (1 x 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kfold_pred_wavelength(oof_data: dict, save: bool = True) -> plt.Figure:
    """
    1 x 2 scatter: OOF predicted vs. true emission wavelength in nm.
    Points are color-coded by spectral class.
    Stat box shows RMSE, MAE, and R^2.

    Use in Results: compare RMSE (nm) and R^2 between ESM-2 and T-scales.
    """
    _apply_style()

    model_specs = [
        ("ESM-2",           "esm"),
        ("T-scales + BERT", "tscales"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    fig.suptitle(
        "Predicted vs. True Emission Wavelength  (Out-of-Fold)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    for ax, (mname, mkey) in zip(axes, model_specs):
        pred = oof_data[mkey]["pred"]
        true = oof_data[mkey]["true"]

        if len(pred) == 0:
            ax.text(0.5, 0.5, "No OOF data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_FS_LABEL)
            ax.set_title(mname, fontsize=_FS_LABEL, fontweight="bold")
            continue

        t = true[:, 0]   # emission wavelength (nm)
        p = pred[:, 0]
        rmse, mae, r2 = _oof_stats(pred, true, col=0)

        mn, mx  = t.min(), t.max()
        margin  = (mx - mn) * 0.04
        pt_clrs = _wl_point_colors(t)

        ax.scatter(t, p, c=pt_clrs, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([mn - margin, mx + margin],
                [mn - margin, mx + margin],
                "k--", lw=1.4, zorder=2)

        _stat_box(ax, rmse, mae, r2, unit="nm")

        ax.set_xlabel(r"True $\lambda_{em}$ (nm)", fontsize=_FS_LABEL)
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title(mname, fontsize=_FS_LABEL, fontweight="bold", pad=8)

    axes[0].set_ylabel(r"Predicted $\lambda_{em}$ (nm)", fontsize=_FS_LABEL)

    # Color legend on the right panel only — kept small to avoid hiding data
    axes[1].legend(
        handles=_color_legend_handles(markersize=6),
        title="Color class", title_fontsize=9,
        fontsize=9, loc="lower right", framealpha=0.88,
        handletextpad=0.4, labelspacing=0.3, borderpad=0.5,
    )

    fig.tight_layout(w_pad=0.5)
    if save:
        path = FIGURES_DIR / "kfold_pred_wavelength.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predicted vs. measured — both targets (2 x 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kfold_pred_vs_actual(oof_data: dict, save: bool = True) -> plt.Figure:
    """
    2 x 2 scatter: OOF predicted vs. true for both targets in original units.
    Points are color-coded by spectral class (using actual wavelength) for all panels.
    Stat box shows RMSE, MAE, and R^2.

    Rows   -> targets  (emission wavelength nm | brightness)
    Cols   -> models   (ESM-2 | T-scales + BERT)

    Use in Results: report RMSE (nm) and R^2 per model x target cell.
    Use in Discussion: compare scatter spread; outliers at rare wavelengths show
    where training data is sparse.
    """
    _apply_style()

    target_labels = ["Emission Wavelength (nm)", "Brightness"]
    target_units  = ["nm", ""]
    model_specs   = [
        ("ESM-2",           "esm"),
        ("T-scales + BERT", "tscales"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Predicted vs. True — All Targets  (Out-of-Fold)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    for col, (mname, mkey) in enumerate(model_specs):
        pred = oof_data[mkey]["pred"]
        true = oof_data[mkey]["true"]

        # Color by actual emission wavelength for both rows
        pt_clrs = _wl_point_colors(true[:, 0]) if len(true) > 0 else []

        for row, (tname, unit) in enumerate(zip(target_labels, target_units)):
            ax = axes[row][col]

            if len(pred) == 0:
                ax.text(0.5, 0.5, "No OOF data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_FS_LABEL)
                continue

            t = true[:, row]
            p = pred[:, row]
            rmse, mae, r2 = _oof_stats(pred, true, col=row)

            mn, mx = t.min(), t.max()
            margin = (mx - mn) * 0.04

            ax.scatter(t, p, c=pt_clrs, s=50, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
            ax.plot([mn - margin, mx + margin],
                    [mn - margin, mx + margin],
                    "k--", lw=1.4, zorder=2)

            _stat_box(ax, rmse, mae, r2, unit=unit)

            ax.set_xlabel(f"True {tname}", fontsize=_FS_LABEL)
            ax.set_ylabel(f"Predicted {tname}", fontsize=_FS_LABEL)
            ax.tick_params(labelsize=_FS_TICK)

            if row == 0:
                ax.set_title(mname, fontsize=_FS_LABEL, fontweight="bold", pad=8)

    # Color legend — bottom-right panel, kept small to avoid hiding data
    axes[1][1].legend(
        handles=_color_legend_handles(markersize=6),
        title="Color class", title_fontsize=9,
        fontsize=9, loc="lower right", framealpha=0.88,
        handletextpad=0.4, labelspacing=0.3, borderpad=0.5,
    )

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "kfold_pred_vs_actual.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Predicted vs. measured — both targets for a single model (1 x 2)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_model_targets(
    oof_data:   dict,
    model_key:  str,
    model_name: str,
    filename:   str,
    save:       bool = True,
) -> plt.Figure:
    """
    1 x 2 scatter for one model: emission wavelength (nm) | brightness.
    Points are color-coded by spectral class.
    """
    _apply_style()

    pred = oof_data[model_key]["pred"]
    true = oof_data[model_key]["true"]

    target_labels = ["Emission Wavelength (nm)", "Brightness"]
    target_units  = ["nm", ""]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{model_name} — Predicted vs. True  (Out-of-Fold)",
        fontsize=_FS_TITLE, fontweight="bold",
    )

    pt_clrs = _wl_point_colors(true[:, 0]) if len(true) > 0 else []

    for ax, (tname, unit), col in zip(axes, zip(target_labels, target_units), [0, 1]):
        if len(pred) == 0:
            ax.text(0.5, 0.5, "No OOF data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_FS_LABEL)
            ax.set_title(tname, fontsize=_FS_LABEL, fontweight="bold")
            continue

        t = true[:, col]
        p = pred[:, col]
        rmse, mae, r2 = _oof_stats(pred, true, col=col)

        mn, mx = t.min(), t.max()
        margin = (mx - mn) * 0.04

        ax.scatter(t, p, c=pt_clrs, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([mn - margin, mx + margin],
                [mn - margin, mx + margin],
                "k--", lw=1.4, zorder=2)

        _stat_box(ax, rmse, mae, r2, unit=unit)

        ax.set_xlabel(f"True {tname}", fontsize=_FS_LABEL)
        ax.set_ylabel(f"Predicted {tname}", fontsize=_FS_LABEL)
        ax.set_title(tname, fontsize=_FS_LABEL, fontweight="bold", pad=8)
        ax.tick_params(labelsize=_FS_TICK)

    # Color legend on the right panel — kept small to avoid hiding data
    axes[1].legend(
        handles=_color_legend_handles(markersize=6),
        title="Color class", title_fontsize=9,
        fontsize=9, loc="lower right", framealpha=0.88,
        handletextpad=0.4, labelspacing=0.3, borderpad=0.5,
    )

    fig.tight_layout(w_pad=2.0)
    if save:
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


def plot_kfold_esm_targets(oof_data: dict, save: bool = True) -> plt.Figure:
    """1 x 2: ESM-2 predicted vs. true — emission wavelength and brightness."""
    return _plot_model_targets(
        oof_data, "esm", "ESM-2", "kfold_esm_targets.png", save=save
    )


def plot_kfold_tsc_targets(oof_data: dict, save: bool = True) -> plt.Figure:
    """1 x 2: T-scales + BERT predicted vs. true — emission wavelength and brightness."""
    return _plot_model_targets(
        oof_data, "tscales", "T-scales + BERT", "kfold_tsc_targets.png", save=save
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-protein error figures — best and worst predicted proteins annotated
# ─────────────────────────────────────────────────────────────────────────────

def _draw_protein_panel(
    ax,
    t: np.ndarray,
    p: np.ndarray,
    ae: np.ndarray,
    labels,
    x_label: str,
    y_label: str,
    title: str,
    unit: str = "",
    true_wl: "np.ndarray | None" = None,
    n_annotate: int = 5,
) -> None:
    """Draw one annotated scatter panel for per-protein error figures."""
    color_wl = true_wl if true_wl is not None else t
    pt_clrs  = _wl_point_colors(color_wl)

    mn, mx  = t.min(), t.max()
    margin  = (mx - mn) * 0.04
    dy_mag  = (mx - mn) * 0.12 if (mx - mn) > 0 else 1.0

    ax.scatter(t, p, c=pt_clrs, s=45, alpha=0.65,
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.plot([mn - margin, mx + margin],
            [mn - margin, mx + margin],
            "k--", lw=1.4, zorder=2)

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(ae))
    ss_r = np.sum((p - t) ** 2)
    ss_t = np.sum((t - t.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-12))
    _stat_box(ax, rmse, mae, r2, unit=unit)

    sorted_idx = np.argsort(ae)
    best_idx   = sorted_idx[:n_annotate]
    worst_idx  = sorted_idx[-n_annotate:]
    u_str = f" {unit}" if unit else ""

    for group_idx, color in [(best_idx, "#1a7a1a"), (worst_idx, "#b80000")]:
        for i in group_idx:
            xi, yi = float(t[i]), float(p[i])
            dy = dy_mag if (yi - xi) >= 0 else -dy_mag
            lbl = f"{labels[i]}\n|{chr(916)}|={ae[i]:.2f}{u_str}"
            ax.annotate(
                lbl,
                xy=(xi, yi), xytext=(xi, yi + dy),
                fontsize=8, color=color, fontweight="bold",
                ha="center", va="bottom" if dy > 0 else "top",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec=color, lw=0.8),
                zorder=6,
            )
            ax.scatter([xi], [yi], c=color, s=80, zorder=7,
                       edgecolors="white", linewidths=0.6)

    ax.set_xlabel(x_label, fontsize=_FS_LABEL)
    ax.set_ylabel(y_label, fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_LABEL, fontweight="bold", pad=8)
    ax.tick_params(labelsize=_FS_TICK)


def _protein_legend(ax, n_annotate: int = 5) -> None:
    """Attach color-class + best/worst legend to an axis."""
    extra = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#1a7a1a",
                      markersize=7, label=f"Top {n_annotate} best predicted"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#b80000",
                      markersize=7, label=f"Top {n_annotate} worst predicted"),
    ]
    ax.legend(
        handles=_color_legend_handles(markersize=6) + extra,
        title="Color class", title_fontsize=9,
        fontsize=9, loc="lower right", framealpha=0.88,
        handletextpad=0.4, labelspacing=0.3, borderpad=0.5, ncol=1,
    )


def plot_kfold_protein_errors(oof_df: "pd.DataFrame", save: bool = True) -> plt.Figure:
    """1 x 2: ESM-2 wavelength | T-scales wavelength — per-protein annotations."""
    _apply_style()
    N  = 5
    lb = oof_df["PDB code"].values
    t  = oof_df["True Wavelength"].values

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
    fig.suptitle(
        "Predicted vs. True Emission Wavelength — Per-Protein View",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    for ax, (mname, pred_col, err_col) in zip(axes, [
        ("ESM-2",           "ESM Pred Wavelength", "ESM Abs Err Wavelength"),
        ("T-scales + BERT", "TSC Pred Wavelength", "TSC Abs Err Wavelength"),
    ]):
        _draw_protein_panel(
            ax, t=t, p=oof_df[pred_col].values, ae=oof_df[err_col].values,
            labels=lb, x_label=r"True $\lambda_{em}$ (nm)",
            y_label=r"Predicted $\lambda_{em}$ (nm)",
            title=mname, unit="nm", true_wl=t, n_annotate=N,
        )
    axes[0].set_ylabel(r"Predicted $\lambda_{em}$ (nm)", fontsize=_FS_LABEL)
    _protein_legend(axes[1], N)
    fig.tight_layout(w_pad=1.5)
    if save:
        path = FIGURES_DIR / "kfold_protein_errors.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


def plot_kfold_protein_errors_esm(oof_df: "pd.DataFrame", save: bool = True) -> plt.Figure:
    """1 x 2: ESM-2 wavelength | ESM-2 brightness — per-protein annotations."""
    _apply_style()
    N      = 5
    lb     = oof_df["PDB code"].values
    true_wl = oof_df["True Wavelength"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "ESM-2 — Predicted vs. True, Per-Protein View",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    _draw_protein_panel(
        axes[0],
        t=oof_df["True Wavelength"].values, p=oof_df["ESM Pred Wavelength"].values,
        ae=oof_df["ESM Abs Err Wavelength"].values, labels=lb,
        x_label=r"True $\lambda_{em}$ (nm)", y_label=r"Predicted $\lambda_{em}$ (nm)",
        title="Emission Wavelength", unit="nm", true_wl=true_wl, n_annotate=N,
    )
    _draw_protein_panel(
        axes[1],
        t=oof_df["True Brightness"].values, p=oof_df["ESM Pred Brightness"].values,
        ae=oof_df["ESM Abs Err Brightness"].values, labels=lb,
        x_label="True Brightness", y_label="Predicted Brightness",
        title="Brightness", unit="", true_wl=true_wl, n_annotate=N,
    )
    _protein_legend(axes[1], N)
    fig.tight_layout(w_pad=2.0)
    if save:
        path = FIGURES_DIR / "kfold_protein_errors_esm.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


def plot_kfold_protein_errors_tsc(oof_df: "pd.DataFrame", save: bool = True) -> plt.Figure:
    """1 x 2: T-scales wavelength | T-scales brightness — per-protein annotations."""
    _apply_style()
    N      = 5
    lb     = oof_df["PDB code"].values
    true_wl = oof_df["True Wavelength"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "T-scales + BERT — Predicted vs. True, Per-Protein View",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    _draw_protein_panel(
        axes[0],
        t=oof_df["True Wavelength"].values, p=oof_df["TSC Pred Wavelength"].values,
        ae=oof_df["TSC Abs Err Wavelength"].values, labels=lb,
        x_label=r"True $\lambda_{em}$ (nm)", y_label=r"Predicted $\lambda_{em}$ (nm)",
        title="Emission Wavelength", unit="nm", true_wl=true_wl, n_annotate=N,
    )
    _draw_protein_panel(
        axes[1],
        t=oof_df["True Brightness"].values, p=oof_df["TSC Pred Brightness"].values,
        ae=oof_df["TSC Abs Err Brightness"].values, labels=lb,
        x_label="True Brightness", y_label="Predicted Brightness",
        title="Brightness", unit="", true_wl=true_wl, n_annotate=N,
    )
    _protein_legend(axes[1], N)
    fig.tight_layout(w_pad=2.0)
    if save:
        path = FIGURES_DIR / "kfold_protein_errors_tsc.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


def plot_kfold_protein_errors_grid(oof_df: "pd.DataFrame", save: bool = True) -> plt.Figure:
    """2 x 2: (ESM-2 wl | T-scales wl) / (ESM-2 br | T-scales br) — per-protein annotations."""
    _apply_style()
    N      = 5
    lb     = oof_df["PDB code"].values
    true_wl = oof_df["True Wavelength"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Predicted vs. True — Per-Protein View  (Both Models & Targets)",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    panels = [
        (0, 0, "True Wavelength", "ESM Pred Wavelength", "ESM Abs Err Wavelength",
         r"True $\lambda_{em}$ (nm)", r"Predicted $\lambda_{em}$ (nm)",
         "ESM-2 — Wavelength", "nm"),
        (0, 1, "True Wavelength", "TSC Pred Wavelength", "TSC Abs Err Wavelength",
         r"True $\lambda_{em}$ (nm)", r"Predicted $\lambda_{em}$ (nm)",
         "T-scales + BERT — Wavelength", "nm"),
        (1, 0, "True Brightness", "ESM Pred Brightness", "ESM Abs Err Brightness",
         "True Brightness", "Predicted Brightness", "ESM-2 — Brightness", ""),
        (1, 1, "True Brightness", "TSC Pred Brightness", "TSC Abs Err Brightness",
         "True Brightness", "Predicted Brightness", "T-scales + BERT — Brightness", ""),
    ]
    for row, col, t_col, p_col, err_col, xlbl, ylbl, subtitle, unit in panels:
        _draw_protein_panel(
            axes[row][col],
            t=oof_df[t_col].values, p=oof_df[p_col].values,
            ae=oof_df[err_col].values, labels=lb,
            x_label=xlbl, y_label=ylbl, title=subtitle, unit=unit,
            true_wl=true_wl, n_annotate=N,
        )
    _protein_legend(axes[1][1], N)
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "kfold_protein_errors_grid.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig


def plot_kfold_protein_errors_brightness(oof_df: "pd.DataFrame", save: bool = True) -> plt.Figure:
    """1 x 2: ESM-2 brightness | T-scales brightness — per-protein annotations."""
    _apply_style()
    N      = 5
    lb     = oof_df["PDB code"].values
    true_wl = oof_df["True Wavelength"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    fig.suptitle(
        "Predicted vs. True Brightness — Per-Protein View",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    for ax, (mname, pred_col, err_col) in zip(axes, [
        ("ESM-2",           "ESM Pred Brightness", "ESM Abs Err Brightness"),
        ("T-scales + BERT", "TSC Pred Brightness", "TSC Abs Err Brightness"),
    ]):
        _draw_protein_panel(
            ax,
            t=oof_df["True Brightness"].values, p=oof_df[pred_col].values,
            ae=oof_df[err_col].values, labels=lb,
            x_label="True Brightness", y_label="Predicted Brightness",
            title=mname, unit="", true_wl=true_wl, n_annotate=N,
        )
    axes[0].set_ylabel("Predicted Brightness", fontsize=_FS_LABEL)
    _protein_legend(axes[1], N)
    fig.tight_layout(w_pad=1.5)
    if save:
        path = FIGURES_DIR / "kfold_protein_errors_brightness.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved -> {path}")
    return fig
