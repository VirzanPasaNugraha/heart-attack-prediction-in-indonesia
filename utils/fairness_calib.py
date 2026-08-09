"""
Utility untuk audit keadilan (fairness) & kalibrasi model CardioSense AI.

PENTING soal sumber data: semua angka di halaman ini dihitung LANGSUNG dari
`data/oof_predictions_slim.parquet` - yaitu prediksi out-of-fold (OOF) untuk
seluruh 158.355 sampel dataset (bukan estimasi/simulasi, dan bukan hasil
prediksi di data training yang sama yang dipakai modelnya - OOF berarti tiap
baris diprediksi oleh fold model yang TIDAK melihat baris itu saat training).
Angka metrik performa di sini cocok persis dengan `final_summary_report.json`
yang dipakai halaman Model Performance, jadi konsisten satu sama lain.

Definisi metrik keadilan yang dipakai mengikuti literatur standar fairness
in ML:
- Demographic Parity (Dwork et al., 2012): selisih tingkat prediksi positif
  antar kelompok.
- Equal Opportunity & Equalized Odds (Hardt et al., 2016): selisih True
  Positive Rate (dan False Positive Rate) antar kelompok.
- Disparate Impact Ratio & aturan "four-fifths" (pedoman EEOC AS): rasio
  selection rate kelompok terendah terhadap tertinggi; < 0.8 lazim ditandai
  sebagai indikasi disparitas yang perlu ditelusuri lebih jauh.

CATATAN INTERPRETASI: metrik-metrik ini adalah alat bantu deteksi disparitas,
BUKAN vonis otomatis "model ini bias". Untuk atribut yang secara klinis
memang berkorelasi dengan risiko sebenarnya (mis. usia untuk penyakit
jantung), perbedaan demographic parity yang besar bisa jadi mencerminkan
perbedaan prevalensi riil, bukan bias algoritmik - kalibrasi per kelompok
dan equalized odds biasanya jadi lensa yang lebih tepat untuk kasus seperti
itu dibanding demographic parity mentah.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OOF_PATH = BASE_DIR / "data" / "oof_predictions_slim.parquet"

AGE_BINS = [0, 45, 65, 200]
AGE_LABELS = ["<45 tahun", "45-64 tahun", "65+ tahun"]

GROUP_META = {
    "gender": {
        "label": "Jenis Kelamin",
        "display": {"Female": "Perempuan", "Male": "Laki-laki"},
    },
    "region": {
        "label": "Wilayah Tempat Tinggal",
        "display": {"Rural": "Pedesaan", "Urban": "Perkotaan"},
    },
    "income_level": {
        "label": "Tingkat Pendapatan",
        "display": {"Low": "Rendah", "Middle": "Menengah", "High": "Tinggi"},
    },
    "age_group": {
        "label": "Kelompok Usia",
        "display": {lbl: lbl for lbl in AGE_LABELS},
    },
}


@st.cache_data
def load_oof() -> pd.DataFrame:
    """Baca prediksi OOF (158.355 baris) & tambahkan kolom kelompok usia."""
    df = pd.read_parquet(OOF_PATH)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


# ---------------------------------------------------------------------------
# KALIBRASI
# ---------------------------------------------------------------------------
@st.cache_data
def reliability_table(n_bins: int = 10) -> pd.DataFrame:
    """Tabel reliability diagram: rata-rata probabilitas prediksi vs frekuensi
    observasi aktual per bin probabilitas (dipakai untuk plot & hitung ECE/MCE)."""
    df = load_oof()
    y_true = df["y_true"].to_numpy()
    proba = df["proba_Heart_Attack"].to_numpy()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1], right=True)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        rows.append(
            {
                "bin_low": bins[b],
                "bin_high": bins[b + 1],
                "bin_label": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": n_b,
                "mean_predicted": float(proba[mask].mean()),
                "observed_freq": float(y_true[mask].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = (out["observed_freq"] - out["mean_predicted"]).abs()
    return out


@st.cache_data
def calibration_summary(n_bins: int = 10) -> dict:
    """Ringkasan skalar kalibrasi: Brier score, ECE, MCE, slope & intercept
    kalibrasi (dari regresi logistik y_true ~ logit(p_hat))."""
    df = load_oof()
    y_true = df["y_true"].to_numpy()
    proba = df["proba_Heart_Attack"].to_numpy()

    rel = reliability_table(n_bins=n_bins)
    n_total = len(y_true)
    ece = float((rel["n"] / n_total * rel["gap"]).sum())
    mce = float(rel["gap"].max())
    brier = float(brier_score_loss(y_true, proba))

    eps = 1e-6
    p_clip = np.clip(proba, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_true)

    return {
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "cal_slope": float(lr.coef_[0][0]),
        "cal_intercept": float(lr.intercept_[0]),
        "n_samples": n_total,
    }


# ---------------------------------------------------------------------------
# KEADILAN (FAIRNESS)
# ---------------------------------------------------------------------------
@st.cache_data
def group_metrics(group_col: str) -> pd.DataFrame:
    """Metrik performa per kelompok untuk satu atribut sensitif."""
    df = load_oof()
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        pr = sub["proba_Heart_Attack"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        rows.append(
            {
                "group": g,
                "n": len(sub),
                "base_rate_actual": float(yt.mean()),
                "selection_rate_pred": float(yp.mean()),
                "TPR": tpr,
                "FPR": fpr,
                "PPV": ppv,
                "accuracy": accuracy_score(yt, yp),
                "AUC": roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan,
                "brier": brier_score_loss(yt, pr),
            }
        )
    return pd.DataFrame(rows)


def disparity_summary(gm: pd.DataFrame) -> dict:
    """Ringkasan disparitas antar kelompok dari tabel group_metrics()."""
    sel = gm["selection_rate_pred"]
    tpr = gm["TPR"]
    fpr = gm["FPR"]
    ppv = gm["PPV"]
    return {
        "demographic_parity_diff": float(sel.max() - sel.min()),
        "disparate_impact_ratio": float(sel.min() / sel.max()) if sel.max() > 0 else np.nan,
        "equal_opportunity_diff": float(tpr.max() - tpr.min()),
        "equalized_odds_diff": float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())),
        "predictive_parity_diff": float(ppv.max() - ppv.min()),
    }


def disparity_label(value: float, thresholds=(0.05, 0.10)) -> str:
    """Beri label kualitatif kasar untuk selisih (diff) 0-1: kecil/sedang/besar."""
    lo, hi = thresholds
    if value < lo:
        return "Kecil"
    if value < hi:
        return "Sedang"
    return "Besar"
