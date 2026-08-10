from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

from utils.feature_meta import METRIC_INFO
from utils.model_loader import load_report
from utils.styling import inject_base_css, section_header

st.set_page_config(page_title="Performa Model — CardioSense AI", page_icon="📊", layout="wide")
inject_base_css()

section_header("Evaluasi Model", "Performa CardioSense AI")
report = load_report()
st.caption(f"Model terbaik: **{report['Best_Model']}**, dievaluasi pada data uji dari total {report['Total_Samples']:,} sampel.")

# Kunci internal tetap Bahasa Inggris (dipakai untuk mengambil nilai dari report
# & mencocokkan ke METRIC_INFO), label yang ditampilkan ke pengguna dalam Bahasa Indonesia.
metrics = {
    "Balanced Accuracy": report["Best_Model_Balanced_Accuracy"],
    "Accuracy": report["Best_Model_Accuracy"],
    "F1 Macro": report["Best_Model_F1_Macro"],
    "MCC": report["Best_Model_MCC"],
    "ROC AUC": report["Best_Model_ROC_AUC"],
    "Average Precision": report["Best_Model_Average_Precision"],
}
label_id = {
    "Balanced Accuracy": "Akurasi Seimbang",
    "Accuracy": "Akurasi",
    "F1 Macro": "F1 Score (Macro)",
    "MCC": "MCC (Korelasi Matthews)",
    "ROC AUC": "ROC AUC",
    "Average Precision": "Rata-rata Presisi",
}

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### Radar Metrik")
    categories = [label_id[k] for k in metrics.keys()]
    values = list(metrics.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#0E4749", width=2),
            fillcolor="rgba(14,71,73,0.18)",
            marker=dict(color="#E0483F", size=6),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### Detail Metrik")
    for name, val in metrics.items():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(label_id[name], f"{val:.3f}" if name not in ("Balanced Accuracy", "Accuracy") else f"{val:.1%}")
        with c2:
            st.markdown(
                f"<div style='color:var(--cs-ink-soft); font-size:0.85rem; padding-top:0.9rem;'>{METRIC_INFO[name]}</div>",
                unsafe_allow_html=True,
            )

st.divider()
section_header("Konfigurasi Model", "Detail Dataset & Hyperparameter")

d1, d2, d3 = st.columns(3)
d1.metric("Total Sampel", f"{report['Total_Samples']:,}")
d2.metric("Total Fitur", report["Total_Features"])
d3.metric("Total Kelas", report["Total_Classes"])
st.caption(f"Label kelas target: {', '.join(report['Class_Labels'])}")

st.markdown("#### Hyperparameter Terbaik (hasil tuning)")
params = report["Best_Parameters"]
pcols = st.columns(len(params))
param_display = {
    "clf__learning_rate": "Laju Pembelajaran",
    "clf__max_depth": "Kedalaman Maksimum",
    "clf__min_child_samples": "Sampel Minimum per Daun",
    "clf__n_estimators": "Jumlah Estimator (Pohon)",
    "clf__num_leaves": "Jumlah Daun (Leaves)",
}
for col, (k, v) in zip(pcols, params.items()):
    col.metric(param_display.get(k, k), v)

# ---------------------------------------------------------------------------
# VALIDASI INDEPENDEN — confusion matrix, kurva ROC & PR dari data OOF asli
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OOF_PATH = BASE_DIR / "data" / "oof_predictions_slim.parquet"


@st.cache_data
def load_oof_slim() -> pd.DataFrame:
    return pd.read_parquet(OOF_PATH)


st.divider()
section_header("Validasi Independen", "Confusion Matrix & Kurva Evaluasi")
st.caption(
    "Dihitung langsung dari **158.355 prediksi out-of-fold (OOF)** di `oof_predictions_slim.parquet` — "
    "baris yang sama persis dengan yang dipakai di halaman Keadilan & Kalibrasi — bukan estimasi. "
    "OOF berarti tiap sampel diprediksi oleh fold model yang tidak melihat sampel itu saat training."
)

oof = load_oof_slim()
y_true = oof["y_true"].to_numpy()
y_proba = oof["proba_Heart_Attack"].to_numpy()

threshold = st.slider(
    "Ambang batas klasifikasi (threshold)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.01,
    help="Geser untuk melihat trade-off sensitivitas vs presisi pada ambang batas selain 0.5 default.",
)
y_pred_t = (y_proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0

cm_col, curve_col = st.columns([1, 1.3], gap="large")

with cm_col:
    st.markdown(f"##### Confusion Matrix @ threshold = {threshold:.2f}")
    z = [[tn, fp], [fn, tp]]
    z_text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Prediksi: No_Heart_Attack", "Prediksi: Heart_Attack"],
            y=["Aktual: No_Heart_Attack", "Aktual: Heart_Attack"],
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=14, family="IBM Plex Mono", color="white"),
            colorscale=[[0, "#8FCFC7"], [1, "#0E4749"]],
            showscale=False,
        )
    )
    fig_cm.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        font=dict(family="IBM Plex Sans", size=11),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        f"""
        <div style="display:flex; gap:0.6rem; margin-top:0.5rem;">
            <div style="flex:1; background:var(--cs-surface); border:1px solid var(--cs-border); border-radius:10px; padding:0.6rem 0.5rem; text-align:center;">
                <div style="font-size:0.72rem; color:var(--cs-ink-soft);">Sensitivitas</div>
                <div style="font-family:var(--cs-font-mono); font-size:1.15rem; font-weight:600; color:var(--cs-primary);">{sensitivity:.1%}</div>
            </div>
            <div style="flex:1; background:var(--cs-surface); border:1px solid var(--cs-border); border-radius:10px; padding:0.6rem 0.5rem; text-align:center;">
                <div style="font-size:0.72rem; color:var(--cs-ink-soft);">Spesifisitas</div>
                <div style="font-family:var(--cs-font-mono); font-size:1.15rem; font-weight:600; color:var(--cs-primary);">{specificity:.1%}</div>
            </div>
            <div style="flex:1; background:var(--cs-surface); border:1px solid var(--cs-border); border-radius:10px; padding:0.6rem 0.5rem; text-align:center;">
                <div style="font-size:0.72rem; color:var(--cs-ink-soft);">Presisi (PPV)</div>
                <div style="font-family:var(--cs-font-mono); font-size:1.15rem; font-weight:600; color:var(--cs-primary);">{precision_val:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Sensitivitas = dari semua kasus *Heart_Attack* sungguhan, berapa persen tertangkap model. "
        "Spesifisitas = dari semua kasus *No_Heart_Attack* sungguhan, berapa persen benar diprediksi aman. "
        "Untuk alat skrining kesehatan, sensitivitas tinggi umumnya lebih diprioritaskan — geser slider ke kiri "
        "(ambang lebih rendah) untuk menaikkan sensitivitas dengan trade-off presisi lebih rendah."
    )

with curve_col:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
    ap_val = average_precision_score(y_true, y_proba)
    prevalence = float(y_true.mean())

    tab_roc, tab_pr = st.tabs(["Kurva ROC", "Kurva Precision-Recall"])

    with tab_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#DCE3E8", dash="dash"), name="Acak (AUC=0.5)"))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#0E4749", width=2.5), name=f"LightGBM (AUC={roc_auc_val:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[1 - specificity], y=[sensitivity], mode="markers", marker=dict(color="#E0483F", size=11, symbol="diamond"), name=f"Threshold={threshold:.2f}"))
        fig_roc.update_layout(
            height=380,
            xaxis=dict(title="False Positive Rate (1 - Spesifisitas)", range=[0, 1]),
            yaxis=dict(title="True Positive Rate (Sensitivitas)", range=[0, 1]),
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(family="IBM Plex Sans", size=11),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_pr:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines", line=dict(color="#DCE3E8", dash="dash"), name=f"Baseline (prevalensi={prevalence:.2f})"))
        fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", line=dict(color="#0E4749", width=2.5), name=f"LightGBM (AP={ap_val:.3f})"))
        fig_pr.add_trace(go.Scatter(x=[sensitivity], y=[precision_val], mode="markers", marker=dict(color="#E0483F", size=11, symbol="diamond"), name=f"Threshold={threshold:.2f}"))
        fig_pr.update_layout(
            height=380,
            xaxis=dict(title="Recall (Sensitivitas)", range=[0, 1]),
            yaxis=dict(title="Precision (PPV)", range=[0, 1]),
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(family="IBM Plex Sans", size=11),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

st.caption(
    "Titik merah pada kedua kurva menandai posisi ambang batas yang sedang dipilih di slider atas. "
    "Kurva ROC & PR dihitung dari seluruh rentang ambang batas 0-1, jadi tidak berubah saat slider "
    "digeser — hanya posisi titiknya yang bergerak sepanjang kurva."
)

st.divider()
st.page_link(
    "pages/04_Keadilan.py",
    label="Lihat audit Keadilan model (dihitung dari prediksi OOF asli) →",
    icon="⚖️",
)
st.page_link(
    "pages/05_Kalibrasi.py",
    label="Lihat audit Kalibrasi model (dihitung dari prediksi OOF asli) →",
    icon="📏",
)