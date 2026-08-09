import plotly.graph_objects as go
import streamlit as st

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

st.caption(
    "Ringkasan ini berasal langsung dari `final_summary_report.json` hasil training. "
    "Kurva ROC/precision-recall aktual tidak ditampilkan karena data prediksi mentah "
    "pada set uji tidak disertakan dalam berkas yang diunggah — hanya metrik ringkasannya."
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
