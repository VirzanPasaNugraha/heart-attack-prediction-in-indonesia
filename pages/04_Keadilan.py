import streamlit as st

from utils.fairness_calib import (
    GROUP_META,
    disparity_label,
    disparity_summary,
    group_metrics,
)
import plotly.graph_objects as go
from utils.styling import inject_base_css, section_header

st.set_page_config(page_title="Keadilan — CardioSense AI", page_icon="⚖️", layout="wide")
inject_base_css()

section_header("Audit Model", "Keadilan Antar Kelompok")
st.caption(
    "Semua angka di halaman ini dihitung langsung dari **158.355 prediksi out-of-fold (OOF)** — "
    "baris yang sama persis dengan yang menghasilkan metrik di halaman Model Performance — "
    "bukan estimasi kasar. OOF berarti tiap sampel diprediksi oleh fold model yang tidak melihat "
    "sampel itu saat training, jadi hasilnya representatif seperti data uji yang belum pernah dilihat model."
)

st.markdown("#### Pilih Atribut Sensitif")
group_key = st.radio(
    "Bandingkan performa model antar kelompok pada atribut:",
    list(GROUP_META.keys()),
    format_func=lambda k: GROUP_META[k]["label"],
    horizontal=True,
    label_visibility="collapsed",
)

gm = group_metrics(group_key)
gm = gm.copy()
gm["group_label"] = gm["group"].astype(str).map(GROUP_META[group_key]["display"]).fillna(gm["group"].astype(str))
disp = disparity_summary(gm)

st.markdown(f"#### Ringkasan Disparitas — {GROUP_META[group_key]['label']}")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Demographic Parity Diff", f"{disp['demographic_parity_diff']:.3f}")
d1.caption(f"Disparitas: **{disparity_label(disp['demographic_parity_diff'])}**")
d2.metric("Disparate Impact Ratio", f"{disp['disparate_impact_ratio']:.3f}")
d2.caption("✅ Aman (≥0.8)" if disp["disparate_impact_ratio"] >= 0.8 else "⚠️ Di bawah ambang 0.8")
d3.metric("Equal Opportunity Diff", f"{disp['equal_opportunity_diff']:.3f}")
d3.caption(f"Selisih TPR: **{disparity_label(disp['equal_opportunity_diff'])}**")
d4.metric("Equalized Odds Diff", f"{disp['equalized_odds_diff']:.3f}")
d4.caption(f"Selisih TPR/FPR: **{disparity_label(disp['equalized_odds_diff'])}**")
d5.metric("Predictive Parity Diff", f"{disp['predictive_parity_diff']:.3f}")
d5.caption(f"Selisih PPV: **{disparity_label(disp['predictive_parity_diff'])}**")

with st.expander("ℹ️ Arti tiap metrik keadilan"):
    st.markdown(
        """
        - **Demographic Parity Diff** — selisih proporsi yang diprediksi *Heart_Attack* antar
          kelompok. Idealnya kecil kalau atribut tsb seharusnya tidak memengaruhi keputusan.
        - **Disparate Impact Ratio** — rasio *selection rate* kelompok terendah terhadap
          tertinggi. Pedoman "four-fifths rule" (EEOC) menandai rasio **< 0.8** sebagai indikasi
          disparitas yang perlu ditelusuri.
        - **Equal Opportunity Diff** — selisih *True Positive Rate* (sensitivitas) antar
          kelompok: apakah model sama baiknya menangkap kasus positif riil di semua kelompok.
        - **Equalized Odds Diff** — nilai terbesar antara selisih TPR dan selisih *False
          Positive Rate*; versi lebih ketat dari Equal Opportunity (Hardt et al., 2016).
        - **Predictive Parity Diff** — selisih *precision* (PPV): dari yang diprediksi
          berisiko, apakah proporsi yang benar-benar berisiko konsisten antar kelompok.
        """
    )

if group_key == "age_group":
    st.info(
        "⚠️ **Catatan interpretasi khusus untuk usia:** usia secara klinis memang berkorelasi "
        "kuat dengan risiko riil serangan jantung (prevalensi aktual naik dari "
        f"{gm.loc[gm['group_label']=='<45 tahun','base_rate_actual'].values[0]:.1%} pada "
        "kelompok <45 tahun menjadi "
        f"{gm.loc[gm['group_label']=='65+ tahun','base_rate_actual'].values[0]:.1%} pada "
        "kelompok 65+ tahun). Karena itu, Demographic Parity Diff yang besar di sini "
        "kemungkinan besar mencerminkan **perbedaan risiko riil**, bukan bias algoritmik. "
        "Equal Opportunity Diff dan kalibrasi per kelompok (di bawah) adalah lensa yang lebih "
        "tepat untuk menilai keadilan pada atribut seperti ini.",
        icon="⚠️",
    )

st.markdown("#### Perbandingan Metrik Antar Kelompok")
metric_cols = ["selection_rate_pred", "TPR", "FPR", "PPV"]
metric_labels = {
    "selection_rate_pred": "Selection Rate",
    "TPR": "TPR (Sensitivitas)",
    "FPR": "FPR",
    "PPV": "PPV (Precision)",
}
colors = ["#0E4749", "#1F7A54", "#C23B32", "#C98A1D"]

fig2 = go.Figure()
for col, color in zip(metric_cols, colors):
    fig2.add_trace(
        go.Bar(
            x=gm["group_label"], y=gm[col], name=metric_labels[col],
            marker=dict(color=color),
        )
    )
fig2.update_layout(
    barmode="group", height=420,
    yaxis=dict(title="Nilai Metrik", tickformat=".0%"),
    margin=dict(l=10, r=20, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Kalibrasi per Kelompok (Brier Score)")
st.caption(
    "Semakin rendah semakin baik — bandingkan apakah ada kelompok yang diprediksi jauh lebih "
    "buruk secara probabilistik dibanding kelompok lain. Untuk audit kalibrasi model secara "
    "keseluruhan (bukan per kelompok), lihat halaman **Kalibrasi**."
)
fig3 = go.Figure(
    go.Bar(x=gm["group_label"], y=gm["brier"], marker=dict(color="#1B6B73"), text=gm["brier"].round(4), textposition="outside")
)
fig3.update_layout(
    height=320, yaxis=dict(title="Brier Score"),
    margin=dict(l=10, r=20, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", size=12),
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("Lihat tabel lengkap metrik per kelompok"):
    display = gm[["group_label", "n", "base_rate_actual", "selection_rate_pred", "TPR", "FPR", "PPV", "accuracy", "AUC", "brier"]].rename(
        columns={
            "group_label": "Kelompok", "n": "Jumlah Sampel",
            "base_rate_actual": "Prevalensi Aktual", "selection_rate_pred": "Selection Rate",
            "TPR": "TPR", "FPR": "FPR", "PPV": "PPV",
            "accuracy": "Akurasi", "AUC": "ROC AUC", "brier": "Brier Score",
        }
    )
    st.dataframe(
        display.style.format({
            "Prevalensi Aktual": "{:.1%}", "Selection Rate": "{:.1%}",
            "TPR": "{:.1%}", "FPR": "{:.1%}", "PPV": "{:.1%}",
            "Akurasi": "{:.1%}", "ROC AUC": "{:.3f}", "Brier Score": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()
st.page_link("pages/05_Kalibrasi.py", label="Lihat audit Kalibrasi model secara keseluruhan →", icon="📏")
st.caption(
    "Definisi metrik keadilan mengacu pada literatur standar: Demographic Parity "
    "(Dwork dkk., 2012), Equal Opportunity & Equalized Odds (Hardt dkk., 2016), dan aturan "
    "*four-fifths* untuk disparate impact (pedoman EEOC). Metrik-metrik ini adalah alat bantu "
    "deteksi disparitas, bukan vonis otomatis — interpretasi tetap perlu mempertimbangkan "
    "konteks klinis tiap atribut."
)
