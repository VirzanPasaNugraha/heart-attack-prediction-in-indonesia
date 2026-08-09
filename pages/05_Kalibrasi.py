import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.fairness_calib import calibration_summary, reliability_table
from utils.styling import inject_base_css, section_header

st.set_page_config(page_title="Kalibrasi — CardioSense AI", page_icon="📏", layout="wide")
inject_base_css()

section_header("Audit Model", "Kalibrasi Model")
st.caption(
    "Semua angka di halaman ini dihitung langsung dari **158.355 prediksi out-of-fold (OOF)** — "
    "baris yang sama persis dengan yang menghasilkan metrik di halaman Model Performance — "
    "bukan estimasi kasar. OOF berarti tiap sampel diprediksi oleh fold model yang tidak melihat "
    "sampel itu saat training, jadi hasilnya representatif seperti data uji yang belum pernah dilihat model."
)

cal = calibration_summary()
rel = reliability_table()

st.markdown("#### Ringkasan Kalibrasi")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Brier Score", f"{cal['brier']:.4f}")
c2.metric("ECE", f"{cal['ece']:.4f}")
c3.metric("MCE", f"{cal['mce']:.4f}")
c4.metric("Slope Kalibrasi", f"{cal['cal_slope']:.3f}")
c5.metric("Intercept Kalibrasi", f"{cal['cal_intercept']:.3f}")

with st.expander("ℹ️ Arti tiap metrik kalibrasi"):
    st.markdown(
        """
        - **Brier Score** — rata-rata kuadrat selisih antara probabilitas prediksi dan label
          aktual (0-1). 0 = sempurna, 0.25 = setara menebak 50/50 terus-menerus.
        - **ECE (Expected Calibration Error)** — rata-rata tertimbang selisih antara
          probabilitas yang diprediksi model dan frekuensi kejadian aktual, dihitung per
          bin probabilitas. Semakin dekat 0 semakin baik.
        - **MCE (Maximum Calibration Error)** — selisih terbesar di satu bin manapun; menyoroti
          titik terburuk yang mungkin "tersembunyi" oleh rata-rata ECE.
        - **Slope Kalibrasi** — dari regresi logistik label aktual terhadap log-odds prediksi.
          Ideal = 1. Di bawah 1 → model *under-confident* pada probabilitas ekstrem, di atas 1
          → *over-confident*.
        - **Intercept Kalibrasi** — ideal = 0. Menunjukkan bias sistematis model menaksir
          terlalu tinggi (positif) atau terlalu rendah (negatif) pada skala log-odds.
        """
    )

st.markdown("#### Reliability Diagram")
st.caption(
    "Garis putus-putus abu = kalibrasi sempurna (prediksi = observasi). "
    "Ukuran titik merah proporsional terhadap jumlah sampel di bin tersebut."
)

size_ref = np.interp(rel["n"], [rel["n"].min(), rel["n"].max()], [10, 30])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#B7C3CB", dash="dash", width=1.5),
        name="Kalibrasi sempurna",
    )
)
fig.add_trace(
    go.Scatter(
        x=rel["mean_predicted"], y=rel["observed_freq"], mode="lines+markers",
        line=dict(color="#0E4749", width=2),
        marker=dict(size=size_ref, color="#E0483F", line=dict(color="#0E4749", width=1)),
        name="CardioSense AI (LightGBM)",
        customdata=rel["n"],
        hovertemplate="Rata-rata prediksi: %{x:.3f}<br>Frekuensi aktual: %{y:.3f}<br>n = %{customdata:,}<extra></extra>",
    )
)
fig.update_layout(
    height=440,
    xaxis=dict(title="Rata-rata probabilitas prediksi", range=[0, 1]),
    yaxis=dict(title="Frekuensi observasi aktual", range=[0, 1]),
    margin=dict(l=10, r=20, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Model sudah cukup terkalibrasi baik secara keseluruhan (ECE = {cal['ece']:.3f}, "
    f"slope = {cal['cal_slope']:.2f}). Bin dengan probabilitas paling ekstrem "
    "(mendekati 0 atau 1) menunjukkan selisih relatif terbesar terhadap kalibrasi sempurna — "
    "wajar terjadi karena jumlah sampel di bin ekstrem lebih sedikit."
)

with st.expander("Lihat tabel lengkap per bin"):
    display = rel[["bin_label", "n", "mean_predicted", "observed_freq", "gap"]].rename(
        columns={
            "bin_label": "Rentang Bin",
            "n": "Jumlah Sampel",
            "mean_predicted": "Rata-rata Prediksi",
            "observed_freq": "Frekuensi Aktual",
            "gap": "Selisih",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()
st.page_link("pages/04_Keadilan.py", label="Lihat audit Keadilan antar kelompok →", icon="⚖️")
st.caption(
    "Kalibrasi mengukur apakah probabilitas yang dikeluarkan model sudah mencerminkan "
    "frekuensi kejadian aktual secara keseluruhan. Untuk melihat apakah kualitas prediksi "
    "(termasuk kalibrasi) berbeda antar kelompok demografis, lihat halaman **Keadilan**."
)
