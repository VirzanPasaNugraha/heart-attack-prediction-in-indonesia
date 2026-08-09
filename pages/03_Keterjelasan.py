import plotly.graph_objects as go
import streamlit as st

from utils.feature_meta import BINARY_META, CATEGORICAL_META, NUMERIC_META
from utils.model_loader import (
    get_shap_explainer,
    global_feature_importance,
    load_model,
    local_shap_explanation,
)
from utils.styling import inject_base_css, risk_badge, section_header

st.set_page_config(page_title="Interpretasi SHAP — CardioSense AI", page_icon="🔍", layout="wide")
inject_base_css()

section_header("Interpretasi Model", "Interpretasi Prediksi dengan SHAP")
st.caption(
    "Nilai SHAP dihitung langsung dari model LightGBM terlatih (bukan estimasi kasar) — "
    "menunjukkan seberapa besar & ke arah mana tiap fitur mendorong prediksi."
)

model = load_model()


def friendly_label(feat: str) -> str:
    if feat in NUMERIC_META:
        return NUMERIC_META[feat]["label"]
    if feat in BINARY_META:
        return BINARY_META[feat]["label"]
    if feat in CATEGORICAL_META:
        return CATEGORICAL_META[feat]["label"]
    return feat


tab1, tab2 = st.tabs(["🌐 Faktor Global (Seluruh Model)", "🎯 Penjelasan Prediksi Terakhir"])

with tab1:
    st.markdown("#### Fitur Paling Berpengaruh di Seluruh Model")
    st.caption(
        "Dihitung dari total *gain* tiap fitur di seluruh pohon LightGBM — "
        "diagregasi kembali dari fitur hasil encoding ke 27 fitur asli."
    )
    imp_df = global_feature_importance(model)
    imp_df["label"] = imp_df["feature"].apply(friendly_label)
    top_n = imp_df.head(15).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top_n["importance"],
            y=top_n["label"],
            orientation="h",
            marker=dict(color="#0E4749"),
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Total Gain (semakin besar = semakin berpengaruh)",
        font=dict(family="IBM Plex Sans", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "last_prediction" not in st.session_state:
        st.info(
            "Belum ada prediksi yang dijalankan di sesi ini. Buka halaman **Prediksi**, "
            "isi form, lalu kembali ke sini untuk melihat breakdown faktor pendorongnya.",
            icon="ℹ️",
        )
        st.page_link("pages/01_Prediksi.py", label="Buka halaman Prediksi →", icon="🩺")
    else:
        pred = st.session_state["last_prediction"]
        input_df = st.session_state["last_input_df"]

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(risk_badge(pred["level"]), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:var(--cs-font-mono); font-size:1.8rem; font-weight:700; "
                f"color:var(--cs-primary); margin-top:0.4rem;'>{pred['prob']:.1%}</div>",
                unsafe_allow_html=True,
            )

        explainer = get_shap_explainer(model)
        shap_df = local_shap_explanation(model, explainer, input_df)
        shap_df["label"] = shap_df["feature"].apply(friendly_label)
        shap_df["arah"] = shap_df["shap_value"].apply(lambda v: "Menaikkan risiko" if v > 0 else "Menurunkan risiko")

        plot_df = shap_df.head(12).iloc[::-1]
        colors = ["#C23B32" if v > 0 else "#1F7A54" for v in plot_df["shap_value"]]

        fig = go.Figure(
            go.Bar(
                x=plot_df["shap_value"],
                y=plot_df["label"],
                orientation="h",
                marker=dict(color=colors),
            )
        )
        fig.add_vline(x=0, line_width=1, line_color="#DCE3E8")
        fig.update_layout(
            height=440,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Kontribusi SHAP (merah = menaikkan risiko, hijau = menurunkan)",
            font=dict(family="IBM Plex Sans", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Nilai dasar (base value) model sebelum melihat data spesifik pasien ini: "
            f"**{shap_df['expected_value'].iloc[0]:.3f}** (skala log-odds). "
            "Setiap fitur mendorong nilai ini naik/turun hingga menghasilkan probabilitas akhir di atas."
        )

        with st.expander("Lihat tabel lengkap kontribusi tiap fitur"):
            display_df = shap_df[["label", "shap_value", "arah"]].rename(
                columns={"label": "Fitur", "shap_value": "Nilai SHAP", "arah": "Arah Pengaruh"}
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
