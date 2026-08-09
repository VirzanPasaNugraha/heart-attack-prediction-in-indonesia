import streamlit as st

from utils.styling import dev_card, inject_base_css, section_header

st.set_page_config(page_title="Tentang Pengembang — CardioSense AI", page_icon="👨‍💻", layout="wide")
inject_base_css()

section_header("Kredit", "Tentang Pengembang")

dev_card(
    name="Virzan Pasa Nugraha",
    role="Pengembang · CardioSense AI",
    blurb=(
        "Merancang dan mengembangkan CardioSense AI sebagai prototipe sistem "
        "prediksi risiko serangan jantung berbasis machine learning — mencakup "
        "pipeline preprocessing, integrasi model LightGBM, modul interpretasi "
        "SHAP, hingga antarmuka aplikasinya."
    ),
)

st.markdown("")
section_header("Tumpukan Teknologi", "Dibangun Dengan")
cols = st.columns(5)
stack = [
    ("🐍", "Python"),
    ("🎈", "Streamlit"),
    ("🌳", "LightGBM"),
    ("🔍", "SHAP"),
    ("📊", "Plotly"),
]
for col, (icon, name) in zip(cols, stack):
    with col:
        st.markdown(
            f"""
            <div style="text-align:center; background:var(--cs-surface); border:1px solid var(--cs-border);
                        border-radius:12px; padding:1rem 0.5rem;">
                <div style="font-size:1.6rem;">{icon}</div>
                <div style="font-weight:600; font-size:0.85rem; margin-top:0.3rem;">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
