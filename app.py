import streamlit as st

from utils.model_loader import load_report
from utils.styling import ekg_divider, hero_banner, inject_base_css, info_card, section_header

st.set_page_config(
    page_title="CardioSense AI — Prediksi Risiko Serangan Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_base_css()

with st.sidebar:
    st.markdown(
        """
        <div style="font-family:var(--cs-font-display); font-size:1.3rem; font-weight:700; color:#FFFFFF;">
            🫀 CardioSense AI
        </div>
        <div style="font-family:var(--cs-font-mono); font-size:0.72rem; color:#8FCFC7; letter-spacing:0.08em;
                    text-transform:uppercase; margin-bottom:1rem;">
            Sistem Prediksi Risiko
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Dikembangkan oleh Virzan Pasa Nugraha")

hero_banner(
    eyebrow="Prototipe Riset · Machine Learning untuk Kesehatan",
    title="Deteksi Dini Risiko Serangan Jantung dengan AI",
    subtitle=(
        "CardioSense AI menganalisis 27 faktor demografis, gaya hidup, dan klinis "
        "menggunakan model LightGBM untuk mengestimasi probabilitas risiko serangan "
        "jantung seseorang — lengkap dengan penjelasan faktor pendorongnya."
    ),
)

report = load_report()

st.markdown("")
col_a, col_b = st.columns([1.3, 1], gap="large")

with col_a:
    section_header("Kenapa CardioSense", "Prediksi yang Bisa Dijelaskan, Bukan Kotak Hitam")
    st.markdown(
        """
        Kebanyakan model prediksi risiko kesehatan berhenti di angka probabilitas
        saja. CardioSense AI melangkah lebih jauh: setiap prediksi dilengkapi
        **breakdown SHAP** yang menunjukkan faktor mana yang paling mendorong
        naik-turunnya risiko seseorang — sehingga hasilnya bisa ditelusuri, bukan
        sekadar dipercaya begitu saja.
        """
    )
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        info_card("🩺", "27 Faktor Klinis", "Dari tekanan darah, kolesterol, gula darah, hingga gaya hidup & lingkungan.", "primary")
    with ic2:
        info_card("🔍", "Bisa Dijelaskan", "Kontribusi tiap faktor terhadap prediksi ditelusuri lewat SHAP values asli.", "accent")
    with ic3:
        info_card("⚡", "Real-time", "Hasil prediksi & penjelasan langsung muncul begitu form dikirim.", "primary")

with col_b:
    st.markdown(
        f"""
        <div style="background:var(--cs-surface); border:1px solid var(--cs-border); border-radius:16px;
                    padding:1.5rem;">
            <div style="font-family:var(--cs-font-mono); font-size:0.72rem; letter-spacing:0.1em;
                        text-transform:uppercase; color:var(--cs-ink-soft); margin-bottom:0.6rem;">
                Ringkasan Model Terlatih
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="font-family:var(--cs-font-display); font-weight:700; font-size:1.3rem;
                    color:var(--cs-primary); margin-bottom:0.6rem;">{report['Best_Model']}</div>""",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m1.metric("ROC AUC", f"{report['Best_Model_ROC_AUC']:.3f}")
    m2.metric("Balanced Acc.", f"{report['Best_Model_Balanced_Accuracy']:.1%}")
    m3, m4 = st.columns(2)
    m3.metric("F1 Macro", f"{report['Best_Model_F1_Macro']:.3f}")
    m4.metric("MCC", f"{report['Best_Model_MCC']:.3f}")
    st.caption(f"Dilatih pada {report['Total_Samples']:,} sampel data pasien.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.page_link("pages/01_Prediksi.py", label="**Coba Prediksi Sekarang →**", icon="🩺")
    st.page_link("pages/03_SHAP_Explainability.py", label="Lihat Faktor Paling Berpengaruh →", icon="🔍")
    st.page_link("pages/04_Keadilan.py", label="Audit Keadilan Model →", icon="⚖️")
    st.page_link("pages/05_Kalibrasi.py", label="Audit Kalibrasi Model →", icon="📏")

st.markdown("")
st.markdown(ekg_divider(color="var(--cs-border)", height=20, opacity=1.0), unsafe_allow_html=True)

section_header("Alur Kerja", "Bagaimana CardioSense AI Bekerja")
w1, w2, w3, w4 = st.columns(4)
steps = [
    ("1", "Input Data", "Isi 27 data demografis, gaya hidup, dan klinis lewat form terpandu."),
    ("2", "Preprocessing", "Data dinormalisasi & di-encode otomatis sesuai skema training model."),
    ("3", "Prediksi", "Model LightGBM menghasilkan probabilitas risiko serangan jantung."),
    ("4", "Penjelasan", "SHAP menunjukkan faktor apa saja yang mendorong hasil tersebut."),
]
for col, (num, title, desc) in zip([w1, w2, w3, w4], steps):
    with col:
        st.markdown(
            f"""
            <div style="padding:1rem 0.2rem;">
                <div style="font-family:var(--cs-font-display); font-size:1.6rem; font-weight:700;
                            color:var(--cs-accent);">{num}</div>
                <div style="font-weight:600; margin:0.2rem 0 0.3rem 0;">{title}</div>
                <div style="color:var(--cs-ink-soft); font-size:0.88rem; line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.info(
    "⚠️ **Disclaimer:** CardioSense AI adalah prototipe riset untuk tujuan edukasi & "
    "portofolio, bukan alat diagnosis medis. Hasil prediksi tidak menggantikan "
    "konsultasi dengan tenaga medis profesional.",
    icon="⚠️",
)
