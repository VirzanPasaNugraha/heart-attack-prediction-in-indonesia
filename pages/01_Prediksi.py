import plotly.graph_objects as go
import streamlit as st

from utils.feature_meta import BINARY_META, CATEGORICAL_META, NUMERIC_META
from utils.model_loader import (
    build_input_dataframe,
    load_model,
    predict,
    risk_level,
)
from utils.styling import inject_base_css, risk_badge, section_header

st.set_page_config(page_title="Prediksi — CardioSense AI", page_icon="🩺", layout="wide")
inject_base_css()

section_header("Form Penilaian", "Prediksi Risiko Serangan Jantung")
st.caption("Isi seluruh data di bawah ini seakurat mungkin, lalu klik **Prediksi Risiko**.")

model = load_model()

with st.form("prediction_form"):
    st.markdown("#### 👤 Data Demografis")
    c1, c2, c3 = st.columns(3)
    with c1:
        m = NUMERIC_META["age"]
        age = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=55
        )
        gender = st.selectbox(
            CATEGORICAL_META["gender"]["label"],
            list(CATEGORICAL_META["gender"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["gender"]["options"][k],
        )
    with c2:
        region = st.selectbox(
            CATEGORICAL_META["region"]["label"],
            list(CATEGORICAL_META["region"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["region"]["options"][k],
        )
        income_level = st.selectbox(
            CATEGORICAL_META["income_level"]["label"],
            list(CATEGORICAL_META["income_level"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["income_level"]["options"][k],
            index=1,
        )
    with c3:
        air_pollution_exposure = st.selectbox(
            CATEGORICAL_META["air_pollution_exposure"]["label"],
            list(CATEGORICAL_META["air_pollution_exposure"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["air_pollution_exposure"]["options"][k],
            index=1,
        )

    st.divider()
    st.markdown("#### 🏃 Gaya Hidup")
    c4, c5, c6 = st.columns(3)
    with c4:
        smoking_status = st.selectbox(
            CATEGORICAL_META["smoking_status"]["label"],
            list(CATEGORICAL_META["smoking_status"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["smoking_status"]["options"][k],
        )
        alcohol_consumption = st.selectbox(
            CATEGORICAL_META["alcohol_consumption"]["label"],
            list(CATEGORICAL_META["alcohol_consumption"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["alcohol_consumption"]["options"][k],
        )
    with c5:
        physical_activity = st.selectbox(
            CATEGORICAL_META["physical_activity"]["label"],
            list(CATEGORICAL_META["physical_activity"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["physical_activity"]["options"][k],
            index=1,
        )
        dietary_habits = st.selectbox(
            CATEGORICAL_META["dietary_habits"]["label"],
            list(CATEGORICAL_META["dietary_habits"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["dietary_habits"]["options"][k],
        )
    with c6:
        stress_level = st.selectbox(
            CATEGORICAL_META["stress_level"]["label"],
            list(CATEGORICAL_META["stress_level"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["stress_level"]["options"][k],
            index=1,
        )
        m = NUMERIC_META["sleep_hours"]
        sleep_hours = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=float(m["plausible"][0]), max_value=float(m["plausible"][1]),
            value=6.5, step=0.5,
        )

    st.divider()
    st.markdown("#### 🩸 Data Klinis")
    c7, c8, c9 = st.columns(3)
    with c7:
        m = NUMERIC_META["blood_pressure_systolic"]
        blood_pressure_systolic = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=130
        )
        m = NUMERIC_META["blood_pressure_diastolic"]
        blood_pressure_diastolic = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=80
        )
        m = NUMERIC_META["fasting_blood_sugar"]
        fasting_blood_sugar = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=109
        )
    with c8:
        m = NUMERIC_META["cholesterol_level"]
        cholesterol_level = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=199
        )
        m = NUMERIC_META["cholesterol_hdl"]
        cholesterol_hdl = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=49
        )
        m = NUMERIC_META["cholesterol_ldl"]
        cholesterol_ldl = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=130
        )
    with c9:
        m = NUMERIC_META["triglycerides"]
        triglycerides = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=m["plausible"][0], max_value=m["plausible"][1], value=149
        )
        m = NUMERIC_META["waist_circumference"]
        waist_circumference = st.number_input(
            f"{m['label']} ({m['unit']})", min_value=float(m["plausible"][0]), max_value=float(m["plausible"][1]),
            value=93.0, step=0.5,
        )
        EKG_results = st.selectbox(
            CATEGORICAL_META["EKG_results"]["label"],
            list(CATEGORICAL_META["EKG_results"]["options"].keys()),
            format_func=lambda k: CATEGORICAL_META["EKG_results"]["options"][k],
        )

    st.divider()
    st.markdown("#### 📋 Riwayat Kesehatan")
    yn_keys = [0, 1]
    yn_labels = {0: "Tidak", 1: "Ya"}
    c10, c11, c12, c13 = st.columns(4)
    with c10:
        hypertension = st.radio(BINARY_META["hypertension"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
        diabetes = st.radio(BINARY_META["diabetes"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
    with c11:
        obesity = st.radio(BINARY_META["obesity"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
        family_history = st.radio(BINARY_META["family_history"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
    with c12:
        previous_heart_disease = st.radio(BINARY_META["previous_heart_disease"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
        medication_usage = st.radio(BINARY_META["medication_usage"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)
    with c13:
        participated_in_free_screening = st.radio(BINARY_META["participated_in_free_screening"]["label"], yn_keys, format_func=lambda k: yn_labels[k], horizontal=True)

    submitted = st.form_submit_button("🔍 Prediksi Risiko", use_container_width=True)

if submitted:
    form_values = {
        "age": age, "hypertension": hypertension, "diabetes": diabetes,
        "cholesterol_level": cholesterol_level, "obesity": obesity,
        "waist_circumference": waist_circumference, "family_history": family_history,
        "sleep_hours": sleep_hours, "blood_pressure_systolic": blood_pressure_systolic,
        "blood_pressure_diastolic": blood_pressure_diastolic, "fasting_blood_sugar": fasting_blood_sugar,
        "cholesterol_hdl": cholesterol_hdl, "cholesterol_ldl": cholesterol_ldl,
        "triglycerides": triglycerides, "previous_heart_disease": previous_heart_disease,
        "medication_usage": medication_usage,
        "participated_in_free_screening": participated_in_free_screening,
        "gender": gender, "region": region, "income_level": income_level,
        "smoking_status": smoking_status, "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity, "dietary_habits": dietary_habits,
        "air_pollution_exposure": air_pollution_exposure, "stress_level": stress_level,
        "EKG_results": EKG_results,
    }

    input_df = build_input_dataframe(form_values)
    label, prob = predict(model, input_df)
    level = risk_level(prob)

    # simpan buat halaman SHAP Explainability
    st.session_state["last_input_df"] = input_df
    st.session_state["last_prediction"] = {"label": label, "prob": prob, "level": level}

    st.divider()
    section_header("Hasil", "Estimasi Risiko")

    r1, r2 = st.columns([1, 1.4], gap="large")
    with r1:
        st.markdown(risk_badge(level), unsafe_allow_html=True)
        st.markdown(
            f"""<div style="font-family:var(--cs-font-mono); font-size:2.4rem; font-weight:700;
                        color:var(--cs-primary); margin-top:0.5rem;">{prob:.1%}</div>
            <div style="color:var(--cs-ink-soft); font-size:0.9rem;">probabilitas risiko serangan jantung</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.page_link("pages/03_SHAP_Explainability.py", label="Lihat faktor pendorong hasil ini →", icon="🔍")

    with r2:
        color = {"low": "#1F7A54", "medium": "#C98A1D", "high": "#C23B32"}[level]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 34, "family": "IBM Plex Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.35},
                    "steps": [
                        {"range": [0, 33], "color": "#E3F3EB"},
                        {"range": [33, 66], "color": "#FBF0DC"},
                        {"range": [66, 100], "color": "#FBE2E0"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob * 100},
                },
            )
        )
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ---- validasi klinis (informasional, tidak menghalangi hasil) ----
    warnings = []
    for key, val in [
        ("cholesterol_level", cholesterol_level), ("blood_pressure_systolic", blood_pressure_systolic),
        ("blood_pressure_diastolic", blood_pressure_diastolic), ("fasting_blood_sugar", fasting_blood_sugar),
        ("cholesterol_ldl", cholesterol_ldl), ("triglycerides", triglycerides),
    ]:
        meta = NUMERIC_META[key]
        if "warn_above" in meta and val > meta["warn_above"]:
            warnings.append(f"**{meta['label']}** = {val} {meta['unit']} — {meta['warn_msg']}")
    for key, val in [("cholesterol_hdl", cholesterol_hdl), ("sleep_hours", sleep_hours)]:
        meta = NUMERIC_META[key]
        if "warn_below" in meta and val < meta["warn_below"]:
            warnings.append(f"**{meta['label']}** = {val} {meta['unit']} — {meta['warn_msg']}")

    if warnings:
        with st.expander(f"⚠️ {len(warnings)} nilai input di luar rentang umum — klik untuk detail", expanded=False):
            for w in warnings:
                st.warning(w, icon="⚠️")

    st.caption(
        "Catatan: hasil ini adalah estimasi statistik dari model, bukan diagnosis medis. "
        "Untuk kepastian, tetap konsultasikan ke tenaga medis profesional."
    )

    with st.expander("Lihat data mentah yang dikirim ke model"):
        st.dataframe(input_df.T.rename(columns={0: "Nilai"}).astype(str), use_container_width=True)
