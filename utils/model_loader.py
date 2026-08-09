"""
Utility untuk load model & metadata prediksi risiko serangan jantung,
plus logika SHAP explainability.

PENTING: model ini disimpan sebagai imblearn.pipeline.Pipeline (ada step
SMOTE di dalamnya untuk training). Karena format serialisasinya dari
joblib, WAJIB di-load pakai `joblib.load()`, bukan `pickle.load()` biasa -
kalau pakai pickle.load() langsung akan error "STACK_GLOBAL requires str".
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "final_model.pkl"
REPORT_PATH = BASE_DIR / "data" / "final_summary_report.json"

# Urutan & tipe fitur PERSIS seperti saat training (27 fitur total)
NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "diabetes",
    "cholesterol_level",
    "obesity",
    "waist_circumference",
    "family_history",
    "sleep_hours",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "fasting_blood_sugar",
    "cholesterol_hdl",
    "cholesterol_ldl",
    "triglycerides",
    "previous_heart_disease",
    "medication_usage",
    "participated_in_free_screening",
]

CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "income_level",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
    "dietary_habits",
    "air_pollution_exposure",
    "stress_level",
    "EKG_results",
]

CONTINUOUS_FEATURES = [
    "age",
    "cholesterol_level",
    "waist_circumference",
    "sleep_hours",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "fasting_blood_sugar",
    "cholesterol_hdl",
    "cholesterol_ldl",
    "triglycerides",
]

BINARY_FLAGS = [
    "hypertension",
    "diabetes",
    "obesity",
    "family_history",
    "previous_heart_disease",
    "medication_usage",
    "participated_in_free_screening",
]

CLASS_LABELS = {0: "No_Heart_Attack", 1: "Heart_Attack"}


@st.cache_resource
def load_model():
    """Load pipeline (preprocessing + SMOTE + LightGBM) dengan joblib."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report() -> dict:
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


def build_input_dataframe(form_values: dict) -> pd.DataFrame:
    """Susun satu baris input user jadi DataFrame sesuai skema training."""
    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    row = {col: form_values[col] for col in all_cols}
    return pd.DataFrame([row])


def predict(model, input_df: pd.DataFrame):
    """Kembalikan (label prediksi, probabilitas kelas Heart_Attack)."""
    pred_class = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]
    prob_heart_attack = float(proba[1])  # classes_ = [0, 1]
    return CLASS_LABELS[pred_class], prob_heart_attack


def risk_level(prob: float) -> str:
    """Kategorikan probabilitas jadi pita risiko low/medium/high."""
    if prob < 0.33:
        return "low"
    if prob < 0.66:
        return "medium"
    return "high"


def _base_feature_name(transformed_name: str) -> str:
    """Petakan nama fitur hasil ColumnTransformer balik ke fitur asli
    (menggabungkan kolom one-hot per kategori jadi satu fitur induk)."""
    name = transformed_name.replace("num__", "").replace("cat__", "")
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


@st.cache_resource
def get_shap_explainer(_model):
    """Buat TreeExplainer sekali saja (cache) dari LGBMClassifier di dalam pipeline."""
    clf = _model.named_steps["clf"]
    return shap.TreeExplainer(clf)


@st.cache_data
def global_feature_importance(_model) -> pd.DataFrame:
    """Feature importance (gain-based) dari booster LightGBM yang sudah
    dilatih, diagregasi kembali ke 27 fitur asli (bukan per-kategori one-hot)."""
    pre = _model.named_steps["pre"]
    clf = _model.named_steps["clf"]
    feature_names = pre.get_feature_names_out()
    importances = clf.feature_importances_

    agg: dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        base = _base_feature_name(name)
        agg[base] = agg.get(base, 0.0) + float(imp)

    df = pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def local_shap_explanation(model, explainer, input_df: pd.DataFrame) -> pd.DataFrame:
    """Hitung kontribusi SHAP per fitur ASLI (agregasi one-hot) untuk satu
    baris input. Mengembalikan DataFrame terurut berdasar |kontribusi|."""
    pre = model.named_steps["pre"]
    feature_names = pre.get_feature_names_out()
    X_trans = pre.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    shap_values = explainer.shap_values(X_trans)[0]

    agg: dict[str, float] = {}
    for name, val in zip(feature_names, shap_values):
        base = _base_feature_name(name)
        agg[base] = agg.get(base, 0.0) + float(val)

    df = pd.DataFrame({"feature": list(agg.keys()), "shap_value": list(agg.values())})
    df["abs_value"] = df["shap_value"].abs()
    df = df.sort_values("abs_value", ascending=False).drop(columns="abs_value").reset_index(drop=True)
    df["expected_value"] = float(explainer.expected_value)
    return df
