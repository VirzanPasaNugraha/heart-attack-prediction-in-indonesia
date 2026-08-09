import pandas as pd
import streamlit as st

from utils.feature_meta import BINARY_META, CATEGORICAL_META, NUMERIC_META
from utils.model_loader import load_report
from utils.styling import inject_base_css, section_header

st.set_page_config(page_title="Tentang Riset — CardioSense AI", page_icon="📚", layout="wide")
inject_base_css()

section_header("Metodologi", "Tentang Model & Riset")
report = load_report()

st.markdown(
    f"""
CardioSense AI dibangun di atas model **{report['Best_Model']}** yang dibungkus dalam
sebuah `scikit-learn` / `imblearn` `Pipeline` dengan tiga tahap:

1. **Preprocessing** — `ColumnTransformer` yang meng-impute nilai numerik kosong
   dengan median, menstandarisasi skala numerik (`StandardScaler`), dan
   meng-encode fitur kategorikal dengan `OneHotEncoder`.
2. **SMOTE** — teknik oversampling kelas minoritas, **hanya aktif saat training**
   untuk menyeimbangkan proporsi kelas `Heart_Attack` vs `No_Heart_Attack`.
3. **LightGBM Classifier** — model gradient boosting berbasis pohon keputusan
   yang menghasilkan probabilitas akhir.
"""
)

st.divider()
section_header("Sumber Data", "Dataset: Heart Attack Prediction in Indonesia")

st.markdown(
    """
Model CardioSense AI dilatih menggunakan dataset publik **"Heart Attack Prediction in
Indonesia"** dari Kaggle (disusun oleh Ankush Panday). Dataset ini menyediakan profil
kesehatan individu di Indonesia secara rinci, dengan fokus pada prediksi risiko serangan
jantung — mencakup faktor demografis, klinis, gaya hidup, dan lingkungan utama yang
berkaitan dengan risiko kardiovaskular, sekaligus merefleksikan tren kesehatan nyata di
Indonesia seperti hipertensi, diabetes, obesitas, kebiasaan merokok, dan paparan polusi.

Mengingat tren peningkatan penyakit kardiovaskular di Indonesia, deteksi dini dan
pencegahan menjadi krusial. Dataset ini disusun untuk mendukung pengembangan model
*machine learning* dalam memprediksi risiko serangan jantung, riset kesehatan masyarakat,
dan studi epidemiologi.
"""
)
st.link_button(
    "🔗 Buka Dataset di Kaggle",
    "https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia",
)

st.markdown("#### Kategori Variabel dalam Dataset")
st.markdown(
    """
1. **Demografi** — usia, jenis kelamin, wilayah tempat tinggal, tingkat pendapatan.
2. **Faktor Risiko Klinis** — hipertensi, diabetes, kolesterol total, obesitas, lingkar
   pinggang, riwayat keluarga jantung.
3. **Gaya Hidup & Perilaku** — status merokok, konsumsi alkohol, aktivitas fisik, pola makan.
4. **Faktor Lingkungan & Sosial** — paparan polusi udara, tingkat stres, rata-rata jam tidur.
5. **Skrining Medis & Sistem Kesehatan** — tekanan darah sistolik/diastolik, gula darah
   puasa, kolesterol HDL/LDL, trigliserida, hasil EKG, riwayat penyakit jantung
   sebelumnya, konsumsi obat rutin, keikutsertaan skrining kesehatan gratis.
6. **Variabel Target** — kejadian serangan jantung (ya/tidak).

Definisi lengkap setiap variabel — satuan, deskripsi, dan rentang nilai wajar yang dipakai
di form Prediksi — dapat dilihat di bagian **Skema Data** tepat di bawah ini, supaya tidak
ada duplikasi informasi.
"""
)

st.divider()
section_header("Skema Data", "Kamus Fitur Lengkap (27 Fitur)")

st.markdown("#### Fitur Numerik & Klinis")
rows = []
for feat, meta in NUMERIC_META.items():
    rng = meta["plausible"]
    rows.append({"Fitur": meta["label"], "Satuan": meta["unit"], "Deskripsi": meta["desc"], "Rentang Wajar": f"{rng[0]}–{rng[1]}"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("#### Indikator Biner (Ya/Tidak)")
rows = [{"Fitur": meta["label"], "Deskripsi": meta["desc"]} for meta in BINARY_META.values()]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("#### Fitur Kategorikal")
rows = []
for feat, meta in CATEGORICAL_META.items():
    opts = ", ".join(meta["options"].values())
    rows.append({"Fitur": meta["label"], "Pilihan Kategori": opts})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
section_header("Batasan", "Catatan & Keterbatasan")
st.markdown(
    """
- Model dilatih pada data historis dan **tidak memperhitungkan** faktor yang tidak
  ada dalam 27 fitur di atas (mis. hasil lab spesifik lain, riwayat genetik detail).
- Ambang batas "rentang wajar" pada tiap fitur bersifat **guardrail input**, bukan
  acuan diagnosis — merujuk kategori klinis umum yang banyak dipakai di literatur
  populer (mis. kategori tekanan darah AHA).
- Kurva evaluasi berbasis data uji mentah (ROC curve, confusion matrix per-kelas)
  tidak tersedia karena berkas yang diunggah hanya berisi model terlatih dan
  ringkasan metrik akhir, bukan prediksi baris-per-baris pada data uji.
- **Ini adalah prototipe riset**, bukan perangkat medis bersertifikasi.
"""
)
