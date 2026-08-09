"""
Kamus metadata fitur: label Bahasa Indonesia, deskripsi, satuan, dan
rentang wajar klinis (dipakai untuk warning validasi input - BUKAN untuk
diagnosis, hanya penanda "nilai ini di luar rentang umum, mohon dicek lagi").

Rentang wajar merujuk ke rentang klinis umum yang dipakai di literatur
kesehatan populer (mis. kategori tekanan darah AHA), sekadar sebagai
guardrail input, bukan rekomendasi medis.
"""

NUMERIC_META = {
    "age": {
        "label": "Usia",
        "unit": "tahun",
        "desc": "Usia pasien saat ini.",
        "plausible": (18, 100),
    },
    "cholesterol_level": {
        "label": "Kolesterol Total",
        "unit": "mg/dL",
        "desc": "Kadar kolesterol total dalam darah.",
        "plausible": (100, 320),
        "warn_above": 240,
        "warn_msg": "Di atas 240 mg/dL tergolong tinggi.",
    },
    "waist_circumference": {
        "label": "Lingkar Pinggang",
        "unit": "cm",
        "desc": "Lingkar pinggang, indikator obesitas sentral.",
        "plausible": (50, 160),
    },
    "sleep_hours": {
        "label": "Rata-rata Jam Tidur",
        "unit": "jam/hari",
        "desc": "Rata-rata durasi tidur per hari.",
        "plausible": (0, 14),
        "warn_below": 4,
        "warn_msg": "Kurang dari 4 jam tergolong sangat kurang tidur.",
    },
    "blood_pressure_systolic": {
        "label": "Tekanan Darah Sistolik",
        "unit": "mmHg",
        "desc": "Tekanan darah saat jantung berkontraksi (angka atas).",
        "plausible": (70, 220),
        "warn_above": 180,
        "warn_msg": "Di atas 180 mmHg tergolong krisis hipertensi.",
    },
    "blood_pressure_diastolic": {
        "label": "Tekanan Darah Diastolik",
        "unit": "mmHg",
        "desc": "Tekanan darah saat jantung relaksasi (angka bawah).",
        "plausible": (40, 140),
        "warn_above": 120,
        "warn_msg": "Di atas 120 mmHg tergolong krisis hipertensi.",
    },
    "fasting_blood_sugar": {
        "label": "Gula Darah Puasa",
        "unit": "mg/dL",
        "desc": "Kadar gula darah setelah puasa 8 jam.",
        "plausible": (50, 400),
        "warn_above": 126,
        "warn_msg": "Di atas 126 mg/dL mengindikasikan diabetes.",
    },
    "cholesterol_hdl": {
        "label": "Kolesterol HDL",
        "unit": "mg/dL",
        "desc": "Kolesterol 'baik'.",
        "plausible": (15, 110),
        "warn_below": 40,
        "warn_msg": "Di bawah 40 mg/dL tergolong rendah (kurang protektif).",
    },
    "cholesterol_ldl": {
        "label": "Kolesterol LDL",
        "unit": "mg/dL",
        "desc": "Kolesterol 'jahat'.",
        "plausible": (30, 260),
        "warn_above": 160,
        "warn_msg": "Di atas 160 mg/dL tergolong tinggi.",
    },
    "triglycerides": {
        "label": "Trigliserida",
        "unit": "mg/dL",
        "desc": "Kadar lemak trigliserida dalam darah.",
        "plausible": (30, 600),
        "warn_above": 200,
        "warn_msg": "Di atas 200 mg/dL tergolong tinggi.",
    },
}

BINARY_META = {
    "hypertension": {"label": "Hipertensi", "desc": "Diagnosis hipertensi (tekanan darah tinggi)."},
    "diabetes": {"label": "Diabetes", "desc": "Diagnosis diabetes."},
    "obesity": {"label": "Obesitas", "desc": "Diagnosis/indikasi obesitas."},
    "family_history": {"label": "Riwayat Keluarga Jantung", "desc": "Ada keluarga inti dengan riwayat penyakit jantung."},
    "previous_heart_disease": {"label": "Riwayat Penyakit Jantung", "desc": "Pernah didiagnosis penyakit jantung sebelumnya."},
    "medication_usage": {"label": "Konsumsi Obat Rutin", "desc": "Sedang mengonsumsi obat resep secara rutin."},
    "participated_in_free_screening": {"label": "Ikut Skrining Gratis", "desc": "Pernah mengikuti program skrining kesehatan gratis."},
}

CATEGORICAL_META = {
    "gender": {
        "label": "Jenis Kelamin",
        "options": {"Female": "Perempuan", "Male": "Laki-laki"},
    },
    "region": {
        "label": "Wilayah Tempat Tinggal",
        "options": {"Rural": "Pedesaan", "Urban": "Perkotaan"},
    },
    "income_level": {
        "label": "Tingkat Pendapatan",
        "options": {"Low": "Rendah", "Middle": "Menengah", "High": "Tinggi"},
    },
    "smoking_status": {
        "label": "Status Merokok",
        "options": {"Never": "Tidak Pernah", "Past": "Mantan Perokok", "Current": "Perokok Aktif"},
    },
    "alcohol_consumption": {
        "label": "Konsumsi Alkohol",
        "options": {"None": "Tidak Pernah", "Moderate": "Sedang", "High": "Tinggi"},
    },
    "physical_activity": {
        "label": "Tingkat Aktivitas Fisik",
        "options": {"Low": "Rendah", "Moderate": "Sedang", "High": "Tinggi"},
    },
    "dietary_habits": {
        "label": "Pola Makan",
        "options": {"Healthy": "Sehat", "Unhealthy": "Tidak Sehat"},
    },
    "air_pollution_exposure": {
        "label": "Paparan Polusi Udara",
        "options": {"Low": "Rendah", "Moderate": "Sedang", "High": "Tinggi"},
    },
    "stress_level": {
        "label": "Tingkat Stres",
        "options": {"Low": "Rendah", "Moderate": "Sedang", "High": "Tinggi"},
    },
    "EKG_results": {
        "label": "Hasil EKG",
        "options": {"Normal": "Normal", "Abnormal": "Tidak Normal"},
    },
}

METRIC_INFO = {
    "Balanced Accuracy": "Rata-rata akurasi di tiap kelas — cocok untuk data yang jumlah kelasnya tidak seimbang.",
    "Accuracy": "Persentase total prediksi yang benar dari seluruh data uji.",
    "F1 Macro": "Rata-rata harmonik precision & recall di tiap kelas, tidak berbobot ukuran kelas.",
    "MCC": "Matthews Correlation Coefficient — korelasi antara prediksi & label asli (1 = sempurna, 0 = acak).",
    "ROC AUC": "Kemampuan model membedakan kelas positif vs negatif di semua ambang batas (1 = sempurna).",
    "Average Precision": "Ringkasan kurva precision-recall — penting saat kelas positif jarang muncul.",
}
