import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = Path("model/kmeans_pipeline.joblib")

st.title("💳 Credit Card Customer Segmentation")
st.caption("K-Means clustering berdasarkan pipeline yang digunakan pada notebook CRISP-DM.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model_data = load_model()

if model_data is None:
    st.error(
        "Model belum ditemukan. Pastikan file "
        "`model/kmeans_pipeline.joblib` sudah di-upload ke repository."
    )
    st.stop()

pipeline = model_data["pipeline"]
feature_cols = model_data["features"]
final_k = model_data["final_k"]

# Ambil komponen model hasil training.
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]
kmeans = pipeline.named_steps["kmeans"]

st.sidebar.header("Model Information")
st.sidebar.write(f"Jumlah cluster: **{final_k}**")
st.sidebar.write(f"Jumlah fitur: **{len(feature_cols)}**")
st.sidebar.write("Preprocessing: Median Imputation + StandardScaler")
st.sidebar.write("Model: K-Means")

st.subheader("Prediksi Cluster Pelanggan")
st.write(
    "Masukkan data numerik pelanggan. Aplikasi menggunakan preprocessing "
    "dan centroid K-Means dari model hasil training pada notebook."
)

default_values = {
    "BALANCE": 0.0,
    "BALANCE_FREQUENCY": 0.0,
    "PURCHASES": 0.0,
    "ONEOFF_PURCHASES": 0.0,
    "INSTALLMENTS_PURCHASES": 0.0,
    "CASH_ADVANCE": 0.0,
    "PURCHASES_FREQUENCY": 0.0,
    "ONEOFF_PURCHASES_FREQUENCY": 0.0,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.0,
    "CASH_ADVANCE_FREQUENCY": 0.0,
    "CASH_ADVANCE_TRX": 0.0,
    "PURCHASES_TRX": 0.0,
    "CREDIT_LIMIT": 0.0,
    "PAYMENTS": 0.0,
    "MINIMUM_PAYMENTS": 0.0,
    "PRC_FULL_PAYMENT": 0.0,
    "TENURE": 0.0,
}

with st.form("customer_form"):
    values = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=float(default_values.get(feature, 0.0)),
                format="%.4f",
            )

    submitted = st.form_submit_button(
        "🔍 Prediksi Cluster",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([values], columns=feature_cols)

    try:
        # Manual preprocessing:
        # Karena input form selalu numerik dan tidak kosong, kita tidak
        # perlu memanggil SimpleImputer.transform(). Ini juga menghindari
        # masalah kompatibilitas SimpleImputer pada model joblib lama.
        X = input_df.to_numpy(dtype=float)

        # Terapkan median hasil training jika ada NaN.
        statistics = np.asarray(imputer.statistics_, dtype=float)
        X = np.where(np.isnan(X), statistics, X)

        # Standardisasi menggunakan parameter scaler hasil training.
        X_scaled = scaler.transform(X)

        # K-Means prediction = centroid terdekat berdasarkan Euclidean distance.
        centers = np.asarray(kmeans.cluster_centers_, dtype=float)
        distances = ((X_scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        prediction = int(np.argmin(distances[0]))

        st.success(f"Pelanggan diprediksi masuk ke **Cluster {prediction}**.")

        st.subheader("Data yang Digunakan")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.divider()

st.subheader("Tentang Model")
st.write(
    "Model menggunakan K-Means Clustering untuk mengelompokkan pelanggan "
    "berdasarkan pola saldo, pembelian, cash advance, pembayaran, limit kredit, "
    "frekuensi transaksi, dan tenure."
)

st.info(
    "Cluster merupakan hasil unsupervised learning. Interpretasi Cluster 0 "
    "dan Cluster 1 mengikuti profil statistik yang dihasilkan pada notebook."
)
