import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Card Customer Clustering", page_icon="💳", layout="wide")

MODEL_PATH = os.path.join("model", "kmeans_pipeline.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

artifact = load_model()
pipeline = artifact["pipeline"]
features = artifact["features"]
final_k = artifact["final_k"]

st.title("💳 Credit Card Customer Clustering")
st.write(
    "Aplikasi segmentasi pelanggan kartu kredit menggunakan K-Means "
    "berdasarkan dataset CC GENERAL dari Kaggle."
)

st.info(
    f"Model menggunakan K-Means dengan {final_k} cluster. "
    "Hasil cluster merupakan segmentasi statistik, bukan label bisnis yang bersifat mutlak."
)

st.subheader("Input Data Pelanggan")

# Default values are simple dataset medians, so the form is immediately usable.
default_row = {
    col: 0.0 for col in features
}
try:
    df_reference = pd.read_csv("CC_GENERAL.csv")
    medians = df_reference[features].median(numeric_only=True).to_dict()
    default_row.update(medians)
except Exception:
    pass

# Organize inputs in three columns.
cols = st.columns(3)
input_data = {}

for i, feature in enumerate(features):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=float(default_row.get(feature, 0.0)),
            format="%.4f"
        )

if st.button("Prediksi Cluster", type="primary"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = int(pipeline.predict(input_df)[0])

    st.success(f"Pelanggan diprediksi masuk ke **Cluster {prediction}**.")
    st.caption(
        "Interpretasikan cluster berdasarkan profil statistik pada notebook. "
        "Model tidak memberikan diagnosis atau keputusan kredit."
    )

st.divider()
st.subheader("Tentang Dataset")
st.markdown(
    "Sumber: [Kaggle — Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)"
)
st.write(
    "Fitur yang digunakan berasal dari data saldo, pembelian, cash advance, "
    "frekuensi transaksi, limit kredit, pembayaran, dan tenure."
)
