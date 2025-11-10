# app.py (robust, defensive, no assumptions about exact feature subset)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Dashboard Literasi & Numerasi (Robust)", layout="wide")

# --------------------------
# Dark-mode styling
# --------------------------
st.markdown(
    """
    <style>
      body {background-color: #0e1117; color: #fafafa;}
      .stApp { background-color: #0e1117; }
      h1,h2,h3,h4 { color: #00c4ff; }
      .block-container {padding-top:1rem;}
      .stButton>button { background-color:#238636; color:white; border-radius:6px; }
      .stButton>button:hover { background-color:#2ea043; }
      .stDataFrame div { color: #fafafa; }
    </style>
    """, unsafe_allow_html=True
)

st.title("📊 SDG 4 Pendidikan Berkualitas - Kelompok 12")
st.markdown("")

# --------------------------
# Helpers
# --------------------------
def try_load(path):
    """Try to load model/scaler with joblib then pickle; return None on failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def infer_feature_names(obj, df_columns):
    """
    Return list of expected feature names from a scaler/model if possible.
    Fallback: use df_columns (excluding targets if present).
    """
    # scaler with attribute feature_names_in_
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    # model with attribute feature_names_in_
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    # if it's a pipeline, try to inspect named_steps
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # fallback: use dataframe columns passed by caller
    return list(df_columns)

def safe_predict(model, scaler, X_df, fallback_fill):
    """
    Align X_df to expected features, fill missing with fallback_fill (dict or series),
    scale with scaler if not None, predict with model.
    """
    expected = infer_feature_names(scaler if scaler is not None else model, X_df.columns)
    # ensure expected is list
    if expected is None:
        expected = list(X_df.columns)

    # add missing cols
    for c in expected:
        if c not in X_df.columns:
            # fill with fallback (mean) if available, else 0
            X_df[c] = fallback_fill.get(c, 0) if fallback_fill is not None else 0

    # reduce to expected order
    X_aligned = X_df.reindex(columns=expected, fill_value=0)

    # convert to numeric (coerce)
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    # scale if scaler provided
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_aligned)
        except Exception as e:
            # try to transform without checking feature names by converting to numpy
            X_scaled = scaler.transform(X_aligned.values)
    else:
        X_scaled = X_aligned.values

    # predict
    y_pred = model.predict(X_scaled)
    return y_pred, expected

# --------------------------
# Sidebar: Data & Model loaders
# --------------------------
st.sidebar.header("1) Data & Model")
st.sidebar.markdown("Jika file tidak ada di folder, upload di sini.")

# Default paths (update if your filenames differ)
DATA_PATH = "data_final.csv"      # change if different
MODEL_LIT_PATH = "model_lit.pkl"
MODEL_NUM_PATH = "model_num.pkl"
SCALER_LIT_PATH = "scaler_lit.pkl"
SCALER_NUM_PATH = "scaler_num.pkl"

# upload options
uploaded_csv = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
uploaded_model_lit = st.sidebar.file_uploader("Upload model_lit (.pkl/.joblib) (optional)", type=["pkl", "joblib"])
uploaded_model_num = st.sidebar.file_uploader("Upload model_num (.pkl/.joblib) (optional)", type=["pkl", "joblib"])
uploaded_scaler_lit = st.sidebar.file_uploader("Upload scaler_lit (.pkl/.joblib) (optional)", type=["pkl", "joblib"])
uploaded_scaler_num = st.sidebar.file_uploader("Upload scaler_num (.pkl/.joblib) (optional)", type=["pkl", "joblib"])

# load dataset
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("Dataset loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            st.sidebar.success(f"Dataset loaded from `{DATA_PATH}`")
        except Exception as e:
            st.sidebar.error(f"Failed to load {DATA_PATH}: {e}")
            st.stop()
    else:
        st.sidebar.warning("No dataset found: upload CSV or place data_final.csv in folder.")
        st.stop()

# load models/scalers (uploaded takes precedence)
def load_obj_from_upload_or_path(uploaded, path):
    if uploaded is not None:
        try:
            # uploaded is BytesIO
            return try_load(uploaded)
        except Exception:
            try:
                # try to read bytes then pickle
                return pickle.load(uploaded)
            except Exception:
                return None
    else:
        return try_load(path)

model_lit = load_obj_from_upload_or_path(uploaded_model_lit, MODEL_LIT_PATH)
model_num = load_obj_from_upload_or_path(uploaded_model_num, MODEL_NUM_PATH)
scaler_lit = load_obj_from_upload_or_path(uploaded_scaler_lit, SCALER_LIT_PATH)
scaler_num = load_obj_from_upload_or_path(uploaded_scaler_num, SCALER_NUM_PATH)

if model_lit is None:
    st.sidebar.error("model_lit not found or failed to load. Upload a model_lit file.")
if model_num is None:
    st.sidebar.error("model_num not found or failed to load. Upload a model_num file.")
if scaler_lit is None:
    st.sidebar.warning("scaler_lit not found or failed to load. Scaling step for LIT will be skipped.")
if scaler_num is None:
    st.sidebar.warning("scaler_num not found or failed to load. Scaling step for NUM will be skipped.")

# --------------------------
# Basic data checks & preprocess hints
# --------------------------
st.sidebar.markdown("---")
st.sidebar.header("2) Quick data checks")
st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
if "LIT" not in df.columns or "NUM" not in df.columns:
    st.sidebar.warning("Dataset should include 'LIT' and 'NUM' columns for full functionality.")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
st.sidebar.write(f"Numeric cols: {len(numeric_cols)}")

# compute fallback fill-values (means) for missing features
fallback_fill = df.mean(numeric_only=True).to_dict()

# --------------------------
# Main layout: Tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Korelasi & Distribusi", "Prediksi Interaktif", "Visualisasi"])

# ---------- Overview ----------
with tab1:
    st.header("Overview Data")
    st.dataframe(df.head(200))
    st.markdown("**Summary (numeric)**")
    st.table(df.describe().T)

# ---------- Korelasi & Distribusi ----------
with tab2:
    st.header("Distribusi & Korelasi")
    col1, col2 = st.columns(2)
    with col1:
        if "LIT" in df.columns:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df["LIT"].dropna(), kde=True, ax=ax, color="#00c4ff")
            ax.axvline(df["LIT"].mean(), color="red", linestyle="--", label=f"Mean {df['LIT'].mean():.2f}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Column LIT not found.")

    with col2:
        if "NUM" in df.columns:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df["NUM"].dropna(), kde=True, ax=ax, color="#ff6b6b")
            ax.axvline(df["NUM"].mean(), color="red", linestyle="--", label=f"Mean {df['NUM'].mean():.2f}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Column NUM not found.")

    st.markdown("---")
    st.subheader("Heatmap korelasi (numeric)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------- Prediksi Interaktif ----------
# ---------- Prediksi Interaktif ----------
with tab3:
    st.header("Prediksi Interaktif (Model: LIT saja)")
    st.markdown(
        "Isi nilai variabel yang ingin digunakan untuk memprediksi **LIT (Literasi)**. "
        "Model NUM tidak digunakan di versi ini."
    )

    # fitur tetap
    num_features = [
        "proporsi_pendidik_min_s1", "proporsi_pendidik_sertifikasi",
        "jumlah_peserta_didik", "jumlah_pendidik", "rasio_pendidik_peserta_didik",
        "jumlah_r_kelas", "jumlah_komp_milik", "jumlah_perpus",
        "SES_siswa", "SES_sekolah",
        "BUL", "SAF", "WEL", "TAS", "ENP", "OCC", "PBR"
    ]

    st.write("Fitur yang digunakan untuk prediksi:")
    st.write(num_features)

    # form input
    with st.form("predict_lit_form"):
        user_vals = {}
        cols_per_row = 3
        rows = (len(num_features) + cols_per_row - 1) // cols_per_row
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = r*cols_per_row + i
                if idx >= len(num_features):
                    break
                feat = num_features[idx]
                if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                    lo = float(df[feat].min())
                    hi = float(df[feat].max())
                    default = float(df[feat].median())
                else:
                    lo, hi, default = 0.0, 100.0, float(fallback_fill.get(feat, 0.0))
                user_vals[feat] = cols[i].number_input(
                    label=feat, min_value=lo, max_value=hi, value=default, format="%.4f"
                )

        submitted = st.form_submit_button("Predict LIT")

    if submitted:
        X_input = pd.DataFrame([user_vals])

        if model_lit is not None:
            try:
                y_lit_pred_arr, used_feats_lit = safe_predict(model_lit, scaler_lit, X_input.copy(), fallback_fill)
                y_lit_pred = float(y_lit_pred_arr[0])
                st.success(f"Prediksi LIT: {y_lit_pred:.2f}")

            except Exception as e:
                st.error(f"Error saat prediksi LIT: {e}")
        else:
            st.warning("Model LIT belum dimuat. Pastikan file model_lit.pkl tersedia.")


# ---------- Model & Files ----------
with tab4:
    st.header("Analisis Tren Faktor Sosial Ekonomi terhadap Capaian Asesmen")

    # 1) Pengaruh faktor terhadap LIT & NUM (Trend-based)
    st.subheader("1. Tren pengaruh faktor terhadap LIT & NUM")
    factor_cols = [c for c in df.columns if c not in ("LIT", "NUM")]
    selected_factor = st.selectbox("Pilih faktor numerik", factor_cols)

    if pd.api.types.is_numeric_dtype(df[selected_factor]):
        fig, ax = plt.subplots(figsize=(6,4))
        sns.regplot(data=df, x=selected_factor, y="LIT", lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'#007BFF'}, label="LIT", ax=ax)
        sns.regplot(data=df, x=selected_factor, y="NUM", lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'#FF5A5F'}, label="NUM", ax=ax)
        ax.set_title(f"Tren hubungan {selected_factor} terhadap LIT & NUM")
        ax.set_xlabel(selected_factor)
        ax.set_ylabel("Nilai Asesmen")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # 2) Kesenjangan hasil belajar berdasarkan SES_siswa (Trend dibanding kategori)
    st.subheader("2. Tren capaian berdasarkan SES_siswa")
    if "SES_siswa" in df.columns:
        ses_mean = df.groupby("SES_siswa")[["LIT", "NUM"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(data=ses_mean, x="SES_siswa", y="LIT", marker="o", color="#007BFF", label="LIT")
        sns.lineplot(data=ses_mean, x="SES_siswa", y="NUM", marker="o", color="#FF5A5F", label="NUM")
        ax.set_title("Tren rata-rata LIT & NUM berdasarkan SES_siswa")
        ax.set_xlabel("Tingkat SES_siswa")
        ax.set_ylabel("Rata-rata Nilai")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # 3) Pengaruh lingkungan sekolah terhadap hasil belajar
    st.subheader("3. Tren pengaruh lingkungan sekolah")
    school_factors = [c for c in df.columns if c.startswith("SES_sekolah") or c.startswith("rasio") or c.startswith("proporsi")]
    if school_factors:
        selected_school_factor = st.selectbox("Pilih faktor lingkungan sekolah", school_factors)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.regplot(data=df, x=selected_school_factor, y="LIT",scatter_kws={'alpha':0.3}, line_kws={'color':'#007BFF'}, label="LIT", ax=ax)
        sns.regplot(data=df, x=selected_school_factor, y="NUM", lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'#FF5A5F'}, label="NUM", ax=ax)
        ax.set_title(f"Tren pengaruh {selected_school_factor} terhadap LIT & NUM")
        ax.set_xlabel(selected_school_factor)
        ax.set_ylabel("Nilai Asesmen")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    
   


st.sidebar.markdown("---")
st.sidebar.info("If predictions are wrong: ensure the models/scalers were trained on the same features and same preprocessing. If you trained the model elsewhere, re-save scaler with attribute feature_names_in_ (scikit-learn scalers set this when fit on a pandas DataFrame).")
