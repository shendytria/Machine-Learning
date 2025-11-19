import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Prediksi Harga Saham - XGBoost & LightGBM",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Prediksi Harga Saham Menggunakan XGBoost & LightGBM")
st.markdown(
    "Aplikasi ini digunakan untuk **memprediksi harga penutupan saham** "
    "menggunakan model regresi berbasis boosting (XGBoost & LightGBM)."
)

# ==========================
# SIDEBAR PENGATURAN
# ==========================
st.sidebar.header("⚙️ Pengaturan Input")

ticker = st.sidebar.text_input("Ticker Saham (Yahoo Finance)", "BBCA.JK")
start_date = st.sidebar.date_input("Tanggal Mulai", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Tanggal Selesai", value=pd.to_datetime("2024-12-31"))

test_size = st.sidebar.slider("Proporsi Data Test", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=9999, value=42)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Jalankan Training & Prediksi")

# ==========================
# FUNGSI EVALUASI
# ==========================
def evaluate(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

# ==========================
# PROSES UTAMA
# ==========================
if run_button:
    with st.spinner("Mengunduh data dan melatih model, mohon tunggu..."):
        # 1. Download data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("Data kosong. Coba ganti ticker atau rentang tanggal.")
        else:
            # Atasi multi-index column
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            st.subheader("📑 Data Harga Saham")
            st.write(f"Ticker: **{ticker}**")
            st.dataframe(data.head())

            # 2. Fitur & Target
            data = data.reset_index()
            data["DayIndex"] = np.arange(1, len(data) + 1)

            X = data[["DayIndex"]]     # fitur sederhana
            y = data["Close"]          # target harga penutupan

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # 3. Training Model
            # --- XGBoost ---
            xgb_model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            )
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            eval_xgb = evaluate("XGBoost", y_test, y_pred_xgb)

            # --- LightGBM ---
            lgbm_model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            )
            lgbm_model.fit(X_train, y_train)
            y_pred_lgbm = lgbm_model.predict(X_test)
            eval_lgbm = evaluate("LightGBM", y_test, y_pred_lgbm)

    # ==========================
    # TAMPILKAN METRIK
    # ==========================
    st.subheader("📊 Hasil Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### XGBoost")
        st.metric("R²", f"{eval_xgb['R2']:.4f}")
        st.metric("MAE", f"{eval_xgb['MAE']:.2f}")
        st.metric("RMSE", f"{eval_xgb['RMSE']:.2f}")
        st.metric("MAPE (%)", f"{eval_xgb['MAPE']:.2f}")

    with col2:
        st.markdown("### LightGBM")
        st.metric("R²", f"{eval_lgbm['R2']:.4f}")
        st.metric("MAE", f"{eval_lgbm['MAE']:.2f}")
        st.metric("RMSE", f"{eval_lgbm['RMSE']:.2f}")
        st.metric("MAPE (%)", f"{eval_lgbm['MAPE']:.2f}")

    # Tabel ringkasan
    hasil_df = pd.DataFrame([eval_xgb, eval_lgbm])
    st.markdown("#### Ringkasan Metrik")
    st.dataframe(hasil_df.set_index("Model"))

    # ==========================
    # GRAFIK PREDIKSI vs AKTUAL
    # ==========================
    st.subheader("📈 Perbandingan Harga Aktual vs Prediksi (Data Test)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values, label="Aktual")
    ax.plot(y_pred_xgb, linestyle="--", label="Prediksi XGBoost")
    ax.plot(y_pred_lgbm, linestyle=":", label="Prediksi LightGBM")
    ax.set_xlabel("Index Data (Test)")
    ax.set_ylabel("Harga Penutupan (Close)")
    ax.set_title(f"Perbandingan Prediksi Harga {ticker}")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # ==========================
    # PREDIKSI HARI BERIKUTNYA
    # ==========================
    st.subheader("🔮 Prediksi Harga untuk Hari Berikutnya (Setelah Data Terakhir)")

    next_index = np.array([[len(data) + 1]])
    next_xgb = xgb_model.predict(next_index)[0]
    next_lgbm = lgbm_model.predict(next_index)[0]

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Prediksi XGBoost", f"{next_xgb:,.2f}")
    with col4:
        st.metric("Prediksi LightGBM", f"{next_lgbm:,.2f}")

else:
    st.info("Silakan atur parameter di sidebar, lalu klik **'Jalankan Training & Prediksi'**.")
