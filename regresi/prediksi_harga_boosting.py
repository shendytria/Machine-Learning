import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =============================
# 1. UNDUH DATA SAHAM ONLINE
# =============================
ticker = "BBCA.JK"  # Ganti jika ingin saham lain
data = yf.download(ticker, start="2023-01-01", end="2024-12-31")

print("=== Head Data ===")
print(data.head())

data.columns = data.columns.get_level_values(0)
data.reset_index(inplace=True)
data['DayIndex'] = np.arange(1, len(data) + 1)

# =============================
# 2. PILIH FITUR & TARGET
# =============================
X = data[['DayIndex']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 3. FUNGSI EVALUASI
# =============================
def evaluate(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2  : {r2:.4f}")
    return [model_name, mae, mse, rmse, mape, r2]

# =============================
# 4. TRAINING XGBOOST
# =============================
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
res_xgb = evaluate("XGBoost", y_test, y_pred_xgb)

# =============================
# 5. TRAINING LIGHTGBM
# =============================
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
res_lgbm = evaluate("LightGBM", y_test, y_pred_lgbm)

# =============================
# 6. GRAFIK HASIL PREDIKSI
# =============================
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual', color='black')
plt.plot(y_pred_xgb, label='XGBoost Pred', linestyle='--')
plt.plot(y_pred_lgbm, label='LightGBM Pred', linestyle=':')
plt.title(f"Perbandingan Prediksi Harga {ticker}")
plt.xlabel("Data Index (Test)")
plt.ylabel("Harga Penutupan (Close)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# 7. RINGKASAN METRIK
# =============================
hasil = pd.DataFrame([res_xgb, res_lgbm], columns=["Model", "MAE", "MSE", "RMSE", "MAPE", "R2"])
print("\n=== RINGKASAN HASIL ===")
print(hasil)
