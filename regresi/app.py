from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

app = Flask(__name__)

# === Fungsi Evaluasi ===
def evaluate(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"]
        start = request.form["start"]
        end = request.form["end"]

        # === Unduh data saham ===
        data = yf.download(ticker, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        data["DayIndex"] = np.arange(1, len(data) + 1)

        # === Split data ===
        X = data[["DayIndex"]]
        y = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === Training XGBoost ===
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        eval_xgb = evaluate("XGBoost", y_test, y_pred_xgb)

        # === Training LightGBM ===
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test)
        eval_lgbm = evaluate("LightGBM", y_test, y_pred_lgbm)

        # === Grafik perbandingan ===
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.values, label="Aktual", color="black")
        ax.plot(y_pred_xgb, linestyle="--", label="Prediksi XGBoost", color="blue")
        ax.plot(y_pred_lgbm, linestyle=":", label="Prediksi LightGBM", color="orange")
        ax.set_xlabel("Index Data (Test)")
        ax.set_ylabel("Harga Penutupan (Close)")
        ax.set_title(f"Perbandingan Harga Aktual vs Prediksi ({ticker})")
        ax.legend()
        ax.grid(True)

        # Simpan grafik ke folder static
        plot_path = os.path.join("static", "plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

        # === Prediksi hari berikutnya ===
        next_index = np.array([[len(data) + 1]])
        next_xgb = float(xgb.predict(next_index)[0])
        next_lgbm = float(lgbm.predict(next_index)[0])

        # === Kirim ke HTML ===
        return render_template(
            "result.html",
            ticker=ticker,
            start=start,
            end=end,
            eval_xgb=eval_xgb,
            eval_lgbm=eval_lgbm,
            next_xgb=next_xgb,
            next_lgbm=next_lgbm,
            plot_path=plot_path
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
