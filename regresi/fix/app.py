from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

app = Flask(__name__, static_folder="static", template_folder="templates")

uploaded_df = None
user_input = None

# ---------- Evaluation helper ----------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true.replace(0, 1e-8)))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ---------- Risk encoding ----------
def encode_risk_column(series):
    map_pre = {"low": 0, "medium": 1, "high": 2}
    try:
        encoded = series.str.lower().map(map_pre)
        if encoded.isna().any():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(series.fillna("unknown").astype(str))
        return encoded
    except Exception:
        return pd.Series(0, index=series.index)

# --------------------------------------------------------------------
# ---------------------------- INDEX ---------------------------------
# --------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_df, user_input

    if request.method == "POST":
        stock = request.form.get("stock", "").strip()
        date_str = request.form.get("date", "").strip()
        open_price = request.form.get("open", "").strip()
        risk = request.form.get("risk", "").strip()

        try:
            open_price = float(open_price)
        except:
            return "Invalid Open price."

        csv_path = os.path.join(os.path.dirname(__file__), "Stock_Price_Dataset.csv")
        df = pd.read_csv(csv_path)
        uploaded_df = df.copy()

        user_input = {
            "stock": stock,
            "date": date_str,
            "open": open_price,
            "risk": risk
        }

        return redirect(url_for("loading"))

    return render_template("index.html")


@app.route("/loading")
def loading():
    return render_template("loading.html")

# --------------------------------------------------------------------
# ---------------------------- PROCESS --------------------------------
# --------------------------------------------------------------------
@app.route("/process")
def process():
    global uploaded_df, user_input

    if uploaded_df is None or user_input is None:
        return redirect(url_for("index"))

    df = uploaded_df.copy()

    if "Price" not in df.columns:
        return "Dataset harus memiliki kolom Price."

    df = df.reset_index(drop=True)
    df["DayIndex"] = np.arange(1, len(df) + 1)

    if "Risk" in df.columns:
        df["Risk_val"] = encode_risk_column(df["Risk"].astype(str))
    else:
        df["Risk_val"] = 1

    if "Open" not in df.columns:
        df["Open"] = df["Price"]

    X = df[["DayIndex", "Open", "Risk_val"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------- Train XGBoost -------
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8
    )
    xgb.fit(X_train, y_train)
    t_xgb = time.time() - t0

    y_pred_xgb = xgb.predict(X_test)
    eval_xgb = {
    "Model": "XGBoost",
    "train_time_s": round(t_xgb, 4),
    **evaluate(y_test, pd.Series(y_pred_xgb))
}
    
    t0 = time.time()
    lgbm = LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8
    )
    lgbm.fit(X_train, y_train)
    t_lgbm = time.time() - t0

    y_pred_lgbm = lgbm.predict(X_test)
    eval_lgbm = {
    "Model": "LightGBM",
    "train_time_s": round(t_lgbm, 4),
    **evaluate(y_test, pd.Series(y_pred_lgbm))
}

    # ------- Main plot (already exist) -------
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred_xgb, "--", label="XGBoost")
    plt.plot(y_pred_lgbm, ":", label="LightGBM")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plot.png")
    plt.close()

    # Save trained models & y_pred for visualization route
    global VIS_DATA
    VIS_DATA = {
        "df": df,
        "y_test": y_test,
        "y_pred_xgb": y_pred_xgb,
        "y_pred_lgbm": y_pred_lgbm,
        "model_xgb": xgb,
        "model_lgbm": lgbm
    }

    # --- Predict user input ---
    next_day = len(df) + 1
    risk_map = {"low": 0, "medium": 1, "high": 2}
    r_val = risk_map.get(user_input["risk"].lower(), 1)
    X_new = np.array([[next_day, user_input["open"], r_val]])

    pred_xgb_new = float(xgb.predict(X_new)[0])
    pred_lgbm_new = float(lgbm.predict(X_new)[0])

    return render_template("result.html",
        eval_xgb=eval_xgb,
        eval_lgbm=eval_lgbm,
        next_pred_xgb=pred_xgb_new,
        next_pred_lgbm=pred_lgbm_new,
        plot_filename="plot.png",
        user_input=user_input,
        train_size=len(X_train),
        test_size=len(X_test)
    )


# --------------------------------------------------------------------
# ---------------------- DATA VISUALIZATION ---------------------------
# --------------------------------------------------------------------
VIS_DATA = {}  # untuk menyimpan hasil training

@app.route("/visualization")
def visualization():
    """Generate 5 visualization plots"""
    global VIS_DATA

    if not VIS_DATA:
        return redirect(url_for("index"))

    df = VIS_DATA["df"]
    y_test = VIS_DATA["y_test"]
    y_pred_xgb = VIS_DATA["y_pred_xgb"]
    y_pred_lgbm = VIS_DATA["y_pred_lgbm"]
    model_xgb = VIS_DATA["model_xgb"]
    model_lgbm = VIS_DATA["model_lgbm"]

    # 1. Histogram residuals
    plt.figure(figsize=(7,4))
    plt.hist(y_test - y_pred_xgb, bins=30, alpha=0.6, label="XGB")
    plt.hist(y_test - y_pred_lgbm, bins=30, alpha=0.6, label="LGBM")
    plt.legend()
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig("static/vis_residuals.png")
    plt.close()

    # 2. Feature importance (XGB)
    plt.figure(figsize=(6,4))
    plt.barh(["DayIndex", "Open", "Risk_val"], model_xgb.feature_importances_)
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig("static/vis_xgb_fi.png")
    plt.close()

    # 3. Feature importance (LGBM)
    plt.figure(figsize=(6,4))
    plt.barh(["DayIndex", "Open", "Risk_val"], model_lgbm.feature_importances_)
    plt.title("Feature Importance - LightGBM")
    plt.tight_layout()
    plt.savefig("static/vis_lgbm_fi.png")
    plt.close()

    # 4. Historical price trend
    plt.figure(figsize=(10,4))
    plt.plot(df["Price"])
    plt.title("Historical Price Trend")
    plt.tight_layout()
    plt.savefig("static/vis_history.png")
    plt.close()

    # 5. Time series sample prediction
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred_xgb, label="XGB")
    plt.legend()
    plt.title("Time Series Prediction")
    plt.tight_layout()
    plt.savefig("static/vis_timeseries.png")
    plt.close()

    return render_template("visualization.html")
    

if __name__ == "__main__":
    app.run(debug=True)
