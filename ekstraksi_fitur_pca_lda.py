import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

DATA_PATH = "teen_phone_addiction_dataset.csv"

def make_target(classes_from, bins=[0.0, 4.0, 7.0, 10.0001], labels=["Low","Medium","High"]):
    return pd.cut(classes_from, bins=bins, labels=labels, right=False, include_lowest=True)

def main():
    df = pd.read_csv(DATA_PATH)
    target_col = "Addiction_Class"
    df[target_col] = make_target(df["Addiction_Level"])

    # --- PCA ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["ID", "Addiction_Level"]]
    X_num = df[num_cols].copy()
    y = df[target_col].astype(str).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df[target_col] = y
    pca_df.to_csv("pca_2d.csv", index=False)

    exp = pca.explained_variance_ratio_
    pd.DataFrame({
        "component": ["PC1","PC2"],
        "explained_variance_ratio": exp,
        "cumulative": np.cumsum(exp)
    }).to_csv("pca_explained_variance.csv", index=False)

    # --- Plot PCA ---
    plt.figure(figsize=(7,5))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pd.factorize(y)[0], cmap="coolwarm", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - Teen Phone Addiction Dataset")
    plt.colorbar(label="Addiction_Class")
    plt.show()

    # --- LDA ---
    feat_cols = [c for c in df.columns if c not in ["ID", "Name", "Addiction_Level", target_col]]
    X_mixed = df[feat_cols].copy()
    cat_cols = X_mixed.select_dtypes(include=["object","category"]).columns.tolist()
    X_enc = pd.get_dummies(X_mixed, columns=cat_cols, drop_first=True)

    le = LabelEncoder()
    y_int = le.fit_transform(df[target_col].astype(str))

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_enc, y_int)

    lda_df = pd.DataFrame(X_lda, columns=["LD1","LD2"])
    lda_df[target_col] = df[target_col].astype(str).values
    lda_df.to_csv("lda_2d.csv", index=False)

    # --- Plot LDA ---
    plt.figure(figsize=(7,5))
    plt.scatter(lda_df["LD1"], lda_df["LD2"], c=y_int, cmap="viridis", alpha=0.7)
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title("LDA - Teen Phone Addiction Dataset")
    plt.colorbar(label="Addiction_Class")
    plt.show()

    print("✅ pca_2d.csv, pca_explained_variance.csv, lda_2d.csv berhasil dibuat + grafik ditampilkan.")

if __name__ == "__main__":
    main()
