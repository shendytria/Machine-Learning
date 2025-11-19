# kmodes_app.py
# ============================================
# STREAMLIT + K-MODES (Modul-parallel features)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="K-Modes Auto", layout="wide")

st.title("🔍 K-Modes Clustering")

st.write("Upload dataset CSV")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is None:
    st.info("Please upload a CSV dataset for K-Modes.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("1) Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# -----------------------
# Statistik & missing
# -----------------------
st.subheader("2) Descriptive Statistics")
# describe for object columns
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(obj_cols) == 0:
    st.warning("No object-type (string) columns detected. The program will attempt to detect categorical columns based on low unique counts.")
# show top value counts for each categorical-like column
categorical_candidates = []
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() <= 20:
        categorical_candidates.append(col)

if len(categorical_candidates) == 0:
    st.error("No categorical columns detected. K-Modes requires categorical features.")
    st.stop()

st.write("Detected categorical columns:", categorical_candidates)
for col in categorical_candidates:
    st.write(f"Top frequencies for {col}:")
    st.write(df[col].value_counts().head(10))

st.subheader("3) Missing Values")
st.write(df.isnull().sum())
if df.isnull().sum().sum() > 0:
    st.warning("Missing values detected. Displaying rows that contain missing data:")
    st.dataframe(df[df.isnull().any(axis=1)])

# -----------------------
# "Outlier" analog for categorical: rare categories
# -----------------------
st.subheader("4) Rare Categories (Frequency <1%)")
rare_info = {}
for col in categorical_candidates:
    counts = df[col].value_counts(normalize=True)
    rare = counts[counts < 0.01]  # <1%
    rare_info[col] = len(rare)
st.write(pd.Series(rare_info, name='Rare category count (<1%)'))

# -----------------------
# Visual tiga fitur kategori (countplots)
# -----------------------
st.subheader("5) Visualization Countplots")
top3 = sorted(categorical_candidates, key=lambda c: df[c].nunique(), reverse=True)[:3]
fig_count, axes = plt.subplots(1, len(top3), figsize=(5*len(top3), 4))
if len(top3) == 1:
    axes = [axes]
for ax, col in zip(axes, top3):
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Countplot {col}")
st.pyplot(fig_count)
fig_count.savefig("kmodes_countplots.png")
st.success("Countplots saved: kmodes_countplots.png")

# -----------------------
# Pairwise crosstab heatmaps (analog regplot 3x3)
# -----------------------
st.subheader("6) Pairwise Crosstab Heatmaps")
cols_for_pair = top3
fig_pair, axes_pair = plt.subplots(len(cols_for_pair), len(cols_for_pair), figsize=(5*len(cols_for_pair), 5*len(cols_for_pair)))
for i, a in enumerate(cols_for_pair):
    for j, b in enumerate(cols_for_pair):
        ax = axes_pair[i, j]
        if i == j:
            counts = df[a].value_counts()
            sns.barplot(x=counts.values, y=counts.index, ax=ax)
            ax.set_title(a)
        else:
            ct = pd.crosstab(df[a], df[b])
            sns.heatmap(ct, cmap="viridis", ax=ax)
            ax.set_title(f"{a} vs {b}")
st.pyplot(fig_pair)
fig_pair.savefig("kmodes_pairwise_heatmaps.png")
st.success("Pairwise heatmaps saved: kmodes_pairwise_heatmaps.png")

# -----------------------
# Per-group scatter analog: if dataset has a numeric pair and a group column, show scatter
# -----------------------
st.subheader("7) Scatter Plot")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = categorical_candidates

if len(num_cols) >= 2 and len(cat_cols) >= 1:
    st.write("Select columns for scatter plot visualization:")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_x = st.selectbox("Numeric X-axis:", ["-- Select --"] + num_cols)
    with col2:
        num_y = st.selectbox("Numeric Y-axis:", ["-- Select --"] + num_cols)
    with col3:
        group_col = st.selectbox("Color group (categorical):", ["-- Select --"] + cat_cols)

    # Only show scatter plot if all three selections are valid
    if (
        num_x != "-- Select --"
        and num_y != "-- Select --"
        and group_col != "-- Select --"
    ):
        fig_scat, ax_scat = plt.subplots(figsize=(8, 6))

        for g in df[group_col].unique():
            sub = df[df[group_col] == g]
            ax_scat.scatter(sub[num_x], sub[num_y], label=str(g), alpha=0.6, s=60)

        ax_scat.set_xlabel(num_x)
        ax_scat.set_ylabel(num_y)
        ax_scat.set_title(f"{num_x} vs {num_y} grouped by {group_col}")
        ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        st.pyplot(fig_scat)
    else:
        st.info("Please select all fields (X, Y, and Group) to display the scatter plot.")
else:
    st.info("Scatter visualization skipped: dataset must have at least 2 numeric columns and 1 categorical column.")

# -----------------------
# Siapkan data kategorikal untuk K-Modes
# -----------------------
st.subheader("8) Categorical Encoding (Label Encoding)")
X = df[categorical_candidates].copy()
encoders = {}
X_enc = X.copy()
for col in X.columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

st.write("Encoding completed. Encoded columns:", X_enc.columns.tolist())

# -----------------------
# Elbow Method for KModes
# -----------------------
st.subheader("9) Elbow Method")

costs = []
K_range = range(2, 16)
X_arr = X_enc.values

for k in K_range:
    km_temp = KModes(
        n_clusters=k,
        init='Cao',
        n_init=5,
        verbose=0,
        random_state=42
    )
    km_temp.fit_predict(X_arr)
    costs.append(km_temp.cost_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
plt.plot(list(K_range), costs, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Cost")
plt.title("KModes Elbow Plot")
plt.grid(True)

st.pyplot(fig_elbow)
fig_elbow.savefig("kmodes_elbow.png")
st.success("Elbow plot saved: kmodes_elbow.png")


# -----------------------
# Select K and Fit KModes
# -----------------------
st.subheader("10) Select K and Run KModes")

k_kmodes = st.number_input(
    "Select number of clusters (K):",
    min_value=2,
    max_value=30,
    value=4,
    step=1
)

km = KModes(
    n_clusters=int(k_kmodes),
    init='Cao',
    n_init=5,
    verbose=0,
    random_state=42
)

clusters = km.fit_predict(X_arr)
df_res = df.copy()
df_res['cluster'] = clusters

st.success(f"KModes clustering completed with K = {int(k_kmodes)}")


# -----------------------
# Cluster Centroids (Modes)
# -----------------------
st.subheader("11) Cluster Centroids")

modes = km.cluster_centroids_
modes_df = pd.DataFrame(modes, columns=X.columns)

# decode encoded categories
for col in modes_df.columns:
    modes_df[col] = modes_df[col].astype(int).apply(
        lambda x: encoders[col].inverse_transform([x])[0]
    )

st.dataframe(modes_df)


# -----------------------
# Cluster Distribution
# -----------------------
st.subheader("12) Cluster Distribution")

st.bar_chart(
    df_res['cluster'].value_counts().sort_index()
)


# -----------------------
# Heatmap Cluster vs Largest Column
# -----------------------
st.subheader("13) Heatmap Cluster vs Largest Feature")

best_col = sorted(
    X.columns,
    key=lambda c: df[c].nunique(),
    reverse=True
)[0]

fig_heat, ax_heat = plt.subplots(figsize=(10, 4))

sns.heatmap(
    pd.crosstab(df_res['cluster'], df_res[best_col]),
    annot=True,
    fmt='d',
    cmap='viridis',
    ax=ax_heat
)

ax_heat.set_title(f"Cluster vs {best_col}")

st.pyplot(fig_heat)
fig_heat.savefig("kmodes_heatmap.png")

st.success("Heatmap saved: kmodes_heatmap.png")

# -----------------------
# Decision boundary analog: pilih 2 kolom encoded untuk meshgrid & predict
# -----------------------
st.subheader("14) Decision Region")

cat_cols = X_enc.columns.tolist()

colA, colB = st.columns(2)
with colA:
    feat_x = st.selectbox("Select categorical feature for X-axis:", ["-- Select --"] + cat_cols)
with colB:
    feat_y = st.selectbox("Select categorical feature for Y-axis:", ["-- Select --"] + cat_cols)

if (
    feat_x != "-- Select --"
    and feat_y != "-- Select --"
    and feat_x != feat_y
):

    A = X_enc[[feat_x, feat_y]].values

    x_min, x_max = A[:,0].min()-1, A[:,0].max()+1
    y_min, y_max = A[:,1].min()-1, A[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 0.5)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # full grid for KModes
    full_grid = np.zeros((grid.shape[0], X_enc.shape[1]), dtype=int)

    idx0 = X_enc.columns.get_loc(feat_x)
    idx1 = X_enc.columns.get_loc(feat_y)

    full_grid[:, idx0] = grid[:,0].astype(int)
    full_grid[:, idx1] = grid[:,1].astype(int)

    Z = km.predict(full_grid)
    Z = Z.reshape(xx.shape)

    fig_db, ax_db = plt.subplots(figsize=(10, 7))
    plt.imshow(
        Z,
        interpolation='nearest',
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Pastel2,
        origin='lower',
        aspect='auto'
    )

    plt.scatter(A[:,0], A[:,1], c=clusters, s=80, edgecolor='k')
    plt.xlabel(f"{feat_x} (encoded)")
    plt.ylabel(f"{feat_y} (encoded)")
    plt.title("KModes Decision Region")

    st.pyplot(fig_db)
    fig_db.savefig("kmodes_decision_region_custom.png")
    st.success("Custom decision-region saved: kmodes_decision_region_custom.png")

else:
    st.info("Please select two different categorical features.")

# -----------------------
# Silhouette Score
# -----------------------
st.subheader("15) Silhouette Score")

try:
    if len(set(clusters)) > 1:
        s_score = silhouette_score(X_arr, clusters, metric='hamming')
        st.write("Silhouette score (Hamming):", s_score)
    else:
        st.write("Silhouette score cannot be calculated because there is only one cluster.")
except Exception as e:
    st.error("Failed to compute silhouette score: " + str(e))


# -----------------------
# Save and Download Results
# -----------------------
st.subheader("16) Save and Download Results")

out_csv = "kmodes_results.csv"
df_res.to_csv(out_csv, index=False)

st.write(f"Results saved to: {out_csv}")

st.download_button(
    "Download KModes Result CSV",
    data=df_res.to_csv(index=False),
    file_name=out_csv
)

st.success("KModes process completed.")
