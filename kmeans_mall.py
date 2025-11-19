# kmeans_mall.py
# ============================================
# STREAMLIT + K-MEANS for Mall Customers Dataset
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="K-Means Mall Customers", layout="wide")

st.title("🛍️ K-Means Clustering – Mall Customers")

FILE_NAME = "Mall_Customers.csv"

if not os.path.exists(FILE_NAME):
    st.error(f"File '{FILE_NAME}' not found. Place Mall_Customers.csv in the same folder.")
    st.stop()

# -----------------------
# Load Data
# -----------------------
st.subheader("1) Data Overview")
df = pd.read_csv(FILE_NAME)
st.write("Shape:", df.shape)
st.dataframe(df.head())

# -----------------------
# Descriptive Statistics
# -----------------------
st.subheader("2) Descriptive Statistics")
st.write(df.describe())

# -----------------------
# Missing Value Check
# -----------------------
st.subheader("3) Missing Values")
missing = df.isnull().sum()
st.write(missing)
if missing.sum() == 0:
    st.success("No missing values detected.")
else:
    st.warning("Missing values found. Showing affected rows:")
    st.dataframe(df[df.isnull().any(axis=1)])

# -----------------------
# Outlier Detection (IQR) & Boxplot
# -----------------------
st.subheader("4) Outlier Check and Boxplots")

numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
present_numeric_cols = [c for c in numeric_cols if c in df.columns]

iqr_info = {}
for col in present_numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    outliers = df[(df[col] < low) | (df[col] > high)].shape[0]
    iqr_info[col] = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "Lower bound": low, "Upper bound": high,
        "Outliers": outliers
    }

st.write(pd.DataFrame(iqr_info).T)

fig_box, axes = plt.subplots(1, len(present_numeric_cols), figsize=(5*len(present_numeric_cols), 4))
if len(present_numeric_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, present_numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"{col} Boxplot")

st.pyplot(fig_box)
fig_box.savefig("boxplots_mall.png")
st.success("Boxplots saved: boxplots_mall.png")

# -----------------------
# Histograms
# -----------------------
st.subheader("5) Histograms")

fig_hist, ax_hist = plt.subplots(1, len(present_numeric_cols), figsize=(5*len(present_numeric_cols), 4))
if len(present_numeric_cols) == 1:
    ax_hist = [ax_hist]

for ax, col in zip(ax_hist, present_numeric_cols):
    sns.histplot(df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"{col} Distribution")

st.pyplot(fig_hist)
fig_hist.savefig("histograms_mall.png")
st.success("Histograms saved: histograms_mall.png")

# -----------------------
# Pairwise Regplot
# -----------------------
st.subheader("6) Pairwise Regplot")

cols_for_pair = present_numeric_cols
fig_pair, axes_pair = plt.subplots(
    len(cols_for_pair), len(cols_for_pair),
    figsize=(5*len(cols_for_pair), 5*len(cols_for_pair))
)

for i, x in enumerate(cols_for_pair):
    for j, y in enumerate(cols_for_pair):
        ax = axes_pair[i, j]
        if i == j:
            sns.histplot(df[x], kde=True, ax=ax)
        else:
            sns.regplot(x=x, y=y, data=df, ax=ax, scatter_kws={"s": 20, "alpha": 0.6})

        if j == 0:
            ax.set_ylabel(x)
        if i == len(cols_for_pair) - 1:
            ax.set_xlabel(y)

st.pyplot(fig_pair)
fig_pair.savefig("pairwise_regplot_mall.png")
st.success("Pairwise regplots saved: pairwise_regplot_mall.png")

# -----------------------
# Scatter by Gender
# -----------------------
st.subheader("7) Income vs Spending Score by Gender")

if 'Gender' in df.columns:
    fig_gender, ax_gender = plt.subplots(figsize=(8, 6))
    for gender in df['Gender'].unique():
        sub = df[df['Gender'] == gender]
        ax_gender.scatter(
            sub['Annual Income (k$)'],
            sub['Spending Score (1-100)'],
            s=80, alpha=0.6, label=gender
        )
    ax_gender.set_xlabel('Annual Income (k$)')
    ax_gender.set_ylabel('Spending Score (1-100)')
    ax_gender.set_title('Income vs Spending Score by Gender')
    ax_gender.legend()

    st.pyplot(fig_gender)
    fig_gender.savefig("income_score_by_gender.png")
    st.success("Gender scatter saved: income_score_by_gender.png")
else:
    st.info("Gender column not found.")

# -----------------------
# Features Used for K-Means
# -----------------------
st.subheader("8) Features Used for K-Means")

st.write("Using default features for clustering:")
st.write("- Annual Income (k$)")
st.write("- Spending Score (1-100)")

if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
    X1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
else:
    # fallback: use first two numeric columns
    X1 = df.select_dtypes(include=['int64','float64']).iloc[:, :2].values
    st.warning("Default columns not found. Using first two numeric columns.")

scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

# -----------------------
# Elbow Method
# -----------------------
st.subheader("9) Elbow Method")

inertia = []
K_range = range(1, 11)

for n_clusters in K_range:
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=111)
    km.fit(X1_scaled)
    inertia.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
plt.plot(K_range, inertia, 'o-')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)

st.pyplot(fig_elbow)
fig_elbow.savefig("elbow_mall.png")
st.success("Elbow plot saved: elbow_mall.png")

# -----------------------
# Select K
# -----------------------
st.subheader("10) Select K")

second_diff = np.diff(inertia, 2)
recommended_k = int(np.argmin(second_diff) + 2) if len(second_diff) > 0 else 5

k = st.number_input(
    "Select number of clusters:",
    min_value=1, max_value=15,
    value=recommended_k, step=1
)

# -----------------------
# Build K-Means
# -----------------------
st.subheader("11) Build K-Means")

km_final = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=111)
km_final.fit(X1_scaled)

labels = km_final.labels_
centroids = km_final.cluster_centers_

st.write("Centroids (scaled values):")
st.write(centroids)

# -----------------------
# Decision Boundary
# -----------------------
st.subheader("12) Decision Boundary")

step = 0.02
x_min, x_max = X1_scaled[:, 0].min() - 1, X1_scaled[:, 0].max() + 1
y_min, y_max = X1_scaled[:, 1].min() - 1, X1_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step),
    np.arange(y_min, y_max, step)
)

Z = km_final.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))
plt.imshow(
    Z,
    interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Pastel2, origin='lower', aspect='auto'
)

plt.scatter(X1_scaled[:, 0], X1_scaled[:, 1], c=labels, s=80, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', alpha=0.5)

plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.title(f"K-Means Clusters (k={k})")

st.pyplot(fig_cluster)
fig_cluster.savefig("cluster_decision_boundary_mall.png")
st.success("Decision boundary saved: cluster_decision_boundary_mall.png")

# -----------------------
# Silhouette Score
# -----------------------
st.subheader("13) Silhouette Score")

try:
    s_score = silhouette_score(X1_scaled, labels)
    st.write("Silhouette score:", s_score)
except Exception as e:
    st.error("Failed to compute silhouette score: " + str(e))

# -----------------------
# Save and Download Results
# -----------------------
st.subheader("14) Save and Download Results")

df_result = df.copy()
df_result["cluster"] = labels

out_csv = "kmeans_mall_results.csv"
df_result.to_csv(out_csv, index=False)

st.write(f"Results saved to: {out_csv}")

st.download_button(
    "Download CSV",
    data=df_result.to_csv(index=False),
    file_name=out_csv
)

st.success("K-Means processing completed.")
