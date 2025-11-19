import pandas as pd #library utama untuk olah data
import numpy as np #library untuk komputasi numerik
from scipy import stats #library untuk komputasi stats
import matplotlib.pyplot as plt #library untuk visualisasi data

# Membaca data
df = pd.read_csv('teen_phone_addiction_dataset.csv')

print("=== 5 Data Pertama ===")
print(df.head())

print("\n=== Missing Values per Kolom ===")
print(df.isnull().sum().to_frame("Jumlah Missing"))

# ---------------- Handling Missing Values ----------------

# Menghapus missing value
df_cleaned = df.dropna() #Cepat, tapi bisa bikin data hilang banyak kalau NaN banyak.

print("\n=== Missing Values setelah dihapus ===")
print(df.isnull().sum().to_frame("Jumlah Missing Value Setelah Dihapus"))

# Isi missing value numerik dengan mean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean()) #Cocok kalau datanya tanpa outlier.

print("\n=== Missing Values setelah diganti dengan mean ===")
print(df.isnull().sum().to_frame("Jumlah Missing Value Setelah Diatasi dengan Mean"))

# Isi missing value numerik dengan median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median()) #Lebih aman kalau ada outlier

print("\n=== Missing Values setelah diganti dengan median ===")
print(df.isnull().sum().to_frame("Jumlah Missing Value Setelah Diatasi dengan Median"))

# Isi missing value kategori dengan mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing Values setelah diganti dengan mode ===")
print(df.isnull().sum().to_frame("Jumlah Missing Value Setelah Diatasi dengan Mode"))

# ---------------- Handling Outlier ----------------

# Menghitung Z-score
kolom_numerik = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[kolom_numerik]))

# Konversi ke DataFrame agar bisa indexing pakai nama kolom
z_scores = pd.DataFrame(z_scores, columns=kolom_numerik)

# Menentukan outlier dengan threshold Z-score > 3 
outliers_z = df[(z_scores > 3).any(axis=1)]
print(f"\n=== Jumlah Outlier (Z-score > 3): {len(outliers_z)}")
print("\n=== Tabel Outlier (Z-score > 3):")
print(outliers_z)

# Menghapus Outlier
df_cleaned = df[(z_scores <= 3).all(axis=1)]

print("\n=== Data setelah outlier dihapus ===")
print(df_cleaned.describe())

# Ganti Outlier dengan MEAN
df_mean = df.copy()
for col in kolom_numerik:
    mean_val = df[col].mean()
    df_mean[col] = np.where(z_scores[col] > 3, mean_val, df[col])

print("\n=== Data setelah outlier diganti dengan MEAN ===")
print(df_mean.describe())

# Ganti Outlier dengan MEDIAN
df_median = df.copy()
for col in kolom_numerik:
    median_val = df[col].median()
    df_median[col] = np.where(z_scores[col] > 3, median_val, df[col])

print("\n=== Data setelah outlier diganti dengan MEDIAN ===")
print(df_median.describe())

# Ganti Outlier dengan IQR (Winsorizing)
df_iqr = df.copy()

# Hitung batas bawah & atas IQR per kolom numerik
batas_bawah = {}
batas_atas = {}

for col in kolom_numerik:
    Q1 = df[col].quantile(0.25)   # Kuartil 1
    Q3 = df[col].quantile(0.75)   # Kuartil 3
    IQR = Q3 - Q1                 # Interquartile Range
    
    batas_bawah[col] = Q1 - 1.5 * IQR
    batas_atas[col] = Q3 + 1.5 * IQR

    # Ganti nilai di luar batas dengan batasnya
    df_iqr[col] = np.where(df[col] < batas_bawah[col], batas_bawah[col], df[col])
    df_iqr[col] = np.where(df[col] > batas_atas[col], batas_atas[col], df_iqr[col])

print("\n=== Data setelah outlier ditangani dengan IQR (Winsorizing) ===")
print(df_iqr.describe())

# ---------------- Visualisasi ----------------

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# 1. Sebelum
df[kolom_numerik].boxplot(ax=axes[0])
axes[0].set_title("Sebelum Handling")

# 2. Z-score
df_cleaned[kolom_numerik].boxplot(ax=axes[1])
axes[1].set_title("Hapus Outlier (Z-score)")

# 3. Mean
df_mean[kolom_numerik].boxplot(ax=axes[2])
axes[2].set_title("Ganti Outlier (Mean)")

# 4. Median
df_median[kolom_numerik].boxplot(ax=axes[3])
axes[3].set_title("Ganti Outlier (Median)")

# 5. IQR
df_iqr[kolom_numerik].boxplot(ax=axes[4])
axes[4].set_title("Ganti Outlier (IQR)")

# Rotasi label supaya tidak bertumpuk
for ax in axes:
    ax.tick_params(axis='x', rotation=90, labelsize=7)

plt.tight_layout()
plt.show()
