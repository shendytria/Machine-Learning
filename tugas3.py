import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Read the data
data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
df = pd.DataFrame(data)

# Pilih kolom numerik
df_numeric = df[['Model Year', 'Electric Range', 'Base MSRP']]

print("=== DATA ASLI (Numerik, 5 baris pertama) ===")
print(df_numeric.head())

# Cek missing values 
print("\n=== CEK Missing Values (sebelum handling) ===")
print(df_numeric.isnull().sum())

# Handling missing values → isi dengan nilai mean
df_numeric_filled = df_numeric.fillna(df_numeric.mean())

print("\n=== Data Setelah Handling Missing Values (5 baris pertama) ===")
print(df_numeric_filled.head())

print("\n=== Cek Missing Values (setelah handling) ===")
print(df_numeric_filled.isnull().sum())

# Cek outlier dengan Z-score
z = np.abs(stats.zscore(df_numeric_filled))
outlier_mask = (z > 3).any(axis=1)
print("\nJumlah outlier terdeteksi dengan Z-score:", outlier_mask.sum())

# Buat dataframe tanpa outlier
df_no_outlier = df_numeric_filled[~outlier_mask]

print("\n=== Data Siap Normalisasi (5 baris pertama) ===")
print(df_no_outlier.head())

# =======================
# Perhitungan Library
# =======================

# Min-Max Normalization (library)
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_no_outlier)
df_normalized = pd.DataFrame(np_scaled, columns=df_no_outlier.columns)

print("\n=== Data Min-Max Normalization (Library, 5 baris pertama) ===")
print(df_normalized.head())

# Z-Score Standardization (library)
z_score_scaler = StandardScaler()
np_standardized = z_score_scaler.fit_transform(df_no_outlier)
df_standardized = pd.DataFrame(np_standardized, columns=df_no_outlier.columns)

print("\n=== Data Z-Score Standardization (Library, 5 baris pertama) ===")
print(df_standardized.head())

# =======================
# Perhitungan Manual
# =======================

# Min-Max Normalization (manual)
df_minmax_manual = (df_no_outlier - df_no_outlier.min()) / (df_no_outlier.max() - df_no_outlier.min())
print("\n=== Data Min-Max Normalization (Manual, 5 baris pertama) ===")
print(df_minmax_manual.head())

# Z-Score Standardization (manual)
df_zscore_manual = (df_no_outlier - df_no_outlier.mean()) / df_no_outlier.std()
print("\n=== Data Z-Score Standardization (Manual, 5 baris pertama) ===")
print(df_zscore_manual.head())

# ===== TABEL PERBANDINGAN STATISTIK =====
print("\n" + "="*80)
print("PERBANDINGAN STATISTIK SEBELUM DAN SESUDAH PREPROCESSING")
print("="*80)

# Statistik data asli
print("\n1. DATA SIAP NORMALISASI:")
print(df_no_outlier.describe().round(2))

# Statistik Min-Max
print("\n2. MIN-MAX NORMALIZATION (Library):")
print(df_normalized.describe().round(4))
print("\n3. MIN-MAX NORMALIZATION (Manual):")
print(df_minmax_manual.describe().round(4))

# Statistik Z-Score
print("\n4. Z-SCORE STANDARDIZATION (Library):")
print(df_standardized.describe().round(4))
print("\n5. Z-SCORE STANDARDIZATION (Manual):")
print(df_zscore_manual.describe().round(4))

# ===== EXPORT KE CSV =====
df_no_outlier.to_csv("Data_Siap_Normalisasi.csv", index=False)
df_normalized.to_csv("MinMax_Normalization_Library.csv", index=False)
df_standardized.to_csv("ZScore_Standardization_Library.csv", index=False)
df_minmax_manual.to_csv("MinMax_Normalization_Manual.csv", index=False)
df_zscore_manual.to_csv("ZScore_Standardization_Manual.csv", index=False)

print("\nFile CSV berhasil dibuat: ")
print("- Data_Siap_Normalisasi.csv")
print("- MinMax_Normalization_Library.csv")
print("- ZScore_Standardization_Library.csv")
print("- MinMax_Normalization_Manual.csv")
print("- ZScore_Standardization_Manual.csv")
