# imbalanced_diabetes.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import metode sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("diabetes.csv")

# Cek kolom
print("Kolom dataset:", df.columns)
print(df.head())

# Target adalah kolom 'Outcome'
if 'Outcome' not in df.columns:
    print("Error: Kolom 'Outcome' tidak ditemukan dalam dataset.")
    exit()

# ===============================
# 2. Distribusi awal
# ===============================
print("\nDistribusi kelas sebelum balancing:")
print(df['Outcome'].value_counts())

plt.figure(figsize=(6,4))
sns.barplot(x=df['Outcome'].value_counts().index,
            y=df['Outcome'].value_counts().values,
            palette='coolwarm')
plt.xlabel("Outcome")
plt.ylabel("Jumlah")
plt.title("Distribusi Awal Outcome")
plt.show()

# ===============================
# 2.5. Cek & Tangani Missing Value
# ===============================
print("\nJumlah missing value per kolom:")
print(df.isnull().sum())

# Jika ada missing, isi dengan median (aman untuk data numerik)
df = df.fillna(df.median())

# ===============================
# 3. Pisahkan fitur (X) dan target (y)
# ===============================
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# ===============================
# 4. SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("\nDistribusi setelah SMOTE:")
print(pd.Series(y_smote).value_counts())

plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_smote).value_counts().index,
            y=pd.Series(y_smote).value_counts().values,
            palette='coolwarm')
plt.xlabel("Outcome")
plt.ylabel("Jumlah")
plt.title("Distribusi Setelah SMOTE")
plt.show()

# ===============================
# 5. Random Oversampling (ROS)
# ===============================
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

print("\nDistribusi setelah ROS:")
print(pd.Series(y_ros).value_counts())

plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_ros).value_counts().index,
            y=pd.Series(y_ros).value_counts().values,
            palette='coolwarm')
plt.xlabel("Outcome")
plt.ylabel("Jumlah")
plt.title("Distribusi Setelah ROS")
plt.show()

# ===============================
# 6. Random Undersampling (RUS)
# ===============================
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

print("\nDistribusi setelah RUS:")
print(pd.Series(y_rus).value_counts())

plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_rus).value_counts().index,
            y=pd.Series(y_rus).value_counts().values,
            palette='coolwarm')
plt.xlabel("Outcome")
plt.ylabel("Jumlah")
plt.title("Distribusi Setelah RUS")
plt.show()

# ===============================
# 7. SMOTE-ENN
# ===============================
smoteenn = SMOTEENN(random_state=42)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X, y)

print("\nDistribusi setelah SMOTE-ENN:")
print(pd.Series(y_smoteenn).value_counts())

plt.figure(figsize=(6,4))
sns.barplot(x=pd.Series(y_smoteenn).value_counts().index,
            y=pd.Series(y_smoteenn).value_counts().values,
            palette='coolwarm')
plt.xlabel("Outcome")
plt.ylabel("Jumlah")
plt.title("Distribusi Setelah SMOTE-ENN")
plt.show()
