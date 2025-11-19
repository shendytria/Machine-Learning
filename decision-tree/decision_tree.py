import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay, classification_report

#membaca data
dataframe = pd.read_excel('BlaBla.xlsx')

data=dataframe[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']]
print("data awal".center(75,"="))
print(data)
print("=========================================================================")

#groupping yang dibagi menjadi dua
print("GROUPING VARIABEL".center(75,"="))
x = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print("data variabel".center(75,"="))
print(x)
print("data kelas".center(75,"="))
print(y)
print("=========================================================================")  

#pembagian trainig dan testing
print("SPLITTING DATA 20-80".center(75,"="))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("instance variabel data training".center(75,"="))
print(x_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(x_test)
print("instance kelas data testing".center(75,"="))
print(y_test)
print("=========================================================================")

#pemodelan pemodelan decision tree
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(x_train, y_train)

#prediksi decision tree
print("instance prediksi data testing")
y_pred = decision_tree.predict(x_test)
print(y_pred)
print("=========================================================================")  
print()

#prediksi akurasi
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print("Akurasi", accuracy, "%")

#display classification report
print("CLASSIFICATION REPORT DECISION TREE".center(75,"="))
print(classification_report(y_test, y_pred))

#display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

# COBA INPUT
print("CONTOH INPUT".center(75, "-"))

# Input dasar pasien
A = int(input("Umur Pasien = "))
print("Isi Jenis kelamin dengan 0 jika Perempuan dan 1 jika Laki-Laki")
B = input("Jenis Kelamin Pasien = ")

print("Isi Y jika mengalami dan N jika tidak")
C = input("Apakah pasien mengalami C? = ")
D = input("Apakah pasien mengalami D? = ")
E = input("Apakah pasien mengalami E? = ")
F = input("Apakah pasien mengalami F? = ")
G = input("Apakah pasien mengalami G? = ")
H = input("Apakah pasien mengalami H? = ")
I = input("Apakah pasien mengalami I? = ")
J = input("Apakah pasien mengalami J? = ")
K = input("Apakah pasien mengalami K? = ")
L = input("Apakah pasien mengalami L? = ")
M = input("Apakah M? = ")

# Inisialisasi variabel
umur_k = 0
A_k = 0
B_k = 0

# Kategori umur pasien
if A < 21:
    A_k = 1
if A > 20 and A < 31:
    A_k = 2
if A > 30 and A < 41:
    A_k = 3
if A > 40 and A < 51:
    A_k = 4
if A > 50:
    A_k = 5

print("Kode umur pasien adalah", A_k)

# Kode jenis kelamin
if B == "P":
    B_k = 1
else:
    B_k = 0

# Konversi gejala ke bentuk numerik
if C == "Y": C = 1
else: C = 0

if D == "Y": D = 1
else: D = 0

if E == "Y": E = 1
else: E = 0

if F == "Y": F = 1
else: F = 0

if G == "Y": G = 1
else: G = 0

if H == "Y": H = 1
else: H = 0

if I == "Y": I = 1
else: I = 0

# Konversi gejala J, K, L, M
if J == "Y": 
    J = 1
else: 
    J = 0

if K == "Y": 
    K = 1
else: 
    K = 0

if L == "Y": 
    L = 1
else: 
    L = 0

if M == "Y": 
    M = 1
else: 
    M = 0

# Buat data input untuk prediksi
Train = [A_k, B_k, C, D, E, F, G, H, I, J, K, L, M]
print("\nData Input Pasien:", Train)

# Ubah ke DataFrame (biar sesuai format model)
test = pd.DataFrame([Train])

# Prediksi dengan model Decision Tree
predtest = decision_tree.predict(test)

# Hasil prediksi
if predtest == 1:
    print("Pasien Positif")
else:
    print("Pasien Negatif")
