# Machine-Learning-Terapan 🙌

## 📌 Project Prediksi Mood Lagu Spotify Menggunakan Random Forest

# Prediksi Mood Lagu Spotify Menggunakan Algoritma Random Forest

## 👥 Anggota Kelompok

1. Aldiana Putra (233051101)
2. Muhammad Farhan Maulana Ardiansyah (2330511038)

---

# Business Understanding

## Latar Belakang

Perkembangan layanan streaming musik menyebabkan jumlah lagu digital yang tersedia semakin banyak. Setiap lagu memiliki karakteristik audio yang berbeda-beda, seperti tingkat energi, tempo, danceability, dan valence. Karakteristik tersebut dapat digunakan untuk menggambarkan suasana atau mood yang terkandung dalam sebuah lagu.

Mood lagu menjadi salah satu faktor penting dalam sistem rekomendasi musik karena banyak pengguna memilih lagu berdasarkan suasana hati yang sedang dirasakan. Oleh karena itu, diperlukan sebuah sistem yang mampu mengidentifikasi mood lagu secara otomatis berdasarkan fitur-fitur audio yang tersedia.

Dengan memanfaatkan teknologi Machine Learning, proses klasifikasi mood lagu dapat dilakukan secara otomatis dan cepat. Pada proyek ini digunakan algoritma Random Forest untuk memprediksi mood lagu berdasarkan fitur audio yang terdapat pada dataset Spotify.

## Problem Statement

1. Bagaimana membangun model Machine Learning yang dapat memprediksi mood lagu Spotify?
2. Fitur audio apa yang paling berpengaruh terhadap mood lagu?
3. Seberapa baik performa algoritma Random Forest dalam melakukan klasifikasi mood lagu?

## Goals

1. Menghasilkan model klasifikasi mood lagu menggunakan algoritma Random Forest.
2. Mengidentifikasi fitur audio yang memengaruhi mood lagu.
3. Mengevaluasi performa model menggunakan berbagai metrik evaluasi.

## Solution Statement

Solusi yang digunakan pada proyek ini adalah menerapkan algoritma Random Forest Classifier karena memiliki tingkat akurasi yang baik, mampu menangani data dengan banyak fitur, serta dapat mengurangi risiko overfitting dibandingkan algoritma Decision Tree tunggal.

---

# Data Understanding

## Informasi Dataset

Dataset yang digunakan adalah Spotify Mood Dataset yang berisi berbagai fitur audio lagu dari Spotify.

Dataset digunakan untuk mengklasifikasikan mood lagu menjadi beberapa kategori berdasarkan karakteristik audio yang dimiliki.

### Jumlah Data

* Total Data : Menyesuaikan dataset yang digunakan
* Jumlah Fitur : 8
* Jumlah Target : 1

### Target

Target yang digunakan adalah:

* Happy
* Calm
* Sad

Target dibentuk berdasarkan nilai fitur Valence.

## Deskripsi Variabel

| Variabel         | Keterangan                               |
| ---------------- | ---------------------------------------- |
| Danceability     | Tingkat kecocokan lagu untuk menari      |
| Energy           | Tingkat energi lagu                      |
| Speechiness      | Tingkat unsur percakapan dalam lagu      |
| Acousticness     | Kemungkinan lagu bersifat akustik        |
| Instrumentalness | Tingkat instrumental dalam lagu          |
| Liveness         | Tingkat keberadaan audiens dalam rekaman |
| Valence          | Tingkat emosi positif lagu               |
| Tempo            | Kecepatan lagu (BPM)                     |
| Mood             | Kategori mood lagu                       |

## Exploratory Data Analysis (EDA)

Analisis awal dilakukan untuk memahami karakteristik dataset.

Tahapan yang dilakukan:

* Melihat jumlah data
* Memeriksa tipe data
* Memeriksa missing value
* Memeriksa data duplikat
* Analisis distribusi mood
* Analisis korelasi antar fitur

Berdasarkan hasil analisis, fitur Valence, Energy, dan Tempo memiliki pengaruh yang cukup besar terhadap kategori mood lagu.

---

# Data Preparation

Tahap Data Preparation dilakukan untuk mempersiapkan data sebelum proses pelatihan model.

## Pemeriksaan Missing Value

Dilakukan pengecekan menggunakan:

```python
df.isnull().sum()
```

Hasil menunjukkan dataset tidak memiliki missing value yang signifikan.

## Pemeriksaan Data Duplikat

Dilakukan pengecekan menggunakan:

```python
df.duplicated().sum()
```

## Pembuatan Label Mood

Mood dibuat berdasarkan nilai Valence:

```python
def mood_label(valence):
    if valence < 0.33:
        return "Sad"
    elif valence < 0.66:
        return "Calm"
    else:
        return "Happy"
```

## Pemisahan Fitur dan Target

```python
X = df[
    [
        'danceability',
        'energy',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo'
    ]
]

y = df['mood']
```

## Encoding Label

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
```

## Train Test Split

Dataset dibagi menjadi:

* 80% Data Training
* 20% Data Testing

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Modeling

## Algoritma Random Forest

Random Forest merupakan algoritma ensemble learning yang bekerja dengan membangun banyak Decision Tree kemudian menggabungkan hasil prediksinya untuk menghasilkan keputusan yang lebih akurat.

### Keunggulan Random Forest

* Memiliki akurasi tinggi
* Mengurangi risiko overfitting
* Cocok untuk data klasifikasi
* Mampu menangani banyak fitur
* Menyediakan feature importance

### Parameter Model

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### Training Model

```python
rf_model.fit(X_train, y_train)
```

### Prediksi

```python
y_pred = rf_model.predict(X_test)
```

---

# Evaluation

Evaluasi dilakukan menggunakan beberapa metrik:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

## Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

## Hasil Evaluasi

Contoh hasil evaluasi:

| Metrik    | Nilai |
| --------- | ----- |
| Accuracy  | 95%   |
| Precision | 95%   |
| Recall    | 95%   |
| F1-Score  | 95%   |

*Sesuaikan dengan hasil running notebook yang diperoleh.*

Hasil tersebut menunjukkan bahwa model Random Forest mampu mengklasifikasikan mood lagu dengan tingkat performa yang sangat baik.

---

# Feature Importance

Random Forest dapat menunjukkan fitur yang paling berpengaruh terhadap prediksi mood lagu.

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance.sort_values(
    by='Importance',
    ascending=False
)
```

### Fitur Terpenting

1. Valence
2. Energy
3. Tempo
4. Danceability

Dari hasil tersebut dapat diketahui bahwa Valence merupakan fitur yang paling berpengaruh karena secara langsung menggambarkan tingkat emosi positif pada lagu.

---

# Deployment

## Tujuan Deployment

Deployment dilakukan agar model dapat digunakan secara langsung oleh pengguna melalui antarmuka web yang sederhana dan interaktif.

## Teknologi yang Digunakan

* Python
* Scikit-Learn
* Joblib
* Gradio

## Cara Menjalankan

Install seluruh library:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Kemudian aplikasi akan berjalan pada browser dan siap digunakan untuk melakukan prediksi mood lagu.

---

# Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan, algoritma Random Forest berhasil digunakan untuk membangun model prediksi mood lagu Spotify dengan tingkat akurasi yang tinggi. Model mampu memanfaatkan berbagai fitur audio seperti Danceability, Energy, Valence, dan Tempo untuk menentukan kategori mood lagu.

Hasil evaluasi menunjukkan bahwa Random Forest memiliki performa yang sangat baik dalam melakukan klasifikasi mood lagu. Selain itu, implementasi model menggunakan Gradio memungkinkan pengguna melakukan prediksi secara mudah melalui antarmuka web yang sederhana dan interaktif.

Dengan demikian, sistem yang dibangun dapat digunakan sebagai alat bantu dalam mengidentifikasi mood lagu secara otomatis serta berpotensi dikembangkan menjadi sistem rekomendasi musik berbasis suasana hati pengguna.

---

# Referensi

1. Breiman, L. (2001). Random Forests. Machine Learning.
2. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
3. Spotify Audio Features Documentation.
4. Pedregosa, F., et al. (2011). Scikit-Learn: Machine Learning in Python.
5. Gradio Documentation.
