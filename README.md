<div align="center">

# 🫁 Implementasi Transfer Learning ResNet-50 untuk Klasifikasi Multikelas Penyakit Paru pada Citra X-Ray

[![Website](https://img.shields.io/badge/🌐_Buka_Live_Documentation_Website-0f172a?style=for-the-badge&logo=github)](https://mrobi27.github.io/lung-disease-classification-resnet50/)

Proyek ini merupakan implementasi **Deep Learning berbasis Computer Vision** untuk mengklasifikasikan penyakit paru dari citra **Chest X-Ray** menggunakan pendekatan **Transfer Learning dengan arsitektur ResNet-50**.

</div>

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 📌 Latar Belakang Masalah

Diagnosis penyakit paru melalui citra **Chest X-Ray** seringkali memiliki tantangan karena adanya **kemiripan pola visual** antara beberapa kondisi paru seperti **COVID-19**, **Viral Pneumonia**, dan **Paru-paru Normal**. 

Kemiripan pola seperti *infiltrat* atau bercak paru dapat menyebabkan:
1. Kesalahan diagnosis (*human error*).
2. Proses analisis klinis yang memakan waktu.
3. Ketergantungan tinggi pada ketersediaan tenaga ahli radiologi.

> **💡 Solusi:** Untuk mengatasi masalah tersebut, penelitian ini mengembangkan model **Deep Learning berbasis ResNet-50** yang mampu melakukan klasifikasi citra secara otomatis, presisi, dan transparan.

---

## 📊 Dataset

Dataset citra medis sekunder yang digunakan berasal dari Kaggle:
🔗 **[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)** *(Tawsifur Rahman et al.)*

Dataset asli memiliki 4 kelas, namun pada penelitian ini **hanya digunakan 3 kelas** (Normal, COVID-19, Viral Pneumonia).

### ⚠️ Alasan Eksklusi Kelas 'Lung Opacity'
Kelas **Lung Opacity** tidak merepresentasikan penyakit spesifik, melainkan hanya temuan radiologis umum yang dapat muncul pada COVID-19 maupun Pneumonia. Penghapusan kelas tersebut bertujuan untuk:
* Mengurangi ambiguitas fitur visual.
* Menajamkan *decision boundary* model dalam membedakan penyakit secara klinis.

<p align="center">
  <img src="results/sample_classes.png" width="700">
  <br>
  <i>Contoh citra Chest X-Ray dari setiap kelas dataset.</i>
</p>

---

## 🧠 Metodologi Penelitian

### 1️⃣ Image Preprocessing
Tahapan prapemrosesan citra meliputi:
* **Image Resizing:** Menyeragamkan dimensi citra menjadi `224 × 224` piksel menyesuaikan standar input ResNet-50.
* **Min-Max Normalization:** Menskalakan nilai piksel ke rentang `[0, 1]` untuk mempercepat konvergensi komputasi.
* **Data Augmentation:** Menerapkan rotasi, zoom, dan pergeseran (*shift*) untuk memperkaya fitur data dan mencegah model menghafal (*overfitting*).

<p align="center">
  <img src="results/data_augmentation.png" width="700">
</p>

### 2️⃣ Dataset Splitting
Membagi kumpulan data secara *advanced* menjadi 3 bagian:
* **Training Set** (Pelatihan)
* **Validation Set** (Validasi saat proses training berjalan)
* **Testing Set** (Pengujian buta / *Blind test* di akhir)

### 3️⃣ Pembangunan Model (CNN)
Model dibangun menggunakan **TensorFlow/Keras** dengan pendekatan **Transfer Learning**.
* Memuat bobot (*weights*) dasar dari arsitektur **ResNet-50** yang memanfaatkan *Skip Connection* untuk mencegah *vanishing gradient*.
* **Fine-Tuning:** Memodifikasi *Fully Connected Layer* di bagian akhir dengan aktivasi **Softmax** untuk klasifikasi probabilitas multikelas.

---

## 📈 Evaluasi & Hasil Analisis

### Kurva Training Model
<p align="center">
  <img src="results/training_accuracy_loss.png" width="750">
</p>

Grafik menunjukkan pergerakan nilai Accuracy dan Loss selama proses pelatihan. Dengan bantuan `ReduceLROnPlateau` dan `EarlyStopping`, model berhasil mencapai titik konvergensi yang optimal tanpa mengalami *overfitting*.

**Hasil evaluasi akhir pada data Testing (Blind Test):**
* **Test Accuracy:** ≈ 86.20%
* **Test Loss:** ≈ 0.3568

### Confusion Matrix
<p align="center">
  <img src="results/confusion_matrix.png" width="600">
</p>

Berkat penerapan teknik *Class Weighting* untuk menangani ketimpangan data, model memiliki sensitivitas yang sangat tinggi terhadap COVID-19. Kesalahan deteksi fatal (*False Negatives* pada pasien terinfeksi) berhasil ditekan secara drastis sebagai prioritas keamanan klinis pasien.

### Contoh Prediksi Model
<p align="center">
  <img src="results/random_predictions_showcase.png" width="800">
</p>
Setiap panel gambar di atas menampilkan simulasi tebakan AI yang terdiri dari Label Asli (Ground Truth), Tebakan Diagnosis (Prediction), dan Tingkat Keyakinan Model (Confidence Score).

---

## 📁 Project Structure

```text
LUNG-DISEASE-CLASSIFICATION-RESNET50
├── docs/                         # Source code untuk website dokumentasi (GitHub Pages)
├── models/                       # Tempat menyimpan model .h5 (download via GDrive)
├── notebooks/                    # File riset Jupyter Notebook (01_data - 08_deploy)
├── results/                      # Output visualisasi, grafik, dan confusion matrix
├── utils/                        # Helper functions untuk preprocessing
├── app.py                        # (WIP) Aplikasi web berbasis Streamlit
├── requirements.txt              # Daftar library dependencies
└── README.md                     # Dokumentasi utama proyek
```

---

## 📥 Download Model & Hasil Training

Karena keterbatasan ukuran *storage* GitHub, file hasil *training* (Model AI) disimpan di Google Drive:

🔗 **[Akses Google Drive - LungAI Model](https://drive.google.com/drive/folders/1z5SccpMrDoy2xTNH4-GE-a0X7-Y3jINH?usp=sharing)**

**Cara Penggunaan:**
1. Download file `resnet50_best_model.h5`.
2. Letakkan file tersebut ke dalam folder `/models/` di repository lokal Anda.

---

## 🚀 Pengembangan Selanjutnya (Future Works)

Beberapa rencana pengembangan riset di masa depan:
* **Eksplorasi Arsitektur Baru:** Menguji perbandingan performa dengan model modern seperti *Vision Transformers (ViT)* atau *EfficientNet*.
* **Explainable AI (Grad-CAM):** Memvisualisasikan *heatmap* pada area paru yang menjadi fokus AI sebagai *second opinion* bagi dokter.
* **Validasi Klinis:** Pengujian aplikasi secara langsung bersama tenaga ahli radiologi di rumah sakit.
* **Cloud Deployment:** Mengunggah aplikasi (*Dashboard*) ke server cloud publik agar dapat diakses oleh fasilitas kesehatan secara luas.

---

## 👨‍💻 Penulis

**Muhammad Robi Ardita** *Mahasiswa Informatika | AI Enthusiast*

* 🌐 Website Dokumentasi: [LungAI Project](https://mrobi27.github.io/lung-disease-classification-resnet50/)
* 💻 GitHub: [@mrobi27](https://github.com/mrobi27)
