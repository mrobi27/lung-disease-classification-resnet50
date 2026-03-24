# 🫁 Implementasi Transfer Learning ResNet-50 untuk Klasifikasi Multikelas Penyakit Paru pada Citra X-Ray

Proyek ini merupakan implementasi **Deep Learning berbasis Computer Vision** untuk mengklasifikasikan penyakit paru dari citra **Chest X-Ray** menggunakan pendekatan **Transfer Learning dengan arsitektur ResNet-50**.

Model dirancang untuk mengklasifikasikan tiga kondisi paru:

* **Normal**
* **COVID-19**
* **Viral Pneumonia**

Pendekatan ini memanfaatkan kemampuan **Convolutional Neural Network (CNN)** untuk mengekstraksi fitur citra medis secara otomatis.

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

# 📌 Latar Belakang Masalah

Diagnosis penyakit paru melalui citra **Chest X-Ray** seringkali memiliki tantangan karena adanya **kemiripan pola visual** antara beberapa kondisi paru seperti:

* COVID-19
* Viral Pneumonia
* Paru-paru Normal

Kemiripan pola seperti **infiltrat atau bercak paru** dapat menyebabkan:

* kesalahan diagnosis
* proses analisis yang memakan waktu
* ketergantungan pada tenaga ahli radiologi

Untuk mengatasi masalah tersebut, penelitian ini mengembangkan model **Deep Learning berbasis ResNet-50** yang mampu melakukan klasifikasi citra secara otomatis.

---

# 📊 Dataset

Dataset yang digunakan berasal dari Kaggle:

**COVID-19 Radiography Database**

[https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Dataset ini dikembangkan oleh **Tawsifur Rahman et al.**

Dataset asli memiliki **4 kelas**:

* COVID
* Normal
* Viral Pneumonia
* Lung Opacity

Namun pada penelitian ini hanya digunakan **3 kelas**.

### Alasan Menghapus Lung Opacity

Kelas **Lung Opacity** tidak merepresentasikan penyakit spesifik, melainkan hanya temuan radiologis umum yang dapat muncul pada COVID-19 maupun Pneumonia.

Penghapusan kelas tersebut bertujuan untuk:

* mengurangi ambiguitas fitur visual
* meningkatkan kemampuan model dalam membedakan penyakit secara klinis

---

# 🧠 Metodologi Penelitian

### 1️⃣ Image Preprocessing

Tahapan prapemrosesan citra meliputi:

**Image Resizing**

```
224 × 224
```

Ukuran ini menyesuaikan standar input **ResNet-50**.

**Normalisasi**

Nilai piksel dinormalisasi ke rentang:

```
0 – 1
```

**Data Augmentation**

Transformasi yang digunakan:

* Rotation
* Zoom
* Shift

Augmentasi membantu model belajar fitur yang lebih robust.

---

### 2️⃣ Dataset Splitting

Dataset dibagi menjadi:

* Training Set
* Validation Set (val)
* Testing Set

Validation digunakan untuk memantau performa model selama training.

---

### 3️⃣ Pembangunan Model

Model dibangun menggunakan **TensorFlow dan Keras** dengan pendekatan **Transfer Learning**.

Arsitektur yang digunakan:

**ResNet-50**

Keunggulan ResNet-50:

* menggunakan **skip connection**
* mengatasi **vanishing gradient**
* efektif untuk jaringan neural yang sangat dalam

Lapisan akhir dimodifikasi dengan **Fully Connected + Softmax** untuk klasifikasi multikelas.

---

# 📊 Contoh Dataset

<p align="center">
<img src="results/sample_classes.png" width="750">
</p>

Gambar di atas menunjukkan contoh citra **Chest X-Ray** dari setiap kelas dataset.
Setiap kelas memiliki pola visual yang berbeda pada area paru-paru.

---

# 🔄 Data Augmentation

<p align="center">
<img src="results/data_augmentation.png" width="750">
</p>

Data augmentation dilakukan untuk meningkatkan variasi dataset dengan menerapkan beberapa transformasi citra seperti rotasi dan perubahan posisi gambar.

Teknik ini membantu model mengenali pola yang sama meskipun posisi citra berbeda.

---

# 📈 Kurva Training Model

<p align="center">
<img src="results/training_accuracy_loss.png" width="800">
</p>

Grafik di atas menunjukkan pergerakan nilai Accuracy dan Loss selama proses pelatihan. Dengan bantuan ReduceLROnPlateau dan EarlyStopping, model berhasil mencapai titik konvergensi yang optimal tanpa mengalami overfitting.

Hasil evaluasi akhir pada data yang belum pernah dilihat model (Blind Test):

```
Test Accuracy ≈ 86.20%
Test Loss     ≈ 0.3568
```

Angka ini merepresentasikan performa model yang sangat realistis dan tangguh untuk data di dunia nyata (real-world data).

---

# 🔎 Confusion Matrix

<p align="center">
<img src="results/confusion_matrix.png" width="650">
</p>

Confusion Matrix digunakan untuk membedah sedetail apa model mengenali tiap-tiap penyakit.

Berkat penerapan teknik Class Weighting untuk menangani ketimpangan data, model ini sekarang memiliki sensitivitas yang sangat tinggi terhadap COVID-19. Kesalahan deteksi yang membahayakan (False Negatives pada pasien COVID) berhasil ditekan secara drastis. Sebagai alat triage medis, model ini lebih memilih bersikap "waspada" (munculnya False Positives pada kelas Normal) daripada meloloskan pasien yang sebenarnya terinfeksi.

---

# 🖼 Contoh Prediksi Model

<p align="center">
<img src="results/random_predictions_showcase.png" width="850">
</p>

Gambar di atas menampilkan simulasi prediksi model pada beberapa citra dari **test dataset** yang diambil secara acak.

Setiap panel gambar menampilkan:

* Label Diagnosis Sebenarnya (Ground Truth)
* Tebakan Diagnosis dari AI (Prediction)
* Tingkat Keyakinan Model (Confidence Score)

---

# 📁 Project Structure

```
LUNG-DISEASE-CLASSIFICATION-RESNET50
│
├── 📂 dataset
│   ├── 📂 raw
│   │   └── Dataset asli dari Kaggle
│   │
│   └── 📂 processed
│       └── Dataset setelah preprocessing
│
├── 📂 models
│   ├── resnet50_best_model.h5
│   │   └── Model terbaik hasil training
│   │
│   └── class_indices.json
│       └── Mapping label kelas
│
├── 📂 notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_data_augmentation.ipynb
│   ├── 04_model_training_resnet.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_visualization_results.ipynb
│   ├── 07_model_inference.ipynb
│   └── 08_prepare_deployment.ipynb
│
├── 📂 results
│   ├── confusion_matrix.png
│   ├── data_augmentation.png
│   ├── random_predictions_showcase.png
│   ├── sample_classes.png
│   └── training_accuracy_loss.png
│
├── 📂 utils
│   └── Helper functions untuk preprocessing dan inference
│
├── app.py
│   └── Aplikasi web (Streamlit) untuk diagnosis paru
│
├── requirements.txt
│   └── Daftar library Python yang digunakan
│
└── README.md
    └── Dokumentasi project
```

# 📥 Download Model & Hasil Training

Karena keterbatasan ukuran GitHub, file hasil training disimpan di Google Drive:

👉 https://drive.google.com/drive/folders/1z5SccpMrDoy2xTNH4-GE-a0X7-Y3jINH?usp=sharing

Isi folder:

* resnet50_best_model.h5
* training results
* visualisasi model

📌 Setelah download:
Letakkan file .h5 ke dalam folder:

models/

---

# 🚀 Pengembangan Selanjutnya

Beberapa pengembangan yang direncanakan:

* Deployment aplikasi menggunakan **Streamlit**
* Visualisasi **Grad-CAM untuk Explainable AI**
* Peningkatan dataset
* Optimasi performa model

---

# 👨‍💻 Penulis

**Robi**
Mahasiswa Informatika

Proyek: **Deep Learning Lung Disease Classification**

