# Laporan Proyek Machine Learning

Proyek pertama Dicoding course Machine Learning Terapan ini mendapatkan bintang 4/5 dari reviewer.

![Penilaian Reviewer](https://github.com/user-attachments/assets/3118a1a3-77dd-4e04-81da-85b4b13b496a)

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Domain Proyek

Kanker adalah sekelompok penyakit yang ditandai oleh pertumbuhan yang tidak terkontrol dan penyebaran sel-sel abnormal. Sel kanker dapat menyebar ke bagian lain dari tubuh melalui sistem darah dan limfa. Jika penyebarannya tidak terkendali, dapat mengakibatkan kematian. Kanker dapat mempengaruhi semua orang, mulai dari anak-anak hingga orang dewasa, baik pria maupun wanita. Kanker paru-paru adalah jenis kanker yang paling umum terjadi pada pria.<sup>[[1]](https://repository.unair.ac.id/124051/5/8.%20Cover-Artikel%20The%20Prevalence%20of%20Cancer%20in%20Indonesia.pdf)</sup>

![Ilustrasi-by-KlikDokter](https://github.com/user-attachments/assets/dd331c40-3fb7-44ed-b985-5648492b79c3)

Kanker paru-paru merupakan salah satu penyebab utama kematian akibat kanker di seluruh dunia. Berdasarkan data dari World Health Organization (WHO) pada tahun 2020, kanker paru-paru menyebabkan 1,80 juta kematian. World Cancer Research Fund melansir statistik kanker paru-paru bahwa Indonesia (men) tercatat memiliki 29 ribu kasus dan menduduki peringkat ke-9 kanker paru terbanyak di dunia pada tahun 2022.<sup>[[2]](https://www.wcrf.org/cancer-trends/lung-cancer-statistics/)</sup> Penelitian yang dipublikasikan pada tahun 2023 berjudul _Lung Cancer in Indonesia_ menyatakan bahwa menurut data dari Kementerian Kesehatan, penyakit tidak menular seperti penyakit kardiovaskular, kanker, dan penyakit paru tidak menular termasuk dalam tiga penyebab utama morbiditas dan mortalitas di Indonesia.<sup>[[3]](https://www.sciencedirect.com/science/article/pii/S1556086423006317)</sup>

Kanker paru-paru merupakan salah satu jenis kanker yang memiliki tingkat mortalitas tinggi dan sering kali tidak terdeteksi pada tahap awal. Masalah ini sangat signifikan karena deteksi dini kanker paru dapat mengurangi tingkat mortalitas. Oleh karena itu, penting untuk membangun model prediktif yang dapat membantu dalam deteksi dini. Dengan adanya model machine learning yang dapat mendeteksi risiko kanker paru-paru pada tahap awal, diharapkan pasien dapat menjalani pengobatan lebih cepat dan mengurangi risiko kematian secara signifikan.

## Business Understanding

### Problem Statements

- Bagaimana cara mengidentifikasi pasien yang berisiko tinggi mengalami kanker paru-paru menggunakan data medis mereka?
- Model machine learning apa yang dapat memberikan prediksi terbaik untuk mendeteksi kanker paru-paru dengan akurasi tinggi?

### Goals

- Membangun model machine learning yang dapat mengidentifikasi pasien dengan risiko tinggi terkena kanker paru-paru menggunakan data medis.
- Mencapai performa prediksi terbaik dengan memanfaatkan algoritma machine learning dan melakukan tuning hyperparameter.

### Solution statements

- Mempertimbangkan lima algoritma klasifikasi yaitu Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan K-Nearest Neighbor.
- Melakukan hyperparameter tuning untuk meningkatkan kinerja model klasifikasi.

## Data Understanding

Dataset yang digunakan adalah Lung cancer dataset<sup>[[4]](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)</sup> dari Kaggle yang memiliki score usability sempurna yaitu 10.00. Usability Score ini merupakan bagian dari sistem penilaian Kaggle yang memberikan gambaran tentang kualitas dan kemudahan penggunaan dataset. Artinya kita dapat mengharapkan bahwa dataset ini telah melalui proses penilaian ketat dan dianggap sangat sesuai untuk keperluan proyek machine learning dalam kemudahan, efisiensi, dan kualitas data. Dataset ini berisi informasi medis sejumlah 1000 pasien yang mencakup 20 variabel kategorikal.

### Deskripsi Variabel

| Nama Kolom               | Deskripsi                                                   |
| ------------------------ | ----------------------------------------------------------- |
| Age                      | Usia pasien. (Numerik)                                      |
| Gender                   | Jenis kelamin pasien. (Kategorikal)                         |
| Air Pollution            | Tingkat paparan polusi udara pada pasien. (Kategorikal)     |
| Alcohol use              | Tingkat penggunaan alkohol oleh pasien. (Kategorikal)       |
| Dust Allergy             | Tingkat alergi debu pada pasien. (Kategorikal)              |
| OccuPational Hazards     | Tingkat bahaya pekerjaan pada pasien. (Kategorikal)         |
| Genetic Risk             | Tingkat risiko genetik pada pasien. (Kategorikal)           |
| chronic Lung Disease     | Tingkat penyakit paru kronis pada pasien. (Kategorikal)     |
| Balanced Diet            | Tingkat pola makan seimbang pasien. (Kategorikal)           |
| Obesity                  | Tingkat obesitas pasien. (Kategorikal)                      |
| Smoking                  | Tingkat merokok oleh pasien. (Kategorikal)                  |
| Passive Smoker           | Tingkat paparan asap rokok pasif pada pasien. (Kategorikal) |
| Chest Pain               | Tingkat nyeri dada pada pasien. (Kategorikal)               |
| Coughing of Blood        | Tingkat batuk berdarah pada pasien. (Kategorikal)           |
| Fatigue                  | Tingkat kelelahan pada pasien. (Kategorikal)                |
| Weight Loss              | Tingkat penurunan berat badan pada pasien. (Kategorikal)    |
| Shortness of Breath      | Tingkat sesak napas pada pasien. (Kategorikal)              |
| Wheezing                 | Tingkat mengi pada pasien. (Kategorikal)                    |
| Swallowing Difficulty    | Tingkat kesulitan menelan pada pasien. (Kategorikal)        |
| Clubbing of Finger Nails | Tingkat pembengkakan ujung jari pada pasien. (Kategorikal)  |

### Data Visualization

Teknik visualisasi data seperti pie chart, reggression plot, violin plot, histogram distribusi normal, dan heatmap matriks korelasi digunakan untuk memahami data.

1.  Pie Chart: Status Distribution of Dataset  
    Visualisasi ini digunakan untuk menunjukkan distribusi kategori pada variabel target Level dalam bentuk persentase. Pie chart memberikan gambaran yang jelas tentang proporsi kategori dalam dataset, seperti Low, Medium, dan High.

    <details>
    <summary>Lihat gambar...</summary>

    ![Pie chart](https://github.com/user-attachments/assets/d2c137c3-a9b2-4290-babb-078fbb7d2863)
    </details>

    Insight:

    - Distribusi tidak merata di antara tiga kategori risiko.
    - Risiko Tinggi (36.5%), Sedang (33.2%), dan Rendah (30.3%).

2.  Regplot: Relationship between Features and Target  
    Untuk setiap fitur dalam dataset, dibuat regression plot terhadap variabel target Level. Regplot ini menunjukkan hubungan linier antara fitur dan target, dengan tambahan Lowess smoothing untuk menampilkan tren yang lebih jelas. Visualisasi ini membantu dalam memahami pola antara fitur dan variabel target.

    <details>
    <summary>Lihat gambar...</summary>

    ![Regplot](https://github.com/user-attachments/assets/5dfd74e6-5737-4207-a7cd-15bdebe7ab05)
    </details>

    Insight:

    - Hubungan antara fitur dan target bervariasi. Beberapa menunjukkan korelasi yang kuat (positif maupun negatif).
    - Meskipun regplot mengasumsikan hubungan linier, kurva Lowess menunjukkan bahwa banyak hubungan sebenarnya non-linear. Artinya peningkatan fitur tidak selalu menghasilkan peningkatan linear pada target.
    - Terdapat fitur-fitur yang memiliki hubungan lebih kuat dan lebih jelas dengan target.

3.  Violin Plot: Distribution of Features by Level  
    Violin plot digunakan untuk menggambarkan distribusi setiap fitur numerik berdasarkan kategori level pada variabel target Level. Plot ini memberikan informasi tentang distribusi dan kepadatan data untuk masing-masing kategori (Low, Medium, High), serta memberikan perbandingan visual yang kuat antar kategori.

    <details>
    <summary>Lihat gambar...</summary>

    ![Violin Plot](https://github.com/user-attachments/assets/3a1725ee-a8bc-45d3-aed6-825de9a4920d)
    </details>

    Insight:

    - Bentuk dan lebar violin plot yang berbeda-beda untuk setiap kategori menunjukkan bahwa distribusi fitur bervariasi secara signifikan antar kelompok 'Level'. Beberapa fitur menunjukkan perbedaan yang jelas antara kelompok 'Level'. Misalnya, untuk fitur 'Usia' pada kelompok 'Tinggi' cenderung lebih tinggi dibandingkan kelompok 'Rendah', maka dapat disimpulkan bahwa usia lebih tinggi terkait dengan tingkat risiko yang lebih tinggi.
    - Garis tengah pada violin plot mewakili median, sedangkan bagian yang lebih tebal menunjukkan rentang interkuartil (IQR). Ini memberikan gambaran tentang lokasi pusat data dan seberapa tersebar data di sekitar median.
    - Titik-titik data yang berada di luar 'kumis' violin plot dianggap sebagai outlier.

4.  Histogram with Normal Distribution Fit  
    Histogram dibuat untuk setiap fitur dalam dataset untuk mengevaluasi distribusi datanya. Selain itu, estimasi mean (μ) dan standard deviation (σ) dari distribusi ditambahkan, bersama dengan KDE plot untuk melihat kepadatan distribusi. Visualisasi ini membantu dalam memahami apakah fitur mengikuti distribusi normal atau tidak.

    <details>
    <summary>Lihat gambar...</summary>

    ![Histogram](https://github.com/user-attachments/assets/285966d6-1c8a-498b-8c3d-1fe4eaf4bba8)
    </details>

    Insight:

    - Jika data mengikuti distribusi normal, maka sebagian besar data akan terkonsentrasi di sekitar rata-rata (mean), dan semakin menjauh dari rata-rata, frekuensinya akan semakin menurun.
    - Jika histogram miring ke kanan, banyak data berkumpul di nilai yang lebih rendah. Jika miring ke kiri, banyak data berkumpul di nilai yang lebih tinggi.
    - Kurtosis yang mengukur seberapa 'runcing' atau 'gepeng' distribusi. Distribusi leptokurtik (runcing), sedangkan distribusi platykurtik (gepeng).

5.  Heatmap: Correlation Matrix of Features  
    Heatmap digunakan untuk memvisualisasikan matriks korelasi antar fitur dalam dataset. Matriks korelasi ini menunjukkan hubungan linear antara fitur, dengan nilai korelasi ditampilkan secara numerik di dalam plot. Heatmap ini sangat berguna untuk mengidentifikasi fitur-fitur yang berkorelasi kuat satu sama lain, yang bisa membantu dalam proses feature selection.

    <details>
    <summary>Lihat gambar...</summary>

    ![Heatmap](https://github.com/user-attachments/assets/ea5c6080-0d98-4385-8eb6-ab6920c0bbca)
    </details>

    Insight:

    - Banyak kotak berwarna merah pekat menandakan tingginya korelasi antar variabel.
    - Variabel 'Air Pollution', 'Alcohol Use', 'Dust Allergy', 'Occupational Hazards', 'Genetic Risk', dan 'Chronic Lung Disease' menunjukkan bahwa paparan polusi udara, konsumsi alkohol, alergi debu, bahaya pekerjaan, risiko genetik, dan penyakit paru-paru kronis saling mempengaruhi.

## Data Preparation

Proses data preparation diperlukan untuk memastikan data yang bersih dan siap digunakan oleh model machine learning. Data preparation meliputi langkah-langkah berikut:

1. Penyesuaian Data: Mengatur kolom `index` pada dataset supaya tidak termasuk ke dalam variabel.
2. Pembersihan Data: Menghapus kolom `'Patient Id'` yang tidak relevan untuk model prediksi.
3. Encoding: Menggunakan teknik label encoding untuk mengubah variabel `'Level'` menjadi variabel numerik.
4. Pemisahan Fitur dan Label: Memisahkan fitur (X) yang merupakan variabel independen dan label (y) yang merupakan variabel dependen untuk model.
5. Train test split: Membagi data menjadi data training (80%) dan data testing (20%).

## Modeling

Pada proyek ini, beberapa model supervised learning diterapkan untuk tugas klasifikasi. Model-model tersebut meliputi:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)

Masing-masing model dilatih pada data pelatihan, dan hasil prediksi diuji menggunakan data uji. Berikut adalah implementasi lengkap dari kelima model:

```python
def models(X_train, y_train):
    # Konversi y_train menjadi array 1D
    y_train = y_train.values.ravel()

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)

    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    return lr, dt, rf, gb, knn

# Prediksi dengan data uji
y_test = y_test.values.ravel()

# Melatih semua model
lr, dt, rf, gb, knn = models(X_train, y_train)

# Prediksi menggunakan model yang telah dilatih
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)
y_pred_knn = knn.predict(X_test)
```

Kelima model machine learning di atas dibangun sekaligus dengan **parameter default** untuk melakukan klasifikasi:

1. Logistic Regression (LR)  
   Algoritma ini digunakan untuk klasifikasi biner, namun dapat diperluas ke klasifikasi multikelas. Model ini menggunakan sigmoid function untuk mengubah keluaran linier menjadi probabilitas. Model ini sederhana dan cepat, namun mungkin tidak terlalu kuat pada dataset yang lebih kompleks.

   Parameter default:

   - `penalty='l2'` Penalti regulasi L2 diterapkan untuk mencegah overfitting.
   - `C=1.0` Parameter inverse regularization strength. Artinya semakin kecil nilai C, semakin kuat regulasi diterapkan.
   - `solver='lbfgs'` Solver digunakan untuk optimasi, dan `'lbfgs'` sangat cocok untuk dataset ukuran kecil hingga menengah.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

2. Decision Tree (DT)  
   Algoritma ini bekerja dengan memisahkan data berdasarkan aturan keputusan yang dihasilkan dari fitur dataset. Setiap node dalam tree mewakili keputusan berbasis satu fitur. Decision Tree cenderung mudah diinterpretasikan namun rentan terhadap overfitting jika tidak dipangkas dengan baik.

   Parameter default:

   - `criterion='gini'` Menggunakan Gini impurity untuk mengukur kualitas split.
   - `max_depth=None` Tidak ada batasan pada kedalaman pohon, yang memungkinkan pohon tumbuh sampai sempurna.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

3. Random Forest (RF)  
   Algoritma _ensemble learning_ yang menggabungkan banyak decision trees untuk membuat prediksi. Setiap tree dilatih pada subset data yang berbeda, dan hasilnya digabungkan menggunakan voting mayoritas untuk klasifikasi. Random Forest sangat akurat dan dapat menangani dataset besar, namun bisa lebih lambat daripada model lain saat jumlah tree besar.

   Parameter default:

   - `n_estimators=100` Jumlah pohon (trees) yang dibangun dalam hutan.
   - `max_depth=None` Setiap pohon dapat tumbuh tanpa batasan kedalaman kecuali semua daun murni.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

4. Gradient Boosting (GB)  
   Algoritma ini membangun model secara bertahap, dengan fokus pada memperbaiki kesalahan model sebelumnya. Ini adalah pendekatan yang kuat namun lebih sensitif terhadap overfitting dibanding Random Forest. Gradient Boosting cocok untuk dataset yang kompleks namun memerlukan tuning parameter yang lebih hati-hati.

   Parameter default:

   - `n_estimators=100` Jumlah pohon yang dibangun secara bertahap, yaitu 100 pohon.
   - `learning_rate=0.1` Kecepatan pembelajaran yang mengontrol kontribusi masing-masing pohon ke model akhir. Yaitu laju pembelajaran sebesar 0.1.
   - `max_depth=3` Membatasi kedalaman maksimum setiap pohon 3, mencegah overfitting.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

5. K-Nearest Neighbors (KNN)  
   Algoritma ini membuat prediksi berdasarkan kedekatan antara sampel baru dengan sampel yang sudah dilabeli dalam data pelatihan. KNN mencari k tetangga terdekat dan memilih kelas mayoritas di antara tetangga tersebut. KNN bekerja dengan baik untuk dataset kecil namun lambat pada dataset besar.

   Parameter default:

   - `n_neighbors=5` Menggunakan 5 tetangga terdekat untuk prediksi.
   - `metric='minkowski'` Menggunakan metrik jarak Minkowski untuk menghitung jarak antara titik data.
   - `p=2` Nilai p=2 merepresentasikan jarak Euclidean.

Setelah proses modeling, akurasi awal adalah sebagai berikut:

```python
                 Model  Accuracy  Precision    Recall
0  Logistic Regression     0.995   0.994792  0.993939
1        Decision Tree     1.000   1.000000  1.000000
2        Random Forest     1.000   1.000000  1.000000
3    Gradient Boosting     1.000   1.000000  1.000000
4                  KNN     0.995   0.994792  0.993939
```

### Hyperparameter Tuning

1. **`StratifiedKFold`**  
   Merupakan teknik cross-validation yang memastikan pembagian data ke dalam beberapa folds dilakukan dengan menjaga persentase sampel untuk setiap kelas. Teknik ini sangat berguna untuk dataset yang memiliki distribusi kelas yang tidak merata seperti klasifikasi kanker paru-paru yang memiliki tiga kelas dengan distribusi 30.3%, 33.2%, dan 36.5%.

   ```python
   folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
   ```

   - `n_splits=2` Jumlah folds atau lipatan. Dataset dibagi menjadi 2 lipatan yang digunakan secara bergantian sebagai data uji dan data latih.

   - `shuffle=True` Menentukan apakah data akan diacak sebelum dibagi menjadi lipatan. Parameter ini penting untuk memastikan distribusi kelas tersebar merata dalam setiap lipatan.

   - `random_state=42` Menetapkan nilai acak untuk memastikan bahwa hasil pembagian data menjadi lipatan dapat direproduksi (reproducible). Nilai 42 sering digunakan sebagai nilai standar untuk tujuan ini.

2. **`GridSearchCV`**  
   Dipakai untuk menguji berbagai kombinasi hyperparameter pada setiap model. Proses ini memungkinkan pencarian hyperparameter optimal yang memberikan hasil terbaik berdasarkan metrik evaluasi yang dipilih, seperti akurasi atau F1 score.

   ```python
   def grid_search(model, folds, params, scoring):
       grid_search = GridSearchCV(model,
                               cv=folds,
                               param_grid=params,
                               scoring=scoring,
                               n_jobs=1,
                               verbose=1)
       return grid_search
   ```

   - `model` Model yang akan diuji, seperti Logistic Regression, Random Forest, atau Gradient Boosting.

   - `cv=folds` Cross-validation yang akan digunakan. Dalam hal ini, menggunakan StratifiedKFold untuk menjaga keseimbangan distribusi kelas.

   - `param_grid=params` dipakai untuk mendefinisikan berbagai nilai hyperparameter yang akan diuji. Setiap kombinasi diuji dengan cross-validation untuk menemukan set hyperparameter yang memberikan kinerja terbaik.

   - `scoring=scoring` digunakan untuk menentukan metrik evaluasi yang digunakan selama tuning.

   - `n_jobs=1` Jumlah pekerjaan atau thread yang akan dijalankan secara paralel. Nilai 1 berarti proses dijalankan secara berurutan, bukan paralel.

   - `verbose` Menentukan tingkat keluaran informasi selama proses grid search. Nilai 1 akan menampilkan informasi proses pencarian di konsol.

3. **`print_best_score_params`**  
   Fungsi ini digunakan untuk mencetak skor terbaik dan parameter hyperparameter terbaik yang ditemukan selama proses grid search.

   ```python
   def print_best_score_params(model):
       print('Best Score: ', model.best_score_)
       print('Best Hyperparameters: ', model.best_params_)
   ```

<details>
<summary>Lihat kode lengkapnya</summary>

```python
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

def grid_search(model, folds, params, scoring):
    grid_search = GridSearchCV(model,
                               cv=folds,
                               param_grid=params,
                               scoring=scoring,
                               n_jobs=1,
                               verbose=1)
    return grid_search

def print_best_score_params(model):
    print('Best Score: ', model.best_score_)
    print('Best Hyperparameters: ', model.best_params_)
```

</details>

## Evaluation

Setelah proses pemodelan yang dilakukan, model Decision Tree, Random Forest, dan Gradient Boosting menunjukkan kinerja sempurna di semua metrik, menjadikan ketiganya model yang sangat kuat untuk proyek klasifikasi ini. Model Logistic Regression dan KNN juga menunjukkan hasil yang hampir sempurna, membuatnya sangat efektif walaupun sedikit di bawah tiga model lainnya dalam hal precision dan recall.

Untuk mengoptimalkan kinerja, telah dilakukan hyperparameter tuning. Hasilnya, akurasi dari model LR meningkat menjadi sempurna kecuali KNN yang tetap berada di tingkat akurasi 0.995. Namun, Logistic Regression dan KNN masih menunjukkan sedikit kelemahan dibandingkan dengan algoritma tree-based seperti Decision Tree, Random Forest, dan Gradient Boosting yang mencapai performa sempurna. Model Random Forest dan Gradient Boosting dipilih sebagai solusi terbaik karena mereka menawarkan performa terbaik dan stabilitas yang lebih tinggi pada dataset ini.

Berdasarkan hasil evaluasi model dan hyperparameter tuning di atas, model **Random Forest (RF)** dan **Gradient Boosting (GB)** dipilih sebagai model terbaik dengan alasan berikut:

1. Performa Sempurna  
   Kedua model ini mencapai akurasi, precision, dan recall 1.000 baik sebelum maupun setelah hyperparameter tuning. Ini menunjukkan bahwa mereka dapat menangani data dengan baik tanpa overfitting atau underfitting.

2. Kapasitas Generalisasi yang Kuat

- Random Forest sangat kuat dalam menangani dataset dengan fitur kompleks dan bersifat robust terhadap overfitting, terutama dengan jumlah pohon yang lebih banyak.
- Gradient Boosting memiliki kemampuan untuk memperbaiki kesalahan dari model sebelumnya secara iteratif, yang menjadikannya pilihan kuat untuk prediksi yang lebih presisi.
- Stabilitas. Kedua model ini terbukti stabil dalam performa mereka baik sebelum maupun setelah tuning, membuat mereka dapat diandalkan dalam penggunaan nyata.

3. Kompleksitas dan Skalabilitas Random Forest cenderung lebih mudah diimplementasikan dan dituning dibanding Gradient Boosting, terutama ketika bekerja dengan dataset yang besar dan kompleks. Meskipun begitu, Gradient Boosting lebih unggul dalam menangani masalah yang memerlukan akurasi prediksi yang sangat tinggi, sehingga dalam beberapa kasus yang sangat sensitif, GB mungkin lebih diutamakan.

Oleh karena itu, Random Forest dipilih karena memberikan kombinasi performa yang sangat baik, skalabilitas, dan kemudahan tuning. Kemudian algoritma Gradient Boosting juga dapat menjadi alternatif kuat ketika dibutuhkan tingkat presisi yang lebih tinggi pada masalah yang lebih kompleks.

### Metrik Evaluasi

Metrik evaluasi yang digunakan dalam proyek ini adalah Accuracy, Precision, dan Recall. Hasil evaluasi menunjukkan perbedaan kinerja di antara model, terutama sebelum dan sesudah hyperparameter tuning:

1. Accuracy: Mengukur persentase prediksi yang benar.
2. Precision: Mengukur ketepatan prediksi positif.
3. Recall: Mengukur kemampuan model dalam mendeteksi semua kasus positif.

#### Sebelum Hyperparameter Tuning:

- Logistic Regression: Akurasi 0.995, Precision 0.994792, Recall 0.993939
- Decision Tree, Random Forest, Gradient Boosting: Semua memiliki akurasi, precision, dan recall sempurna (1.000).
- KNN: Akurasi 0.995, Precision 0.994792, Recall 0.993939

#### Setelah Hyperparameter Tuning:

- Semua model kecuali KNN mencapai performa sempurna dengan akurasi 1.000.
- KNN masih memiliki akurasi yang sama, yaitu 0.995, menunjukkan bahwa model ini mungkin tidak cocok dengan dataset ini dibandingkan dengan algoritma lain.

#### Confusion Matrix:

<details>
<summary>Logistic Regression</summary>

![LR](https://github.com/user-attachments/assets/eba1f951-985b-4e1e-9143-a98604c83feb)

</details>

<details>
<summary>Decision Tree</summary>

![DT](https://github.com/user-attachments/assets/97afe11e-e519-4201-8a67-e850e8f8aede)

</details>

<details>
<summary>Random Forest</summary>

![RF](https://github.com/user-attachments/assets/2f97203a-3dce-4933-8159-8f0d0ef5a074)

</details>

<details>
<summary>Gradient Boosting</summary>

![GB](https://github.com/user-attachments/assets/ac35fbf3-6daf-4c40-9eca-9bf85844028f)

</details>

<details>
<summary>K-Nearest Neighbor</summary>

![KNN](https://github.com/user-attachments/assets/76a89873-c29d-4421-b46a-5d61d3bb5e7e)

</details>

### Kesimpulan

Dengan membangun lima algoritma klasifikasi sekaligus melakukan tuning untuk meningkatkan kinerja, kita menemukan model machine learning yang memiliki prediksi terbaik untuk mendeteksinya. Kini kita dapat mengidentifikasi pasien yang memiliki risiko terjangkit kanker paru-paru menggunakan data medis mereka, dengan menggunakan model algoritma Random Forest dan Gradient Boosting. Model ini dipilih karena memiliki performa prediksi terbaik, pengaturan tuning hyperparameter yang mudah, dan alternatif kuat ketika dibutuhkan tingkat presisi yang lebih tinggi pada masalah yang lebih kompleks.

## Referensi

[1] Asmara et al. (2023). _Lung Cancer in Indonesia_._Journal of Thoracic Oncology_, 18(9), 1134-1145. Tersedia: [tautan](https://www.sciencedirect.com/science/article/pii/S1556086423006317). Diakses pada 17 September 2024.

[2] Santoso, H., Chalidyanto, D., & Laksono, A. D. (2021). _The Prevalence of Cancer in Indonesia: An Ecological Analysis_. _Indian Journal of Forensic Medicine & Toxicology_, 15(3), 3170-3176. Tersedia: [tautan](https://repository.unair.ac.id/124051/5/8.%20Cover-Artikel%20The%20Prevalence%20of%20Cancer%20in%20Indonesia.pdf). Diakses pada 17 September 2024.

[3] World Cancer Research Fund. (n.d). _Lung cancer statistics_. Tersedia: [tautan](https://www.wcrf.org/cancer-trends/lung-cancer-statistics/). Diakses pada 17 September 2024.

[4] Kaggle. (n.d). _Lung cancer dataset_. Tersedia: [tautan](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link). Diakses pada 17 September 2024.
