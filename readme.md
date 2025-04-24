# Proyek : Klasifikasi Kelayakan Kredit Komputer

Repositori ini memuat seluruh pipeline *machine learning* untuk memprediksi kelayakan kredit komputer menggunakan **Decision Tree Classifier** pada dataset `dataset_buys_comp.csv`.

Nama: Muhammad Arkan Raihan

NIM: 1227050085

Kelas: Praktikum Pembelajaran Mesin E

---

## 1. Persiapan Lingkungan
1. **Install dependensi**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn jupyterlab joblib
   ```
2. **Struktur folder minimal**
   ```
   ├── data
   │   └── dataset_buys_comp.csv
   ├── notebooks
   │   └── klasifikasi_kelayakan_pc.ipynb
   ├── train_decision_tree.py
   └── README.md
   ```

## 2. Eksplorasi Data (EDA)
1. `pd.read_csv()` memuat data.  
2. Periksa `df.shape`, `head()`, dan statistik deskriptif.  
3. Deteksi duplikat & *missing values*.  
4. Visualisasikan fitur kategorikal (barplot) & numerik (histogram).  
5. Analisis korelasi numerik (heatmap).

## 3. Pra‑Pemrosesan Data
- Pisahkan **X** & **y** (`Buys_Computer`).  
- Imputasi: median untuk numerik, modus untuk kategorikal.  
- Encoding kategorikal → `OneHotEncoder`.  
- `train_test_split` 80 : 20 (`stratify=y`, `random_state=42`).  
- Bungkus dengan `ColumnTransformer` + `Pipeline`.

## 4. Pelatihan Model Decision Tree
Parameter disetel via `GridSearchCV` (5‑fold):
```python
param_grid = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [None, 3, 5, 7, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}
```
Lihat kode lengkap di `train_decision_tree.py`.

## 5. Evaluasi
- **Accuracy, Precision, Recall, F1** (`classification_report`).  
- **Confusion Matrix** → `confusion_matrix.png`.  
- **ROC‑AUC** & kurva (jika label biner) → `roc_curve.png`.  
- Cetak `best_params_` & `best_score_`.

## 6. Interpretasi
- Plot *feature importance* Decision Tree.  
- Visualisasi pohon keputusan untuk pemahaman aturan.

## 7. Deployment (Opsional)
Simpan model terlatih:
```bash
joblib.dump(best_estimator, "model_dt.pkl")
```
Buat API (Flask/FastAPI) untuk prediksi online.

## 8. Menjalankan Eksperimen
```bash
# Opsi 1 – jalankan skrip end‑to‑end (disarankan)
python train_decision_tree.py --data data/dataset_buys_comp.csv --model_out model_dt.pkl

# Opsi 2 – jalankan notebook interaktif (jika ingin eksplorasi manual)
# File notebook contoh tersedia di notebooks/01_decision_tree.ipynb
jupyter lab notebooks/01_decision_tree.ipynb
```

## 9. Struktur Notebook Contoh
Notebook `01_decision_tree.ipynb` berisi sel‑sel berikut:
1. **Import & Load Data**  
2. **EDA ringkas** (visualisasi sederhana)  
3. **Pra‑pemrosesan** (kolom numerik & kategorikal)  
4. **Training Decision Tree** (tanpa *grid search*)  
5. **Evaluasi singkat** (accuracy & confusion matrix)  
6. **Simpan Model** (`joblib`)  

Anda dapat membuat notebook baru dengan menyalin kode dari `train_decision_tree.py` ke sel‑sel notebook untuk eksplorasi interaktif.

## 10. Hasil & Kesimpulan
Berikut metrik evaluasi pada data uji (20 % dari 1 000 data):

| Metrik | Nilai |
|--------|-------|
| Akurasi | **0.81** |
| Precision (layak = 1) | 0.91 |
| Recall (layak = 1) | 0.79 |
| F1‑Score (layak = 1) | 0.85 |
| ROC‑AUC | 0.93 |

**Confusion Matrix**  
`[[56 10]  [28 106]]`

### Kesimpulan
- Model Decision Tree dengan parameter terbaik (`criterion='gini'`, `max_depth=None`) mencapai akurasi **81 %** dan ROC‑AUC **0.93**, menandakan performa baik dalam membedakan calon pembeli layak vs tidak layak.
- Precision tinggi (0.91) menunjukkan prediksi "layak" jarang salah, penting untuk meminimalkan kredit macet.
- Recall 0.79 berarti sebagian kecil pemohon layak masih terlewat; dapat ditingkatkan dengan tuning threshold atau model ensembel.
- Feature importance & visualisasi pohon membantu tim kredit memahami faktor keputusan.

