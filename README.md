# LandCover AI — Klasifikasi Tutupan Lahan dari Citra Satelit (Sesi 3)

## 1. Identitas Proyek

| Item | Keterangan |
|---|---|
| Nama Proyek | LandCover AI |
| Sesi | 3 dari 6 |
| Judul Penelitian | Klasifikasi Tutupan Lahan Menggunakan Citra Satelit dengan Algoritma Machine Learning |
| Jenis | Klasifikasi piksel multi-kelas (Random Forest) |

## 2. Latar Belakang

Pemetaan tutupan lahan penting untuk perencanaan tata ruang, pertanian, dan pemantauan
lingkungan. Citra satelit menyediakan informasi spektral yang dapat dipakai untuk
membedakan kelas tutupan lahan. Proyek ini membangun pipeline klasifikasi berbasis fitur
spektral dan indeks NDVI menggunakan Random Forest.

## 3. Tujuan

1. Memuat citra satelit dan memeriksa metadata/georeferensi.
2. Mengekstraksi fitur spektral (dan NDVI bila kanal NIR tersedia).
3. Mengklasifikasi tutupan lahan dengan Random Forest.
4. Mengevaluasi model (accuracy, precision, recall, F1, confusion matrix).
5. Menampilkan peta hasil klasifikasi bila data ter-georeferensi.
6. Mengekspor hasil analisis.

## 4. Teknologi

Python, Rasterio, GeoPandas, NumPy, Pandas, scikit-learn, Streamlit, Plotly, Folium, pytest.

## 5. Sumber Dataset

- **Data nyata:** unggah GeoTIFF multi-kanal (mis. Sentinel-2/Landsat) melalui aplikasi.
  Aplikasi membaca metadata, kanal, resolusi, dan CRS secara otomatis.
- **Mode demo:** dataset sintetik 4 kanal (Blue, Green, Red, NIR) dibuat di dalam kode
  untuk memvalidasi pipeline.

> **Catatan penting:** data demo sintetik **bukan** citra satelit nyata; akurasi pada demo
> hanya menunjukkan bahwa pipeline berfungsi. Untuk hasil penelitian, gunakan citra nyata
> dengan label referensi yang valid.
>
> **Jangan memperlakukan citra RGB sebagai multispektral.** NDVI hanya dihitung bila kanal
> NIR benar-benar tersedia dan teridentifikasi pada deskripsi kanal.

## 6. Struktur Folder

```
Sesi-3-Land-Cover-Classification/
├── app.py
├── train.py
├── requirements.txt
├── config/kelas.py            # Definisi kelas tutupan lahan & warna
├── src/
│   ├── data_loader.py         # Baca raster, NDVI, ekstraksi fitur, data sintetik
│   └── model_ml.py            # Pipeline Random Forest & evaluasi
├── tests/
├── data/
├── models/model_rf.joblib
└── outputs/results/
```

## 7. Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Cara Menjalankan

```bash
python train.py          # latih model demo + laporan
streamlit run app.py     # antarmuka (unggah GeoTIFF / demo)
```

## 9. Cara Melatih Model

`python train.py` melatih Random Forest pada dataset sintetik demo dan menyimpan
`models/model_rf.joblib` + `outputs/results/laporan_pelatihan.json`. Untuk data nyata,
gunakan kembali `src/model_ml.latih_dan_evaluasi()` dengan fitur dari `src/data_loader`.

## 10. Cara Evaluasi

Metrik dihitung pada data uji terpisah (stratified split) untuk mencegah bias. Confusion
matrix dan classification report ditampilkan di aplikasi dan tersimpan di laporan.

## 11. Contoh Hasil (mode demo)

| Metrik | Nilai |
|---|---|
| Accuracy | 0.802 |
| Precision (macro) | 0.801 |
| Recall (macro) | 0.802 |
| F1 (macro) | 0.802 |

## 12. Keterbatasan

- Hasil demo berasal dari data sintetik, bukan citra nyata.
- NDVI memerlukan kanal NIR; pada citra RGB hanya fitur warna dasar yang tersedia.
- Pemisahan data uji acak belum menjamin bebas kebocoran spasial; untuk data nyata
  gunakan pemisahan spasial (mis. blok wilayah).
- Peta Folium memerlukan CRS WGS84 agar akurat.

## 13. Pengujian

```bash
python -m pytest tests -q
```

## 14. Referensi

- Sentinel-2 User Handbook (ESA).
- Breiman, L. (2001). Random Forests.
- Rouse et al. (1974) — NDVI.
