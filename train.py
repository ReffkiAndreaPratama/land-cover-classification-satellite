"""Entry point pelatihan LandCover AI (Sesi 3).

Alur:
1. Muat citra satelit (GeoTIFF) + label, atau pakai dataset sintetik (mode demo).
2. Ekstraksi fitur spektral (termasuk NDVI bila kanal NIR tersedia).
3. Latih Random Forest.
4. Evaluasi & simpan model + laporan.

Jalankan dari folder proyek:
    python train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.kelas import NAMA_KELAS  # noqa: E402
from src.data_loader import buat_dataset_sintetik, ekstraksi_fitur, tangani_nan  # noqa: E402
from src.model_ml import latih_dan_evaluasi, simpan_model  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "outputs" / "results"


def main() -> int:
    print("=" * 70)
    print("LandCover AI - Pelatihan Klasifikasi Tutupan Lahan (Sesi 3)")
    print("=" * 70)

    print("[1/4] Menyiapkan data (mode demo sintetik)...")
    X, y, nama_kanal = buat_dataset_sintetik(n_sampel=1600, n_kanal=4)
    X = tangani_nan(X, "median")
    print(f"      Data sintetik: {X.shape[0]} piksel sampel, {X.shape[1]} fitur {nama_kanal}")
    print("      CATATAN: data ini BUKAN citra satelit nyata (hanya validasi pipeline).")

    print("[2/4] Melatih Random Forest...")
    hasil = latih_dan_evaluasi(X, y, n_estimators=200, max_depth=16)
    metrik = hasil["metrik"]

    print("[3/4] Evaluasi:")
    print(f"      - Accuracy      : {metrik['accuracy']:.4f}")
    print(f"      - Precision(mac): {metrik['precision_macro']:.4f}")
    print(f"      - Recall (mac)  : {metrik['recall_macro']:.4f}")
    print(f"      - F1 (macro)    : {metrik['f1_macro']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    simpan_model(hasil["model"], str(MODELS_DIR / "model_rf.joblib"))

    laporan = {
        "mode": "demo_sintetik",
        "n_fitur": metrik["n_fitur"],
        "nama_kanal": nama_kanal,
        "n_train": metrik["n_train"],
        "n_test": metrik["n_test"],
        "metrik": {k: v for k, v in metrik.items() if k != "classification_report"},
        "classification_report": metrik["classification_report"],
        "kelas": {str(k): v for k, v in NAMA_KELAS.items()},
    }
    with open(RESULTS_DIR / "laporan_pelatihan.json", "w", encoding="utf-8") as fh:
        json.dump(laporan, fh, indent=2, ensure_ascii=False)

    print("[4/4] Tersimpan:")
    print(f"      - model : {MODELS_DIR / 'model_rf.joblib'}")
    print(f"      - laporan: {RESULTS_DIR / 'laporan_pelatihan.json'}")
    print("\nJalankan aplikasi: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
