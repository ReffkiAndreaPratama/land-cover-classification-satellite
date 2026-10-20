"""Konfigurasi LandCover AI (Sesi 3)."""

from __future__ import annotations

KELAS_TUTUPAN_LAHAN = {
    0: {"nama": "Vegetasi", "warna": "#2e7d32"},
    1: {"nama": "Lahan Terbuka", "warna": "#c9a227"},
    2: {"nama": "Kawasan Terbangun", "warna": "#b23b3b"},
    3: {"nama": "Badan Air", "warna": "#2b6cb0"},
}

NAMA_KELAS = {k: v["nama"] for k, v in KELAS_TUTUPAN_LAHAN.items()}
WARNA_KELAS = {k: v["warna"] for k, v in KELAS_TUTUPAN_LAHAN.items()}

TARGET_CRS = "EPSG:4326"
RANDOM_STATE = 42
TEST_SIZE = 0.3

KANAL_STANDAR = ["Blue", "Green", "Red", "NIR"]
