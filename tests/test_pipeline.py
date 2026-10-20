"""Pengujian alur teknis LandCover AI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.kelas import KELAS_TUTUPAN_LAHAN, NAMA_KELAS  # noqa: E402
from src.data_loader import (  # noqa: E402
    buat_dataset_sintetik,
    ekstraksi_fitur,
    hitung_ndvi,
    tangani_nan,
)
from src.model_ml import buat_pipeline, hitung_metrik, latih_dan_evaluasi  # noqa: E402


def test_synthetic_dataset_shapes() -> None:
    X, y, nama_kanal = buat_dataset_sintetik(n_sampel=400, n_kanal=4)
    assert X.shape[1] == 4
    assert len(np.unique(y)) == len(NAMA_KELAS)
    assert nama_kanal == ["Blue", "Green", "Red", "NIR"]


def test_ndvi_range() -> None:
    arr = np.zeros((4, 5, 5), dtype=np.float32)
    arr[3] = 0.5  # NIR
    arr[2] = 0.1  # Red
    ndvi = hitung_ndvi(arr, idx_red=2, idx_nir=3)
    assert np.nanmin(ndvi) >= -1
    assert np.nanmax(ndvi) <= 1
    assert ndvi[0, 0] > 0


def test_ekstraksi_fitur_with_nir() -> None:
    arr = np.random.rand(4, 8, 8).astype(np.float32)
    fitur, nama = ekstraksi_fitur(arr, ["Blue", "Green", "Red", "NIR"])
    assert fitur.shape == (64, 5)  # 4 band + NDVI
    assert "NDVI" in nama


def test_ekstraksi_fitur_without_nir() -> None:
    arr = np.random.rand(3, 8, 8).astype(np.float32)
    fitur, nama = ekstraksi_fitur(arr, ["Blue", "Green", "Red"])
    assert fitur.shape == (64, 3)
    assert "NDVI" not in nama


def test_tangani_nan_median() -> None:
    fitur = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]])
    hasil = tangani_nan(fitur, "median")
    assert not np.isnan(hasil).any()


def test_hitung_metrik() -> None:
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 2]
    metrik = hitung_metrik(y_true, y_pred)
    assert 0.0 <= metrik["accuracy"] <= 1.0
    assert "confusion_matrix" in metrik


def test_train_end_to_end_synthetic() -> None:
    X, y, _ = buat_dataset_sintetik(n_sampel=400, n_kanal=4)
    X = tangani_nan(X, "median")
    hasil = latih_dan_evaluasi(X, y, n_estimators=30, max_depth=8)
    assert hasil["metrik"]["accuracy"] > 0.6
    assert hasil["model"].n_features_in_ == 4


def test_pipeline_predict_shape() -> None:
    X, y, _ = buat_dataset_sintetik(n_sampel=200, n_kanal=4)
    pipe = buat_pipeline(n_estimators=10)
    pipe.fit(X, y)
    pred = pipe.predict(X[:5])
    assert len(pred) == 5
