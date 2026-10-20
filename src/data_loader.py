"""Pemuatan citra satelit dan ekstraksi fitur spektral (Sesi 3).

Modul ini menangani:
- Pembacaan raster (GeoTIFF) via rasterio, termasuk metadata & georeferensi.
- Pemeriksaan kanal, resolusi, NoData, dan nilai tidak valid.
- Ekstraksi fitur spektral (termasuk indeks NDVI bila kanal NIR tersedia).
- Pembuatan dataset sintetik untuk mode demo yang diberi label jelas.

Prinsip: JANGAN memperlakukan citra RGB sebagai multispektral. Indeks vegetasi
hanya dihitung jika kanal NIR benar-benar tersedia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

from config.kelas import KANAL_STANDAR, NAMA_KELAS


@dataclass
class InfoRaster:
    path: str
    lebar: int
    tinggi: int
    jumlah_kanal: int
    crs: str | None
    resolusi: tuple | None
    bounds: tuple | None
    nodata: float | None
    tipe_data: str
    deskripsi_kanal: list = field(default_factory=list)

    def sebagai_dict(self) -> dict:
        return {
            "path": self.path,
            "lebar": self.lebar,
            "tinggi": self.tinggi,
            "jumlah_kanal": self.jumlah_kanal,
            "crs": self.crs,
            "resolusi": self.resolusi,
            "bounds": self.bounds,
            "nodata": self.nodata,
            "tipe_data": self.tipe_data,
            "deskripsi_kanal": self.deskripsi_kanal,
        }


def baca_info_raster(path: str | Path) -> InfoRaster:
    """Membaca metadata dan georeferensi file raster tanpa memuat seluruh data."""
    if rasterio is None:
        raise ImportError("rasterio belum terpasang. Install: pip install rasterio")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster tidak ditemukan: {path}")
    with rasterio.open(path) as src:
        return InfoRaster(
            path=str(path),
            lebar=src.width,
            tinggi=src.height,
            jumlah_kanal=src.count,
            crs=str(src.crs) if src.crs else None,
            resolusi=src.res,
            bounds=tuple(src.bounds),
            nodata=src.nodata,
            tipe_data=src.dtypes[0] if src.dtypes else "unknown",
            deskripsi_kanal=[d or f"Band-{i + 1}" for i, d in enumerate(src.descriptions)],
        )


def baca_raster(path: str | Path, max_piksel: int | None = 200_000) -> np.ndarray:
    """Membaca raster multi-kanal sebagai array (kanal, tinggi, lebar).

    Bila melebihi max_piksel, citra di-sub-sample secara seragam agar hemat memori.
    """
    if rasterio is None:
        raise ImportError("rasterio belum terpasang. Install: pip install rasterio")
    with rasterio.open(path) as src:
        step = 1
        if max_piksel is not None and src.width * src.height > max_piksel:
            step = int(np.ceil(np.sqrt(src.width * src.height / max_piksel)))
        data = src.read(
            out_shape=(src.count, int(src.height / step), int(src.width / step)),
            resampling=rasterio.enums.Resampling.nearest,
        )
        nodata = src.nodata
    arr = data.astype(np.float32)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def hitung_ndvi(arr: np.ndarray, idx_red: int, idx_nir: int) -> np.ndarray:
    """Menghitung NDVI = (NIR - Red) / (NIR + Red)."""
    red = arr[idx_red]
    nir = arr[idx_nir]
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    ndvi[~np.isfinite(ndvi)] = np.nan
    return np.clip(ndvi, -1, 1)


def ekstraksi_fitur(arr: np.ndarray, nama_kanal: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Mengekstraksi fitur spektral dari array (kanal, tinggi, lebar).

    Mengembalikan (fitur, nama_fitur) dengan fitur berbentuk (n_piksel, n_fitur).
    Menambahkan NDVI hanya jika kanal NIR terdeteksi.
    """
    n_kanal = arr.shape[0]
    nama_kanal = nama_kanal or KANAL_STANDAR[:n_kanal]
    fitur_list = [arr[c].ravel() for c in range(n_kanal)]
    nama_fitur = [nama_kanal[c] if c < len(nama_kanal) else f"Band-{c + 1}" for c in range(n_kanal)]

    if "NIR" in nama_kanal and "Red" in nama_kanal:
        i_red = nama_kanal.index("Red")
        i_nir = nama_kanal.index("NIR")
        ndvi = hitung_ndvi(arr, i_red, i_nir).ravel()
        fitur_list.append(ndvi)
        nama_fitur.append("NDVI")

    fitur = np.vstack(fitur_list).T.astype(np.float32)
    return fitur, nama_fitur


def tangani_nan(fitur: np.ndarray, strategi: str = "median") -> np.ndarray:
    """Menangani nilai NaN: 'median' (default), 'nol', atau 'drop' (via boolean mask)."""
    fitur = fitur.copy()
    if strategi == "nol":
        return np.nan_to_num(fitur, nan=0.0)
    if strategi == "drop":
        return fitur
    kolom_median = np.nanmedian(fitur, axis=0)
    inds = np.where(np.isnan(fitur))
    fitur[inds] = np.take(kolom_median, inds[1])
    return np.nan_to_num(fitur, nan=0.0)


def buat_dataset_sintetik(
    n_sampel: int = 1500, n_kanal: int = 4, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Membuat dataset tutupan lahan sintetik untuk MODE DEMO.

    PENTING: data ini BUKAN citra satelit nyata. Distribusi tiap kelas dibuat
    terpisah (vegetasi, lahan terbuka, terbangun, air) agar model dapat
    mempelajarinya, tetapi hasilnya hanya validasi teknis pipeline.
    """
    rng = np.random.default_rng(seed)
    nama_kanal = KANAL_STANDAR[:n_kanal]
    n_kelas = len(NAMA_KELAS)

    pusat = {
        0: {"Blue": 0.06, "Green": 0.20, "Red": 0.08, "NIR": 0.45},   # vegetasi
        1: {"Blue": 0.18, "Green": 0.20, "Red": 0.24, "NIR": 0.26},   # lahan terbuka
        2: {"Blue": 0.20, "Green": 0.21, "Red": 0.26, "NIR": 0.22},   # terbangun
        3: {"Blue": 0.10, "Green": 0.12, "Red": 0.06, "NIR": 0.03},   # air
    }
    spread = {
        0: {"Blue": 0.02, "Green": 0.04, "Red": 0.02, "NIR": 0.08},
        1: {"Blue": 0.05, "Green": 0.05, "Red": 0.06, "NIR": 0.06},
        2: {"Blue": 0.06, "Green": 0.05, "Red": 0.07, "NIR": 0.05},
        3: {"Blue": 0.03, "Green": 0.03, "Red": 0.02, "NIR": 0.02},
    }

    X, y = [], []
    per_kelas = n_sampel // n_kelas
    for kelas in range(n_kelas):
        baris = []
        for kanal in nama_kanal:
            mu = pusat[kelas][kanal]
            sd = spread[kelas][kanal]
            baris.append(rng.normal(mu, sd, per_kelas))
        sampel = np.vstack(baris).T
        X.append(sampel)
        y.append(np.full(per_kelas, kelas))
    X = np.clip(np.vstack(X), 0, 1).astype(np.float32)
    y = np.concatenate(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], nama_kanal
