"""LandCover AI - Antarmuka Streamlit (Sesi 3).

Fitur:
- Muat citra satelit (GeoTIFF) dan tampilkan metadata/georeferensi.
- Ekstraksi fitur spektral + NDVI.
- Latih Random Forest & tampilkan evaluasi + confusion matrix.
- Peta hasil klasifikasi (Folium) bila georeferensi valid.
- Mode demo sintetik yang diberi label jelas.
- Ekspor hasil analisis.

Jalankan:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.kelas import NAMA_KELAS, WARNA_KELAS  # noqa: E402
from src.data_loader import (  # noqa: E402
    baca_info_raster,
    baca_raster,
    buat_dataset_sintetik,
    ekstraksi_fitur,
    hitung_ndvi,
    tangani_nan,
)
from src.model_ml import latih_dan_evaluasi, muat_model, simpan_model  # noqa: E402

MODEL_PATH = ROOT / "models" / "model_rf.joblib"
st.set_page_config(page_title="LandCover AI", page_icon="🛰️", layout="wide")


def _tampilkan_metadata(info: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lebar × Tinggi", f"{info['lebar']} × {info['tinggi']}")
    c2.metric("Jumlah Kanal", info["jumlah_kanal"])
    c3.metric("CRS", info["crs"] or "tidak ada")
    c4.metric("Tipe Data", info["tipe_data"])
    st.json(info)


def _render_peta(nilai_kelas: np.ndarray, info: dict) -> None:
    """Menampilkan peta hasil klasifikasi bila georeferensi valid."""
    try:
        import folium
        from streamlit_folium import st_folium  # type: ignore
    except ImportError:
        st.info(
            "Untuk peta interaktif, pasang `folium` dan `streamlit-folium`. "
            "Menampilkan preview raster sebagai gantinya."
        )
        st.image(_raster_ke_rgb(nilai_kelas), caption="Preview klasifikasi (tanpa georeferensi)")
        return

    bounds = info.get("bounds")
    if not bounds or not info.get("crs"):
        st.warning("Raster tidak memiliki georeferensi valid. Peta tidak dapat dibuat.")
        return
    minx, miny, maxx, maxy = bounds
    if info["crs"] not in ("EPSG:4326", "OGC:CRS84") and "4326" not in str(info["crs"]):
        st.warning(
            f"CRS {info['crs']} bukan WGS84. Peta mungkin tidak akurat tanpa reproyeksi."
        )
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
    folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], color="#333", fill=False).add_to(m)
    st_folium(m, use_container_width=True, height=420)


def _raster_ke_rgb(labels: np.ndarray) -> np.ndarray:
    """Ubah array label kelas menjadi citra RGB berwarna."""
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for kelas, warna in WARNA_KELAS.items():
        c = tuple(int(warna[i : i + 2], 16) for i in (1, 3, 5))
        rgb[labels == kelas] = c
    return rgb


def render_demo() -> None:
    st.subheader("Mode Demo (Dataset Sintetik)")
    st.warning(
        "Dataset sintetik ini BUKAN citra satelit nyata. Digunakan hanya untuk memvalidasi "
        "pipeline teknis. Untuk hasil penelitian, unggah GeoTIFF nyata beserta label."
    )
    X, y, nama_kanal = buat_dataset_sintetik(n_sampel=1600, n_kanal=4)
    X = tangani_nan(X, "median")
    st.write(f"Shape: {X.shape}, fitur: {nama_kanal}")
    st.dataframe(pd.DataFrame(X[:10], columns=nama_kanal), use_container_width=True)

    if st.button("Latih Random Forest (Demo)", type="primary"):
        with st.spinner("Melatih model..."):
            hasil = latih_dan_evaluasi(X, y, n_estimators=200, max_depth=16)
        metrik = hasil["metrik"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrik['accuracy']:.3f}")
        c2.metric("Precision", f"{metrik['precision_macro']:.3f}")
        c3.metric("Recall", f"{metrik['recall_macro']:.3f}")
        c4.metric("F1 (macro)", f"{metrik['f1_macro']:.3f}")
        _tampilkan_confusion(metrik["confusion_matrix"], metrik["labels"])


def _tampilkan_confusion(mat: list, labels: list) -> None:
    nama = [NAMA_KELAS.get(int(l), str(l)) for l in labels]
    fig = px.imshow(
        np.array(mat),
        text_auto=True,
        x=nama,
        y=nama,
        labels={"x": "Prediksi", "y": "Aktual"},
        color_continuous_scale="Greens",
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_upload() -> None:
    st.subheader("Analisis Citra Satelit (GeoTIFF)")
    raster_file = st.file_uploader("Unggah citra satelit (GeoTIFF)", type=["tif", "tiff"])
    if not raster_file:
        st.info("Unggah file GeoTIFF untuk melihat metadata dan melakukan klasifikasi.")
        return

    tmp = ROOT / "outputs" / "_upload.tif"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raster_file.getbuffer())

    info = baca_info_raster(tmp)
    _tampilkan_metadata(info.sebagai_dict())

    deskripsi = [d.upper() for d in info.deskripsi_kanal]
    st.caption(f"Deskripsi kanal: {info.deskripsi_kanal}")

    try:
        arr = baca_raster(tmp)
    except Exception as exc:  # pragma: no cover
        st.error(f"Gagal membaca raster: {exc}")
        return

    st.markdown("**Pratinjau kanal**")
    n_preview = min(arr.shape[0], 4)
    cols = st.columns(n_preview)
    for i in range(n_preview):
        band = arr[i]
        band_norm = (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-8)
        cols[i].image(band_norm, caption=info.deskripsi_kanal[i], clamp=True, use_container_width=True)

    if "NIR" in deskripsi and "RED" in deskripsi:
        i_red = deskripsi.index("RED")
        i_nir = deskripsi.index("NIR")
        ndvi = hitung_ndvi(arr, i_red, i_nir)
        st.markdown("**NDVI** (hijau = vegetasi lebat)")
        st.image((ndvi + 1) / 2, caption="NDVI", clamp=True, use_container_width=True)
    else:
        st.warning(
            "Kanal NIR tidak terdeteksi. NDVI tidak dihitung. "
            "Pastikan deskripsi kanal berisi 'Red' dan 'NIR'."
        )

    if st.button("Klasifikasi Citra Ini", type="primary"):
        if not MODEL_PATH.exists():
            st.error("Model belum tersedia. Latih dulu di tab Mode Demo atau `python train.py`.")
            return
        model = muat_model(str(MODEL_PATH))
        fitur, nama_fitur = ekstraksi_fitur(arr, info.deskripsi_kanal)
        fitur = tangani_nan(fitur, "median")
        if fitur.shape[1] != model.n_features_in_:
            st.error(
                f"Jumlah fitur citra ({fitur.shape[1]}) tidak cocok dengan model "
                f"({model.n_features_in_}). Model demo dilatih dengan 4 kanal."
            )
            return
        pred = model.predict(fitur).reshape(arr.shape[1], arr.shape[2])
        rgb = _raster_ke_rgb(pred)
        st.markdown("**Peta hasil klasifikasi**")
        st.image(rgb, caption="Klasifikasi tutupan lahan", use_container_width=True)
        _render_peta(pred, info.sebagai_dict())

        dist = pd.Series(pred.ravel()).value_counts().sort_index()
        df_dist = pd.DataFrame(
            {"kelas": [NAMA_KELAS.get(int(k), str(k)) for k in dist.index], "jumlah": dist.values}
        )
        fig = px.bar(df_dist, x="kelas", y="jumlah", color="kelas", title="Distribusi kelas hasil")
        st.plotly_chart(fig, use_container_width=True)

        out = io.BytesIO()
        np.save(out, pred)
        st.download_button(
            "Unduh Hasil Klasifikasi (.npy)",
            data=out.getvalue(),
            file_name="hasil_klasifikasi.npy",
            mime="application/octet-stream",
        )


def render_laporan() -> None:
    path = ROOT / "outputs" / "results" / "laporan_pelatihan.json"
    if not path.exists():
        return
    st.subheader("Laporan Pelatihan Terakhir")
    st.json(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    st.title("🛰️ LandCover AI")
    st.caption(
        "Klasifikasi Tutupan Lahan dari Citra Satelit - Sesi 3 "
        "(Random Forest, fitur spektral & NDVI)"
    )
    tab1, tab2, tab3 = st.tabs(["Analisis Citra", "Mode Demo", "Laporan"])
    with tab1:
        render_upload()
    with tab2:
        render_demo()
    with tab3:
        render_laporan()


if __name__ == "__main__":
    main()
