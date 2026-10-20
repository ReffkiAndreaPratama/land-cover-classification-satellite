"""Pelatihan, evaluasi, dan prediksi klasifikasi tutupan lahan (Sesi 3).

Model utama: Random Forest (scikit-learn). Mendukung pemisahan data latih/uji
dengan stratifikasi, penyimpanan konfigurasi preprocessing, dan evaluasi lengkap.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.kelas import RANDOM_STATE, TEST_SIZE

MODEL_NAME = "RandomForest"


def buat_pipeline(n_estimators: int = 200, max_depth: int | None = 16) -> Pipeline:
    """Membangun pipeline (scaler + Random Forest)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def bagi_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split data latih/uji dengan stratifikasi."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def latih_model(X_train, y_train, n_estimators: int = 200, max_depth: int | None = 16) -> Pipeline:
    """Melatih pipeline Random Forest."""
    pipe = buat_pipeline(n_estimators=n_estimators, max_depth=max_depth)
    pipe.fit(X_train, y_train)
    return pipe


def hitung_metrik(y_true, y_pred) -> dict:
    """Menghitung metrik evaluasi klasifikasi multi-kelas."""
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [int(x) for x in labels],
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def latih_dan_evaluasi(X, y, n_estimators: int = 200, max_depth: int | None = 16) -> dict:
    """Alur lengkap: split -> latih -> evaluasi. Mengembalikan model + metrik."""
    X_train, X_test, y_train, y_test = bagi_data(X, y)
    model = latih_model(X_train, y_train, n_estimators, max_depth)
    y_pred = model.predict(X_test)
    metrik = hitung_metrik(y_test, y_pred)
    metrik["n_train"] = int(len(y_train))
    metrik["n_test"] = int(len(y_test))
    metrik["n_fitur"] = int(X.shape[1])
    return {"model": model, "metrik": metrik, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}


def simpan_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def muat_model(path: str) -> Pipeline:
    return joblib.load(path)
