"""Datenzugriff für MGC.csv (GTZAN, 1000 Clips, 10 Genres, 26 Features).

Bewusst ohne pandas: csv + numpy reichen und halten die Abhängigkeiten klein.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPECTRAL_FEATURES = ["rmse", "chroma_stft", "spec_cent", "spec_bw", "rolloff", "zcr"]
MFCC_FEATURES = [f"mfcc_{i}" for i in range(20)]
ALL_FEATURES = SPECTRAL_FEATURES + MFCC_FEATURES

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "MGC.csv"


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray          # (n, d) float32
    y: np.ndarray          # (n,) int64, Klassenindex
    genres: list[str]      # Index -> Genrename
    features: list[str]    # Spaltennamen von X

    @property
    def n_classes(self) -> int:
        return len(self.genres)

    def select(self, names: list[str]) -> "Dataset":
        idx = [self.features.index(n) for n in names]
        return Dataset(self.X[:, idx], self.y, self.genres, list(names))


def load_mgc(path: str | Path = DEFAULT_CSV) -> Dataset:
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    col = {name: i for i, name in enumerate(header)}
    missing = [f for f in ALL_FEATURES if f not in col]
    if missing:
        raise ValueError(f"Spalten fehlen in {path}: {missing}")
    genres = sorted({r[col["genre"]] for r in rows})
    g2i = {g: i for i, g in enumerate(genres)}
    X = np.array([[float(r[col[f]]) for f in ALL_FEATURES] for r in rows], dtype=np.float32)
    y = np.array([g2i[r[col["genre"]]] for r in rows], dtype=np.int64)
    if not np.all(np.isfinite(X)):
        raise ValueError("Nicht-endliche Werte in den Features")
    return Dataset(X, y, genres, list(ALL_FEATURES))


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratifizierte k-fache Aufteilung. Gibt (train_idx, test_idx) je Fold zurück."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    folds = []
    for k in range(n_folds):
        test = np.flatnonzero(fold_of == k)
        train = np.flatnonzero(fold_of != k)
        folds.append((train, test))
    return folds


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-Standardisierung mit Statistik ausschliesslich aus dem Trainingsfold (kein Leakage)."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd
