"""mgc_jax: hypothesengetriebene Genre-Klassifikation auf GTZAN-Features in reinem JAX."""
from .data import ALL_FEATURES, MFCC_FEATURES, SPECTRAL_FEATURES, load_mgc, stratified_folds
from .models import predict, train_classifier
from .stats import bootstrap_ci, mcnemar_test, paired_cv_test

__all__ = [
    "ALL_FEATURES", "MFCC_FEATURES", "SPECTRAL_FEATURES",
    "load_mgc", "stratified_folds", "train_classifier", "predict",
    "paired_cv_test", "bootstrap_ci", "mcnemar_test",
]
