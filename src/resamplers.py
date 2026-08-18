"""
Resampler registry. build_resampler() returns None for "none" (meaning:
rely on class_weight="balanced" on the model instead of resampling).
"""
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from config import RANDOM_STATE


def build_resampler(name: str, trial=None):
    if name == "none":
        return None
    if name == "random_over":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == "random_under":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if name == "smote":
        k = trial.suggest_int("smote_k_neighbors", 3, 10) if trial else 5
        return SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    if name == "adasyn":
        k = trial.suggest_int("adasyn_k_neighbors", 3, 10) if trial else 5
        return ADASYN(n_neighbors=k, random_state=RANDOM_STATE)
    if name == "smotetomek":
        return SMOTETomek(random_state=RANDOM_STATE)
    if name == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown resampler: {name}")
