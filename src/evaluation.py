import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.config import SEED, N_FOLDS


def calc_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["logloss"] = log_loss(y_true, y_proba)
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def get_cv_splitter(n_folds: int = N_FOLDS, seed: int = SEED) -> StratifiedKFold:
    """Return a StratifiedKFold splitter."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def cross_validate(model_fn, X, y, n_folds: int = N_FOLDS, seed: int = SEED,
                    feature_builder=None):
    """Run cross-validation and return fold-wise and mean metrics.

    Parameters
    ----------
    feature_builder : callable or None
        (X_train_raw, X_val_raw) -> (X_train, X_val).
        If None, X is used as-is (backward compatible).
    """
    cv = get_cv_splitter(n_folds, seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if feature_builder is not None:
            X_train, X_val = feature_builder(X_train, X_val)

        model = model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = calc_metrics(y_val, y_pred, y_proba)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    mean_metrics = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in fold_metrics[0]
        if k != "fold"
    }
    return fold_metrics, mean_metrics


def cross_validate_oof(model_fn, X, y, n_folds: int = N_FOLDS, seed: int = SEED,
                       feature_builder=None):
    """Run cross-validation with out-of-fold predictions.

    Parameters
    ----------
    feature_builder : callable or None
        (X_train_raw, X_val_raw) -> (X_train, X_val).
        If None, X is used as-is (backward compatible).

    Returns
    -------
    fold_metrics : list[dict]
    mean_metrics : dict
    oof_pred : np.ndarray  (shape [n_samples])
    oof_proba : np.ndarray (shape [n_samples])
    models : list  (trained model per fold)
    """
    cv = get_cv_splitter(n_folds, seed)
    fold_metrics = []
    models = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if feature_builder is not None:
            X_train, X_val = feature_builder(X_train, X_val)

        model = model_fn()
        model.fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_val)
        oof_pred[val_idx] = y_pred

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = proba
        else:
            oof_proba[val_idx] = y_pred.astype(float)

        y_proba = oof_proba[val_idx] if hasattr(model, "predict_proba") else None
        metrics = calc_metrics(y_val, y_pred, y_proba)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    mean_metrics = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in fold_metrics[0]
        if k != "fold"
    }
    return fold_metrics, mean_metrics, oof_pred, oof_proba, models
