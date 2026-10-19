"""Étape 7 : entraînement et évaluation (baseline logistique + LightGBM).

Usage, depuis la racine du projet :
    python -m src.models.train
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg
from src.features.build_features import CATEGORICAL, NUMERIC

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
FEATURES = NUMERIC + CATEGORICAL


def temporal_split(df):
    last = int(df["month_index"].max())
    test_start = last - cfg.N_TEST_MONTHS + 1
    val_end = test_start - 2
    val_start = val_end - cfg.N_TEST_MONTHS + 1
    train_end = val_start - 2
    m = df["month_index"]
    return (df[m <= train_end].copy(),
            df[(m >= val_start) & (m <= val_end)].copy(),
            df[m >= test_start].copy(),
            (train_end, val_start, val_end, test_start, last))


def top_k_metrics(y, p, frac=0.10):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    captured = y[idx].sum()
    recall = captured / y.sum()
    precision = captured / k
    return recall, precision, precision / y.mean()


def evaluate(name, y, p):
    recall, precision, lift = top_k_metrics(y, p)
    res = {
        "modele": name,
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Recall@10%": recall,
        "Precision@10%": precision,
        "Lift@10%": lift,
    }
    return res


def prepare_lgb(df, cats):
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    train, val, test, bounds = temporal_split(df)
    train_end, val_start, val_end, test_start, last = bounds
    print(f"Entrainement : mois 0-{train_end}  ({len(train):,} lignes, taux {train['target'].mean():.2%})")
    print(f"Validation   : mois {val_start}-{val_end} ({len(val):,} lignes, taux {val['target'].mean():.2%})")
    print(f"Test         : mois {test_start}-{last} ({len(test):,} lignes, taux {test['target'].mean():.2%})")

    y_tr, y_va, y_te = (d["target"].to_numpy() for d in (train, val, test))

    # ---- Baseline : régression logistique ----
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    logit.fit(train[FEATURES], y_tr)
    p_logit = logit.predict_proba(test[FEATURES])[:, 1]

    # ---- LightGBM ----
    cats = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL}
    gbm = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=31,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, random_state=cfg.SEED, verbose=-1,
    )
    gbm.fit(
        prepare_lgb(train, cats), y_tr,
        eval_set=[(prepare_lgb(val, cats), y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"\nLightGBM : {gbm.best_iteration_} arbres retenus")

    p_val_raw = gbm.predict_proba(prepare_lgb(val, cats))[:, 1]
    p_te_raw = gbm.predict_proba(prepare_lgb(test, cats))[:, 1]

    # ---- Calibration isotonique (ajustée sur la validation uniquement) ----
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val_raw, y_va)
    p_te_cal = iso.predict(p_te_raw)

    # ---- Résultats sur le test ----
    results = pd.DataFrame([
        evaluate("Logistique", y_te, p_logit),
        evaluate("LightGBM brut", y_te, p_te_raw),
        evaluate("LightGBM calibre", y_te, p_te_cal),
    ]).set_index("modele")
    print("\n=== Performance sur le TEST ===")
    print(results.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nTaux de base du test : {y_te.mean():.3f}  (un tirage aleatoire donne Lift = 1.0)")

    # ---- Calibration par décile ----
    deciles = pd.qcut(p_te_cal, 10, duplicates="drop")
    cal = (pd.DataFrame({"pred": p_te_cal, "reel": y_te})
           .groupby(deciles, observed=True).agg(pred=("pred", "mean"), reel=("reel", "mean"), n=("reel", "size")))
    print("\n=== Calibration par decile (LightGBM calibre) ===")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- Importance des variables (gain) ----
    imp = pd.Series(gbm.booster_.feature_importance(importance_type="gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print("\n=== Top 10 variables (part du gain) ===")
    print(imp.head(10).to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- Contrôle agrégé par agence (risque pondéré par l'encours) ----
    out = test[["loan_id", "client_id", "officer_id", "branch_id", "snapshot_date",
                "outstanding_est", "target"]].copy()
    out["p_raw"] = p_te_raw
    out["p_cal"] = p_te_cal
    w = out["outstanding_est"]
    by_branch = pd.DataFrame({
        "reel": (out["target"] * w).groupby(out["branch_id"]).sum() / w.groupby(out["branch_id"]).sum(),
        "predit": (out["p_cal"] * w).groupby(out["branch_id"]).sum() / w.groupby(out["branch_id"]).sum(),
    })
    print("\n=== Risque pondere par l'encours, par agence (test) ===")
    print(by_branch.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"Correlation agences : {by_branch['reel'].corr(by_branch['predit']):.2f}")

    # ---- Sauvegardes ----
    joblib.dump({"model": gbm, "calibrator": iso, "features": FEATURES,
                 "categories": cats, "logit": logit}, MODEL_DIR / "lgbm_par30.joblib")
    out.to_parquet(PROC_DIR / "test_predictions.parquet", index=False)
    results.to_csv(MODEL_DIR / "metrics_test.csv")
    print("\nModele sauvegarde : models/lgbm_par30.joblib")


if __name__ == "__main__":
    main()