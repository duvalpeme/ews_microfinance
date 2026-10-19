"""Étape 8 : explicabilité SHAP (globale, par client, par agence).

Usage, depuis la racine du projet :
    python -m src.explain.shap_analysis
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as cfg
from src.features.build_features import CATEGORICAL, NUMERIC

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
FIG_DIR = Path("docs/figures")
SCORE_DIR = Path("outputs/scores")
FEATURES = NUMERIC + CATEGORICAL

# Phrases lisibles pour l'agent de terrain
REASONS = {
    "dpd": lambda v: f"retard actuel de {v:.0f} jours",
    "n_unpaid_due": lambda v: f"{v:.0f} échéance(s) impayée(s)",
    "n_late_total": lambda v: f"{v:.0f} retards depuis le décaissement",
    "late_share": lambda v: f"{v:.0%} des échéances payées en retard",
    "max_delay_all": lambda v: f"retard maximal historique de {v:.0f} jours",
    "last_inst_delay": lambda v: f"dernière échéance avec {v:.0f} jours de retard",
    "avg_delay_last3": lambda v: f"retard moyen de {v:.0f} jours sur les 3 dernières échéances",
    "max_delay_last3": lambda v: f"retard maximal de {v:.0f} jours sur les 3 dernières échéances",
    "n_late_last3": lambda v: f"{v:.0f} retard(s) sur les 3 dernières échéances",
    "delay_trend": lambda v: "retards qui s'aggravent" if v > 0 else "évolution des retards",
    "days_since_last_payment": lambda v: f"dernier paiement il y a {v:.0f} jours",
    "pct_repaid": lambda v: f"crédit remboursé à {v:.0%}",
    "installments_remaining": lambda v: f"{v:.0f} échéance(s) restante(s)",
    "loan_age_months": lambda v: f"crédit décaissé il y a {v:.0f} mois",
    "amount": lambda v: f"montant du crédit de {v:,.0f} XAF".replace(",", " "),
    "term_months": lambda v: f"durée du crédit de {v:.0f} mois",
    "interest_rate": lambda v: f"taux d'intérêt de {v:.1%}",
    "cycle_number": lambda v: f"crédit de cycle {v:.0f}",
    "monthly_installment": lambda v: f"échéance de {v:,.0f} XAF".replace(",", " "),
    "installment_income_ratio": lambda v: f"échéance représentant {v:.0%} du revenu",
    "outstanding_income_ratio": lambda v: f"encours égal à {v:.1f} mois de revenu",
    "age": lambda v: f"âge de {v:.0f} ans",
    "client_tenure_months": lambda v: f"client depuis {v:.0f} mois",
    "declared_monthly_income": lambda v: f"revenu déclaré de {v:,.0f} XAF".replace(",", " "),
    "month_of_year": lambda v: "période saisonnière sensible",
    "officer_par30": lambda v: f"PAR30 du portefeuille de l'agent à {v:.1%}",
    "officer_par30_chg3": lambda v: "PAR30 de l'agent en hausse sur 3 mois",
    "branch_par30": lambda v: f"PAR30 de l'agence à {v:.1%}",
    "branch_par30_chg3": lambda v: "PAR30 de l'agence en hausse sur 3 mois",
    "sector_par30": lambda v: f"PAR30 du secteur d'activité à {v:.1%}",
    "sector_par30_chg3": lambda v: "PAR30 du secteur en hausse sur 3 mois",
    "product": lambda v: f"produit : {v}",
    "collateral_type": lambda v: f"garantie : {v}",
    "sector": lambda v: f"secteur : {v}",
}


def describe(feature, value):
    if pd.isna(value):
        return feature
    fn = REASONS.get(feature)
    return fn(value) if fn else f"{feature} = {value}"


def prepare(df, cats):
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def get_shap(explainer, X):
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_DIR / "lgbm_par30.joblib")
    gbm, iso, cats = bundle["model"], bundle["calibrator"], bundle["categories"]
    explainer = shap.TreeExplainer(gbm)

    df = pd.read_parquet(PROC_DIR / "features.parquet")
    labeled = df[df["target"].notna()]
    test_start = int(labeled["month_index"].max()) - cfg.N_TEST_MONTHS + 1
    test = labeled[labeled["month_index"] >= test_start]
    sample = test.sample(min(5000, len(test)), random_state=cfg.SEED)

    # ---------- 1. Explicabilité globale ----------
    X_s = prepare(sample, cats)
    sv_s = get_shap(explainer, X_s)
    mean_abs = pd.Series(np.abs(sv_s).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    top = mean_abs.head(15)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color="#1f4e79")
    plt.xlabel("Impact moyen sur le score (|SHAP|, log-odds)")
    plt.title("Facteurs de risque : importance globale")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_importance.png", dpi=150)
    plt.close()

    X_plot = X_s.copy()
    for c in CATEGORICAL:
        X_plot[c] = X_plot[c].cat.codes
    plt.figure()
    shap.summary_plot(sv_s, X_plot, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=== Top 12 variables (impact moyen |SHAP|) ===")
    print(mean_abs.head(12).to_string(float_format=lambda x: f"{x:.3f}"))

    # ---------- 2. Raisons par client (dernier mois) ----------
    latest = df[df["month_index"] == df["month_index"].max()].copy().reset_index(drop=True)
    X_l = prepare(latest, cats)
    sv_l = get_shap(explainer, X_l)
    latest["risk_score"] = iso.predict(gbm.predict_proba(X_l)[:, 1])

    for k in (1, 2, 3):
        latest[f"reason_{k}_feature"] = ""
        latest[f"reason_{k}"] = ""
    for i in range(len(latest)):
        order = np.argsort(-sv_l[i])[:3]
        for k, j in enumerate(order, start=1):
            if sv_l[i, j] <= 0:
                break
            feat = FEATURES[j]
            latest.at[i, f"reason_{k}_feature"] = feat
            latest.at[i, f"reason_{k}"] = describe(feat, latest.at[i, feat])

    keep = ["loan_id", "client_id", "officer_id", "branch_id", "snapshot_date",
            "outstanding_est", "risk_score",
            "reason_1_feature", "reason_1", "reason_2_feature", "reason_2",
            "reason_3_feature", "reason_3"]
    reasons = latest[keep].sort_values("risk_score", ascending=False)
    reasons.to_parquet(PROC_DIR / "client_reasons.parquet", index=False)
    reasons.to_csv(SCORE_DIR / "client_reasons.csv", index=False, encoding="utf-8-sig")

    print(f"\nCredits scores au dernier mois : {len(reasons):,}")
    print(f"Score moyen : {reasons['risk_score'].mean():.3f} | "
          f"score max : {reasons['risk_score'].max():.3f}")
    print("\n=== 5 credits les plus a risque : raisons ===")
    for r in reasons.head(5).itertuples(index=False):
        print(f"\n{r.client_id} | {r.branch_id} | {r.officer_id} | score {r.risk_score:.1%}")
        for txt in (r.reason_1, r.reason_2, r.reason_3):
            if txt:
                print(f"   - {txt}")

    # ---------- 3. Facteurs par agence ----------
    top6 = list(mean_abs.head(6).index)
    by_branch = (pd.DataFrame(sv_s, columns=FEATURES)
                 .groupby(sample["branch_id"].to_numpy())[top6].mean())
    by_branch.to_csv(SCORE_DIR / "branch_drivers.csv", encoding="utf-8-sig")
    print("\n=== SHAP moyen par agence (top 6 variables, test) ===")
    print(by_branch.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\nFigures : docs/figures/shap_global_importance.png, shap_beeswarm.png")


if __name__ == "__main__":
    main()