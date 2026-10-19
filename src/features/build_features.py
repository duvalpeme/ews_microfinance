"""Étape 6 : features sans fuite temporelle.

Toutes les features d'une ligne (crédit, date t) n'utilisent que l'information
connue à t. Une échéance payée après t est traitée comme impayée à t.

Usage, depuis la racine du projet :
    python -m src.features.build_features
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config as cfg
from src.features.build_panel import month_ends

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

NUMERIC = [
    "dpd", "n_unpaid_due", "n_late_total", "late_share", "max_delay_all",
    "last_inst_delay", "avg_delay_last3", "max_delay_last3", "n_late_last3",
    "delay_trend", "days_since_last_payment", "pct_repaid", "installments_remaining",
    "loan_age_months", "amount", "term_months", "interest_rate", "cycle_number",
    "monthly_installment", "installment_income_ratio", "outstanding_income_ratio",
    "age", "client_tenure_months", "declared_monthly_income", "month_of_year",
    "officer_par30", "officer_par30_chg3", "branch_par30", "branch_par30_chg3",
    "sector_par30", "sector_par30_chg3",
]
CATEGORICAL = ["product", "collateral_type", "sector"]
ID_COLS = ["loan_id", "client_id", "officer_id", "branch_id", "snapshot_date",
           "month_index", "outstanding_est", "target"]


def installment_delays(inst, t):
    """Echéances échues à t, avec leur retard connu à t (censuré si impayé)."""
    d = inst[inst["due_date"] <= t].copy()
    d["paid_known"] = d["paid_date"].notna() & (d["paid_date"] <= t)
    end = d["paid_date"].where(d["paid_known"], t)
    d["delay"] = (end - d["due_date"]).dt.days
    d["delay_pos"] = d["delay"].clip(lower=0)
    d["late"] = (d["delay_pos"] > 0).astype(int)
    d = d.sort_values(["loan_id", "installment_no"])
    d["rank_from_last"] = d.groupby("loan_id").cumcount(ascending=False)  # 0 = dernière échéance
    return d


def loan_behaviour_features(d, t):
    g = d.groupby("loan_id")
    out = pd.DataFrame({
        "n_due": g.size(),
        "n_late_total": g["late"].sum(),
        "max_delay_all": g["delay_pos"].max(),
        "last_inst_delay": d[d["rank_from_last"] == 0].set_index("loan_id")["delay_pos"],
    })
    last3 = d[d["rank_from_last"] < 3].groupby("loan_id")
    out["avg_delay_last3"] = last3["delay_pos"].mean()
    out["max_delay_last3"] = last3["delay_pos"].max()
    out["n_late_last3"] = last3["late"].sum()
    prev3 = d[(d["rank_from_last"] >= 3) & (d["rank_from_last"] < 6)].groupby("loan_id")["delay_pos"].mean()
    out["delay_trend"] = (out["avg_delay_last3"] - prev3).fillna(0)
    out["late_share"] = out["n_late_total"] / out["n_due"]
    last_paid = d[d["paid_known"]].groupby("loan_id")["paid_date"].max()
    out["days_since_last_payment"] = (t - last_paid).dt.days
    return out


def main():
    panel = pd.read_parquet(PROC_DIR / "loan_month_panel.parquet")
    loans = pd.read_csv(RAW_DIR / "loans.csv", parse_dates=["disbursement_date"])
    clients = pd.read_csv(RAW_DIR / "clients.csv")
    inst = pd.read_csv(RAW_DIR / "installments.csv", parse_dates=["due_date", "paid_date"])

    loans = loans.merge(
        clients[["client_id", "officer_id", "branch_id", "sector", "age",
                 "client_tenure_months", "declared_monthly_income"]], on="client_id")
    loans["maturity_date"] = [d + pd.DateOffset(months=int(m))
                              for d, m in zip(loans["disbursement_date"], loans["term_months"])]

    ends = month_ends()
    feats = []
    ctx = {"officer_id": [], "branch_id": [], "sector": []}

    for i, t in enumerate(ends):
        d = installment_delays(inst, t)
        lf = loan_behaviour_features(d, t)

        # photo de TOUT le portefeuille actif à t (y compris PAR30+) pour les indicateurs de contexte
        dpd_all = d[~d["paid_known"]].groupby("loan_id")["delay"].max()
        n_paid_all = d[d["paid_known"]].groupby("loan_id").size()
        al = loans[loans["disbursement_date"] <= t].copy()
        al["dpd"] = al["loan_id"].map(dpd_all).fillna(0)
        al["n_paid"] = al["loan_id"].map(n_paid_all).fillna(0)
        al = al[(t < al["maturity_date"]) | (al["dpd"] > 0)]
        al["outstanding"] = al["amount"] * (1 - al["n_paid"] / al["term_months"]).clip(lower=0)

        for level in ctx:
            tot = al.groupby(level)["outstanding"].sum()
            bad = al[al["dpd"] >= cfg.PAR_THRESHOLD_DAYS].groupby(level)["outstanding"].sum()
            par = (bad.reindex(tot.index).fillna(0) / tot).fillna(0).rename("par30")
            c = par.reset_index()
            c["month_index"] = i
            ctx[level].append(c)

        sub = panel[panel["month_index"] == i].merge(lf, left_on="loan_id", right_index=True, how="left")
        feats.append(sub)

    feat = pd.concat(feats, ignore_index=True)

    static = loans[["loan_id", "interest_rate", "cycle_number", "product", "collateral_type",
                    "monthly_installment", "disbursement_date", "sector", "age",
                    "client_tenure_months", "declared_monthly_income"]]
    feat = feat.merge(static, on="loan_id", how="left")

    # features dérivées
    loan_age_days = (feat["snapshot_date"] - feat["disbursement_date"]).dt.days
    feat["loan_age_months"] = loan_age_days / 30.4
    feat["pct_repaid"] = feat["n_paid"] / feat["term_months"]
    feat["installments_remaining"] = (feat["term_months"] - feat["n_paid"]).clip(lower=0)
    feat["installment_income_ratio"] = feat["monthly_installment"] / feat["declared_monthly_income"]
    feat["outstanding_income_ratio"] = feat["outstanding_est"] / feat["declared_monthly_income"]
    feat["month_of_year"] = feat["snapshot_date"].dt.month
    feat["days_since_last_payment"] = feat["days_since_last_payment"].fillna(loan_age_days)
    for col in ["n_late_total", "late_share", "max_delay_all", "last_inst_delay",
                "avg_delay_last3", "max_delay_last3", "n_late_last3"]:
        feat[col] = feat[col].fillna(0)

    # contexte : PAR30 de l'agent, de l'agence, du secteur, et variation sur 3 mois
    for level, name in [("officer_id", "officer"), ("branch_id", "branch"), ("sector", "sector")]:
        c = pd.concat(ctx[level], ignore_index=True).rename(columns={"par30": f"{name}_par30"})
        lag = c.copy()
        lag["month_index"] += 3
        lag = lag.rename(columns={f"{name}_par30": f"{name}_par30_lag3"})
        feat = feat.merge(c, on=[level, "month_index"], how="left")
        feat = feat.merge(lag, on=[level, "month_index"], how="left")
        feat[f"{name}_par30_chg3"] = (feat[f"{name}_par30"] - feat[f"{name}_par30_lag3"]).fillna(0)
        feat = feat.drop(columns=[f"{name}_par30_lag3"])

    out = feat[ID_COLS + NUMERIC + CATEGORICAL].copy()
    out.to_parquet(PROC_DIR / "features.parquet", index=False)

    # ---- contrôles ----
    lab = out[out["target"].notna()].copy()
    lab["target"] = lab["target"].astype(int)
    print(f"Lignes : {len(out):,} (dont {len(lab):,} avec cible), {len(NUMERIC) + len(CATEGORICAL)} features")
    print(f"Taux de cible : {lab['target'].mean():.2%}")

    print("\nTaux de cible selon le nombre de retards sur les 3 dernières échéances :")
    print(lab.groupby("n_late_last3")["target"].agg(["mean", "size"]).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nTaux de cible selon le retard actuel :")
    buckets = pd.cut(lab["dpd"], [-1, 0, 7, 14, 29], labels=["0 j", "1-7 j", "8-14 j", "15-29 j"])
    print(lab.groupby(buckets, observed=True)["target"].agg(["mean", "size"]).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nAUC univarié (distance à 0.5 = pouvoir prédictif individuel) :")
    aucs = {c: roc_auc_score(lab["target"], lab[c].fillna(0)) for c in NUMERIC}
    for c, a in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:12]:
        print(f"  {c:28s} {a:.3f}")


if __name__ == "__main__":
    main()