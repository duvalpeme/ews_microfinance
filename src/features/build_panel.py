"""Étape 5 : panel mensuel (un crédit × un mois) et cible de migration PAR30.

Usage, depuis la racine du projet :
    python -m src.features.build_panel
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def month_ends():
    start = pd.Timestamp(cfg.START_DATE)
    return [start + pd.DateOffset(months=i) + pd.offsets.MonthEnd(0)
            for i in range(cfg.N_MONTHS)]


def state_at(inst, t):
    """Etat de chaque crédit à la date t, avec uniquement l'information connue à t."""
    due = inst[inst["due_date"] <= t]
    unpaid = due[due["paid_date"].isna() | (due["paid_date"] > t)].copy()
    unpaid["dpd"] = (t - unpaid["due_date"]).dt.days
    g = unpaid.groupby("loan_id")["dpd"].agg(["max", "size"])
    g.columns = ["dpd", "n_unpaid_due"]
    n_paid = (inst[inst["paid_date"].notna() & (inst["paid_date"] <= t)]
              .groupby("loan_id").size().rename("n_paid"))
    return g, n_paid


def build_panel():
    loans = pd.read_csv(RAW_DIR / "loans.csv", parse_dates=["disbursement_date"])
    clients = pd.read_csv(RAW_DIR / "clients.csv",
                          usecols=["client_id", "officer_id", "branch_id"])
    inst = pd.read_csv(RAW_DIR / "installments.csv", parse_dates=["due_date", "paid_date"])

    loans = loans.merge(clients, on="client_id")
    loans["maturity_date"] = [d + pd.DateOffset(months=int(m))
                              for d, m in zip(loans["disbursement_date"], loans["term_months"])]

    ends = month_ends()
    states = {t: state_at(inst, t) for t in ends}
    keep = ["loan_id", "client_id", "officer_id", "branch_id", "amount",
            "term_months", "maturity_date"]
    frames = []

    for i, t in enumerate(ends):
        st, n_paid = states[t]
        base = loans.loc[loans["disbursement_date"] <= t, keep].copy()
        base["dpd"] = base["loan_id"].map(st["dpd"]).fillna(0).astype(int)
        base["n_unpaid_due"] = base["loan_id"].map(st["n_unpaid_due"]).fillna(0).astype(int)
        base["n_paid"] = base["loan_id"].map(n_paid).fillna(0).astype(int)

        # crédit actif : pas encore à échéance, ou encore en retard
        active = (t < base["maturity_date"]) | (base["dpd"] > 0)
        # population d'alerte : PAR 0 ou 1-29 jours
        eligible = active & (base["dpd"] < cfg.PAR_THRESHOLD_DAYS)
        base = base[eligible].copy()

        base["snapshot_date"] = t
        base["month_index"] = i
        base["outstanding_est"] = (base["amount"]
                                   * (1 - base["n_paid"] / base["term_months"]).clip(lower=0)).round()

        if i + 1 < len(ends):
            st_next, _ = states[ends[i + 1]]
            dpd_next = base["loan_id"].map(st_next["dpd"]).fillna(0)
            base["target"] = (dpd_next > cfg.PAR_THRESHOLD_DAYS).astype("Int8")
        else:
            base["target"] = pd.array([pd.NA] * len(base), dtype="Int8")

        frames.append(base.drop(columns=["maturity_date"]))

    return pd.concat(frames, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()

    assert (panel["dpd"] < cfg.PAR_THRESHOLD_DAYS).all(), "Fuite : retard >= 30 j dans la population"
    assert panel.duplicated(["loan_id", "snapshot_date"]).sum() == 0, "Doublons credit x mois"

    panel.to_parquet(OUT_DIR / "loan_month_panel.parquet", index=False)

    lab = panel[panel["target"].notna()].copy()
    lab["target"] = lab["target"].astype(int)
    low, high = cfg.TARGET_RATE_RANGE

    print(f"Lignes du panel         : {len(panel):,}")
    print(f"Lignes avec cible       : {len(lab):,}")
    print(f"Credits distincts       : {panel['loan_id'].nunique():,}")
    print(f"Taux de cible global    : {lab['target'].mean():.2%}  (vise {low:.0%}-{high:.0%})")
    print("\nTaux de cible par trimestre :")
    q = lab.groupby(lab["snapshot_date"].dt.to_period("Q"))["target"].agg(["mean", "size"])
    print(q.rename(columns={"mean": "taux", "size": "n"}).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTaux de cible par agence :")
    print(lab.groupby("branch_id")["target"].mean().round(3).to_string())


if __name__ == "__main__":
    main()