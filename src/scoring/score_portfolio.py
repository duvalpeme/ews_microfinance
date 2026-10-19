"""Étape 9 : scoring complet et agrégation client / agent / agence.

Usage, depuis la racine du projet :
    python -m src.scoring.score_portfolio
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
SCORE_DIR = Path("outputs/scores")
SAMPLE_DIR = Path("docs/sample_outputs")

RED_TOP = 0.05      # part du portefeuille classée Rouge
ORANGE_TOP = 0.15   # Rouge + Orange (cumulé)


def add_alert_levels(df, score_col="risk_score", tie_col="outstanding_est"):
    df = df.sort_values([score_col, tie_col], ascending=False).reset_index(drop=True)
    pct = (np.arange(len(df)) + 1) / len(df)
    df["alert_level"] = np.select([pct <= RED_TOP, pct <= ORANGE_TOP], ["Rouge", "Orange"], default="Vert")
    return df


def aggregate(df, keys):
    d = df.assign(
        exp_amt=df["risk_score"] * df["outstanding_est"],
        is_red=(df["alert_level"] == "Rouge").astype(int),
        is_orange=(df["alert_level"] == "Orange").astype(int),
    )
    out = d.groupby(keys).agg(
        n_loans=("loan_id", "size"),
        n_clients=("client_id", "nunique"),
        outstanding=("outstanding_est", "sum"),
        expected_amount_at_risk=("exp_amt", "sum"),
        avg_score=("risk_score", "mean"),
        n_red=("is_red", "sum"),
        n_orange=("is_orange", "sum"),
    ).reset_index()
    out["expected_migration_rate"] = out["expected_amount_at_risk"] / out["outstanding"]
    return out


def history(pred, keys):
    g = pred.groupby(keys).agg(
        n_loans=("loan_id", "size"),
        outstanding=("outstanding_est", "sum"),
        predicted_amount=("w_p", "sum"),
        actual_amount=("w_y", "sum"),
    ).reset_index()
    g["predicted_rate"] = g["predicted_amount"] / g["outstanding"]
    g["actual_rate"] = g["actual_amount"] / g["outstanding"]
    return g


def main():
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    branches = pd.read_csv(RAW_DIR / "branches.csv")
    officers = pd.read_csv(RAW_DIR / "loan_officers.csv")

    # ---------- Scores clients (dernier mois) ----------
    cl = pd.read_parquet(PROC_DIR / "client_reasons.parquet")
    feats = pd.read_parquet(PROC_DIR / "features.parquet")
    latest = feats[feats["month_index"] == feats["month_index"].max()]
    ctx = latest[["loan_id", "sector", "product", "dpd", "n_late_last3",
                  "officer_par30", "branch_par30"]]
    cl = cl.merge(ctx, on="loan_id", how="left")
    cl = add_alert_levels(cl)
    cl["priority_rank"] = (cl.groupby("officer_id")["risk_score"]
                           .rank(ascending=False, method="first").astype(int))
    cl["snapshot_date"] = pd.to_datetime(cl["snapshot_date"]).dt.date

    # ---------- Agrégations ----------
    off = aggregate(cl, ["officer_id", "branch_id"])
    off = off.merge(officers[["officer_id", "tenure_months"]], on="officer_id", how="left")
    off = off.merge(cl.groupby("officer_id")["officer_par30"].first().rename("current_par30").reset_index(),
                    on="officer_id", how="left")

    br = aggregate(cl, ["branch_id"]).merge(branches[["branch_id", "name", "region", "setting"]],
                                            on="branch_id", how="left")
    br = br.merge(cl.groupby("branch_id")["branch_par30"].first().rename("current_par30").reset_index(),
                  on="branch_id", how="left")

    network_rate = cl["risk_score"].mul(cl["outstanding_est"]).sum() / cl["outstanding_est"].sum()
    for t in (off, br):
        t["risk_vs_network"] = t["expected_migration_rate"] / network_rate
        t["risk_rank"] = t["expected_migration_rate"].rank(ascending=False, method="first").astype(int)
    off["risk_rank_in_branch"] = (off.groupby("branch_id")["expected_migration_rate"]
                                  .rank(ascending=False, method="first").astype(int))
    off = off.sort_values("risk_rank")
    br = br.sort_values("risk_rank")

    # ---------- Contrôles de cohérence entre les niveaux ----------
    total = (cl["risk_score"] * cl["outstanding_est"]).sum()
    assert np.isclose(off["expected_amount_at_risk"].sum(), total), "Agents != clients"
    assert np.isclose(br["expected_amount_at_risk"].sum(), total), "Agences != clients"
    assert off["n_loans"].sum() == len(cl) == br["n_loans"].sum(), "Nombre de credits incoherent"
    print("Controles de coherence client / agent / agence : OK")

    # ---------- Historique prédit vs réel (période de test, hors échantillon) ----------
    pred = pd.read_parquet(PROC_DIR / "test_predictions.parquet")
    pred["target"] = pred["target"].astype(float)
    pred["w_p"] = pred["p_cal"] * pred["outstanding_est"]
    pred["w_y"] = pred["target"] * pred["outstanding_est"]
    hb = history(pred, ["branch_id", "snapshot_date"]).merge(
        branches[["branch_id", "name"]], on="branch_id", how="left")
    ho = history(pred, ["officer_id", "branch_id", "snapshot_date"])
    hb["snapshot_date"] = pd.to_datetime(hb["snapshot_date"]).dt.date
    ho["snapshot_date"] = pd.to_datetime(ho["snapshot_date"]).dt.date

    # ---------- Validation des seuils d'alerte sur le test ----------
    p = pred.sort_values(["snapshot_date", "p_cal", "outstanding_est"], ascending=[True, False, False]).copy()
    p["pct"] = (p.groupby("snapshot_date").cumcount() + 1) / p.groupby("snapshot_date")["p_cal"].transform("size")
    tot_y = p["target"].sum()
    cap_red = p.loc[p["pct"] <= RED_TOP, "target"].sum() / tot_y
    cap_ro = p.loc[p["pct"] <= ORANGE_TOP, "target"].sum() / tot_y

    # ---------- Sauvegardes ----------
    cl_cols = ["loan_id", "client_id", "officer_id", "branch_id", "snapshot_date", "sector", "product",
               "outstanding_est", "dpd", "n_late_last3", "risk_score", "alert_level", "priority_rank",
               "reason_1", "reason_2", "reason_3"]
    cl = cl[cl_cols]
    files = {
        "scores_clients.csv": cl, "scores_officers.csv": off, "scores_branches.csv": br,
        "history_branch_month.csv": hb, "history_officer_month.csv": ho,
    }
    for name, d in files.items():
        d.to_csv(SCORE_DIR / name, index=False, encoding="utf-8-sig")

    # extraits publics pour le dépôt GitHub (données synthétiques)
    br.to_csv(SAMPLE_DIR / "scores_branches.csv", index=False, encoding="utf-8-sig")
    off.to_csv(SAMPLE_DIR / "scores_officers.csv", index=False, encoding="utf-8-sig")
    cl.head(100).to_csv(SAMPLE_DIR / "scores_clients_top100.csv", index=False, encoding="utf-8-sig")

    # ---------- Résumé ----------
    print(f"\nCredits scores : {len(cl):,} | encours estime : {cl['outstanding_est'].sum():,.0f} XAF")
    print(f"Risque de migration attendu (reseau) : {network_rate:.2%}")
    print("\nNiveaux d'alerte :")
    summ = cl.groupby("alert_level").agg(n=("loan_id", "size"), score_moyen=("risk_score", "mean"),
                                         encours=("outstanding_est", "sum")).reindex(["Rouge", "Orange", "Vert"])
    print(summ.to_string(float_format=lambda x: f"{x:,.3f}" if x < 10 else f"{x:,.0f}"))

    print(f"\nSur le test (hors echantillon, mois par mois) :")
    print(f"  Alertes Rouge  ({RED_TOP:.0%} du portefeuille)  captent {cap_red:.1%} des migrations reelles")
    print(f"  Rouge + Orange ({ORANGE_TOP:.0%} du portefeuille) captent {cap_ro:.1%} des migrations reelles")

    print("\n=== Agences (du plus au moins risque) ===")
    print(br[["branch_id", "name", "n_loans", "expected_migration_rate", "current_par30",
              "n_red", "risk_vs_network"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== 5 agents les plus a risque ===")
    print(off[["officer_id", "branch_id", "n_loans", "expected_migration_rate", "current_par30",
               "n_red"]].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== 5 agents les moins a risque ===")
    print(off[["officer_id", "branch_id", "n_loans", "expected_migration_rate", "current_par30",
               "n_red"]].tail(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nFichiers ecrits dans outputs/scores/ et docs/sample_outputs/")


if __name__ == "__main__":
    main()