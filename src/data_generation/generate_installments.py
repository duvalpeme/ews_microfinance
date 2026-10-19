"""Étape 4 : génération des échéances et des retards de paiement (données synthétiques).

Usage, depuis la racine du projet :
    python -m src.data_generation.generate_installments
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg

RAW_DIR = Path("data/raw")
HIDDEN_DIR = RAW_DIR / "hidden"

# --- Paramètres calibrables -------------------------------------------------
BASE = -3.9             # niveau général de risque (monter = plus de retards)
DEFAULT_OFFSET = -2.0   # monter = plus de crédits qui cessent de payer
W_RATIO = 3.0           # poids du ratio échéance/revenu
W_CYCLE1 = 0.35         # surrisque des premiers cycles
W_OFFICER = 1.0
W_BRANCH = 1.0
W_PREV_LATE = 1.1       # persistance : un retard en entraîne d'autres
W_SEASON = 1.0
W_SHOCK = 1.0

SECTOR_RISK = {
    "commerce": 0.00, "agriculture": 0.35, "elevage": 0.20,
    "transport": 0.15, "artisanat": 0.05, "services": -0.15,
}
SEASON = {1: 0.35, 2: 0.25, 9: 0.35, 10: 0.15}  # frais scolaires, rentrée

# (secteur, mois de départ depuis janvier 2023, durée en mois, intensité)
EVENTS = [
    ("commerce", 8, 3, 0.6),      # sept. 2023 : tension de liquidité
    ("transport", 17, 4, 0.9),    # juin 2024 : hausse du carburant
    ("agriculture", 31, 4, 1.0),  # août 2025 : inondations
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def season(month, sector):
    s = SEASON.get(month, 0.0)
    if sector in ("agriculture", "elevage") and month in (6, 7, 8):
        s += 0.4  # période de soudure
    return s


def month_index(ts):
    start = pd.Timestamp(cfg.START_DATE)
    return (ts.year - start.year) * 12 + (ts.month - start.month)


def build_sector_shocks(rng):
    shocks = {}
    for sector in cfg.SECTORS:
        x = np.zeros(cfg.N_MONTHS)
        for t in range(1, cfg.N_MONTHS):
            x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.20)
        shocks[sector] = x
    for sector, start, length, intensity in EVENTS:
        shocks[sector][start:start + length] += intensity
    return shocks


def simulate_installments(rng):
    loans = pd.read_csv(RAW_DIR / "loans.csv")
    clients = pd.read_csv(RAW_DIR / "clients.csv")
    h_clients = pd.read_csv(HIDDEN_DIR / "client_hidden.csv")
    h_officers = pd.read_csv(HIDDEN_DIR / "officer_hidden.csv")
    h_branches = pd.read_csv(HIDDEN_DIR / "branch_hidden.csv")

    df = (loans
          .merge(clients[["client_id", "officer_id", "branch_id", "sector",
                          "declared_monthly_income"]], on="client_id")
          .merge(h_clients, on="client_id")
          .merge(h_officers, on="officer_id")
          .merge(h_branches, on="branch_id"))

    shocks = build_sector_shocks(rng)
    end = pd.Timestamp(cfg.END_DATE)
    rows = []

    for r in df.itertuples(index=False):
        disb = pd.Timestamp(r.disbursement_date)
        ratio = r.monthly_installment / r.declared_monthly_income
        static = (BASE
                  + W_RATIO * ratio
                  + SECTOR_RISK[r.sector]
                  + (W_CYCLE1 if r.cycle_number == 1 else 0.0)
                  + W_OFFICER * r.officer_effect
                  + W_BRANCH * r.branch_effect
                  + r.client_risk_latent)

        prev_delay = 0
        defaulted = False

        for k in range(1, r.term_months + 1):
            due = disb + pd.DateOffset(months=k)
            if due > end:
                break

            z = (static
                 + W_SEASON * season(due.month, r.sector)
                 + W_SHOCK * shocks[r.sector][month_index(due)])
            if prev_delay > 0:
                z += W_PREV_LATE

            if not defaulted and rng.random() < sigmoid(z + DEFAULT_OFFSET):
                defaulted = True  # le client cesse de payer

            paid_date, amount_paid, delay = None, 0, 0
            if not defaulted:
                if rng.random() < sigmoid(z):
                    scale = 6 + 0.5 * prev_delay + 10 * max(0.0, z + 2.5)
                    delay = int(min(90, np.ceil(rng.exponential(scale))))
                else:
                    delay = -int(rng.integers(0, 4))  # payé à temps ou en avance
                paid = due + pd.Timedelta(days=delay)
                if paid <= end:
                    paid_date = paid.date()
                    amount_paid = r.monthly_installment

            prev_delay = 90 if defaulted else max(delay, 0)
            rows.append({
                "loan_id": r.loan_id,
                "installment_no": k,
                "due_date": due.date(),
                "amount_due": r.monthly_installment,
                "paid_date": paid_date,
                "amount_paid": amount_paid,
            })

    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(cfg.SEED + 1)
    inst = simulate_installments(rng)
    inst.to_csv(RAW_DIR / "installments.csv", index=False)

    end = pd.Timestamp(cfg.END_DATE)
    due = pd.to_datetime(inst["due_date"])
    paid = pd.to_datetime(inst["paid_date"])
    dpd = np.where(paid.notna(), (paid - due).dt.days, (end - due).dt.days)
    inst["dpd"] = dpd

    print(f"Echeances generees : {len(inst):,}")
    print(f"Payees en retard (>0 j)        : {(inst['dpd'] > 0).mean():.1%}")
    print(f"Retard ou impaye > 30 jours     : {(inst['dpd'] > 30).mean():.1%}")
    print(f"Echeances impayees a la fin     : {paid.isna().mean():.1%}")
    loan_max = inst.groupby("loan_id")["dpd"].max()
    print(f"Credits ayant depasse 30 jours  : {(loan_max > 30).mean():.1%}")


if __name__ == "__main__":
    main()