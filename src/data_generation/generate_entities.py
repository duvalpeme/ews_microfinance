"""Étape 3 : génération des agences, agents, clients et crédits (données synthétiques).

Usage, depuis la racine du projet :
    python -m src.data_generation.generate_entities
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg

RAW_DIR = Path("data/raw")
HIDDEN_DIR = RAW_DIR / "hidden"  # variables cachées : jamais données au modèle

# Revenu mensuel médian par secteur (XAF)
SECTOR_INCOME = {
    "commerce": 150_000,
    "agriculture": 90_000,
    "elevage": 110_000,
    "transport": 180_000,
    "artisanat": 120_000,
    "services": 200_000,
}
SECTOR_PROBS = {
    "urbain": [0.40, 0.05, 0.03, 0.15, 0.12, 0.25],
    "rural": [0.20, 0.35, 0.15, 0.08, 0.10, 0.12],
}
ANNUAL_RATE = {"individuel": 0.24, "groupe_solidaire": 0.20, "agricole": 0.18}


def generate_branches(rng):
    names = ["Douala-Akwa", "Yaounde-Mvog-Mbi", "Bafoussam", "Garoua",
             "Bertoua", "Ebolowa", "Maroua", "Bamenda"][: cfg.N_BRANCHES]
    regions = ["Littoral", "Centre", "Ouest", "Nord",
               "Est", "Sud", "Extreme-Nord", "Nord-Ouest"][: cfg.N_BRANCHES]
    settings = ["urbain", "urbain", "urbain", "urbain",
                "rural", "rural", "rural", "urbain"][: cfg.N_BRANCHES]
    sizes = ["grande", "grande", "moyenne", "moyenne",
             "petite", "petite", "moyenne", "petite"][: cfg.N_BRANCHES]
    branches = pd.DataFrame({
        "branch_id": [f"BR{i+1:02d}" for i in range(cfg.N_BRANCHES)],
        "name": names,
        "region": regions,
        "setting": settings,
        "size": sizes,
    })
    hidden = pd.DataFrame({
        "branch_id": branches["branch_id"],
        "branch_effect": rng.normal(0, 0.30, cfg.N_BRANCHES),
    })
    return branches, hidden


def generate_officers(rng, branches):
    branch_ids = branches["branch_id"].to_numpy()
    weights = branches["size"].map({"grande": 3, "moyenne": 2, "petite": 1}).to_numpy(float)
    weights /= weights.sum()

    # 2 agents minimum par agence, le reste réparti selon la taille
    n_extra = cfg.N_OFFICERS - 2 * cfg.N_BRANCHES
    assigned = np.concatenate([
        np.repeat(branch_ids, 2),
        rng.choice(branch_ids, size=n_extra, p=weights),
    ])
    tenure = rng.integers(3, 121, cfg.N_OFFICERS)
    officers = pd.DataFrame({
        "officer_id": [f"OF{i+1:03d}" for i in range(cfg.N_OFFICERS)],
        "branch_id": assigned,
        "tenure_months": tenure,
    })
    hidden = pd.DataFrame({
        "officer_id": officers["officer_id"],
        # un agent expérimenté suit mieux son portefeuille
        "officer_effect": rng.normal(0, 0.30, cfg.N_OFFICERS) - 0.004 * np.minimum(tenure, 60),
    })
    return officers, hidden


def generate_clients(rng, branches, officers):
    n = cfg.N_CLIENTS
    officer_ids = officers["officer_id"].to_numpy()
    officer_to_branch = officers.set_index("officer_id")["branch_id"]
    branch_setting = branches.set_index("branch_id")["setting"]

    chosen = rng.choice(officer_ids, size=n)
    branch_ids = officer_to_branch.loc[chosen].to_numpy()
    settings = branch_setting.loc[branch_ids].to_numpy()

    sectors = np.empty(n, dtype=object)
    for setting, probs in SECTOR_PROBS.items():
        mask = settings == setting
        sectors[mask] = rng.choice(cfg.SECTORS, size=mask.sum(), p=probs)

    median_income = np.array([SECTOR_INCOME[s] for s in sectors], dtype=float)
    income = median_income * rng.lognormal(0, 0.4, n) * np.where(settings == "urbain", 1.2, 1.0)
    income = np.maximum(30_000, np.round(income / 5_000) * 5_000)

    clients = pd.DataFrame({
        "client_id": [f"CL{i+1:05d}" for i in range(n)],
        "officer_id": chosen,
        "branch_id": branch_ids,
        "age": np.clip(rng.normal(38, 10, n), 21, 65).astype(int),
        "sector": sectors,
        # ancienneté de la relation (mois) AVANT le 1er crédit du jeu de données
        "client_tenure_months": np.clip(rng.exponential(24, n), 0, 96).astype(int),
        "declared_monthly_income": income.astype(int),
    })
    hidden = pd.DataFrame({
        "client_id": clients["client_id"],
        "client_risk_latent": rng.normal(0, 0.60, n),
    })
    return clients, hidden


def pick_product(rng, sector, income):
    if sector in ("agriculture", "elevage") and rng.random() < 0.6:
        return "agricole"
    if sector in ("commerce", "artisanat") and income < 100_000 and rng.random() < 0.4:
        return "groupe_solidaire"
    return "individuel"


def pick_collateral(rng, product, amount):
    if product == "groupe_solidaire":
        return "caution_solidaire"
    if amount > 800_000:
        return str(rng.choice(["hypotheque", "gage_materiel"], p=[0.6, 0.4]))
    return str(rng.choice(["aucune", "gage_materiel", "caution_personnelle"], p=[0.4, 0.3, 0.3]))


def generate_loans(rng, clients):
    start_ts = pd.Timestamp(cfg.START_DATE)
    end_ts = pd.Timestamp(cfg.END_DATE)
    loans = []
    loan_no = 1

    for c in clients.itertuples(index=False):
        cycle = min(3, c.client_tenure_months // 12) + 1  # cycles déjà faits avant le jeu de données
        start = start_ts + pd.Timedelta(days=int(rng.integers(0, 700)))
        n_loans = 0

        while start <= end_ts - pd.Timedelta(days=120) and n_loans < 4:
            term = int(rng.choice([6, 9, 12, 18], p=[0.30, 0.20, 0.35, 0.15]))
            product = pick_product(rng, c.sector, c.declared_monthly_income)

            amount = c.declared_monthly_income * term * rng.uniform(0.10, 0.22) * (1 + 0.10 * (cycle - 1))
            amount = int(np.clip(round(amount / 5_000) * 5_000, 50_000, 2_500_000))

            rate = round(ANNUAL_RATE[product] + rng.normal(0, 0.015), 4)
            installment = round(amount * (1 + rate * term / 12) / term / 500) * 500

            loans.append({
                "loan_id": f"LN{loan_no:06d}",
                "client_id": c.client_id,
                "product": product,
                "amount": amount,
                "interest_rate": rate,
                "term_months": term,
                "disbursement_date": start.date(),
                "cycle_number": cycle,
                "collateral_type": pick_collateral(rng, product, amount),
                "monthly_installment": int(installment),
            })
            loan_no += 1
            n_loans += 1
            cycle += 1

            if rng.random() > 0.75:  # 25 % ne renouvellent pas
                break
            start = start + pd.DateOffset(months=term) + pd.Timedelta(days=int(rng.integers(0, 45)))

    return pd.DataFrame(loans)


def main():
    rng = np.random.default_rng(cfg.SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

    branches, h_branches = generate_branches(rng)
    officers, h_officers = generate_officers(rng, branches)
    clients, h_clients = generate_clients(rng, branches, officers)
    loans = generate_loans(rng, clients)

    # taille réelle du portefeuille de chaque agent
    counts = clients["officer_id"].value_counts()
    officers["target_portfolio"] = officers["officer_id"].map(counts).fillna(0).astype(int)

    branches.to_csv(RAW_DIR / "branches.csv", index=False)
    officers.to_csv(RAW_DIR / "loan_officers.csv", index=False)
    clients.to_csv(RAW_DIR / "clients.csv", index=False)
    loans.to_csv(RAW_DIR / "loans.csv", index=False)
    h_branches.to_csv(HIDDEN_DIR / "branch_hidden.csv", index=False)
    h_officers.to_csv(HIDDEN_DIR / "officer_hidden.csv", index=False)
    h_clients.to_csv(HIDDEN_DIR / "client_hidden.csv", index=False)

    print(f"Agences : {len(branches)}")
    print(f"Agents  : {len(officers)}")
    print(f"Clients : {len(clients)}")
    print(f"Credits : {len(loans)}")
    print("\nCredits par produit :")
    print(loans["product"].value_counts().to_string())
    print("\nCredits par cycle :")
    print(loans["cycle_number"].value_counts().sort_index().to_string())
    ratio = loans["monthly_installment"] / loans["client_id"].map(
        clients.set_index("client_id")["declared_monthly_income"])
    print(f"\nRatio echeance/revenu : median={ratio.median():.2f}, p90={ratio.quantile(0.9):.2f}")


if __name__ == "__main__":
    main()