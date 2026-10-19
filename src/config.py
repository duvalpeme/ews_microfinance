"""Paramètres globaux du projet."""
from datetime import date

SEED = 42

N_BRANCHES = 8
N_OFFICERS = 40
N_CLIENTS = 6000

START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)
N_MONTHS = 36
N_TEST_MONTHS = 6

PAR_THRESHOLD_DAYS = 30
TARGET_HORIZON_DAYS = 30
TARGET_RATE_RANGE = (0.05, 0.10)

SECTORS = ["commerce", "agriculture", "elevage", "transport", "artisanat", "services"]
PRODUCTS = ["individuel", "groupe_solidaire", "agricole"]
