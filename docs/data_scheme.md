# Schéma de données (synthétique)

## Tables
- branches(branch_id, name, region, setting, size)
- loan_officers(officer_id, branch_id, tenure_months, target_portfolio)
- clients(client_id, officer_id, branch_id, age, sector, client_tenure_months, declared_monthly_income)
- loans(loan_id, client_id, product, amount, interest_rate, term_months, disbursement_date, cycle_number, collateral_type)
- installments(loan_id, installment_no, due_date, amount_due, paid_date, amount_paid)

## Variables cachées (génération uniquement)
client_risk_latent, officer_effect, branch_effect, sector_shock(month)

## Cible (étape features)
Au mois t, pour un crédit PAR 0 ou 1-29 j : target = 1 si dépassement de 30 jours de retard dans les 30 jours suivants.

## Exclusions éthiques
Aucun genre, aucune religion, aucune ethnie.
