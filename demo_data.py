"""Generate realistic demo data for the Medical Expat Scheme Risk Dashboard."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io


COUNTRIES = [
    "United Arab Emirates", "Singapore", "Hong Kong", "United Kingdom",
    "United States", "Switzerland", "Germany", "China", "Thailand", "Brazil",
    "Saudi Arabia", "Nigeria", "South Africa", "India", "Australia",
]

BENEFIT_TYPES = [
    "Inpatient", "Outpatient", "Dental", "Maternity",
    "Mental Health", "Preventive Care", "Emergency Evacuation",
]

CLAIM_STATUSES = ["Paid", "Reserved", "Open", "Declined"]


def generate_claims(n_claims: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_range = (end_date - start_date).days

    # Country weights (some countries have more expats)
    country_weights = np.array([
        0.15, 0.12, 0.10, 0.10, 0.08, 0.07, 0.07, 0.06,
        0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03,
    ])

    # Benefit type weights
    benefit_weights = np.array([0.20, 0.35, 0.12, 0.08, 0.07, 0.13, 0.05])

    # Average claim amounts by benefit type (USD)
    avg_amounts = {
        "Inpatient": 12000, "Outpatient": 800, "Dental": 500,
        "Maternity": 15000, "Mental Health": 2000,
        "Preventive Care": 400, "Emergency Evacuation": 35000,
    }

    countries = rng.choice(COUNTRIES, size=n_claims, p=country_weights)
    benefits = rng.choice(BENEFIT_TYPES, size=n_claims, p=benefit_weights)
    statuses = rng.choice(
        CLAIM_STATUSES, size=n_claims, p=[0.65, 0.15, 0.12, 0.08]
    )

    dates = [
        start_date + timedelta(days=int(rng.integers(0, days_range)))
        for _ in range(n_claims)
    ]

    amounts = []
    for benefit in benefits:
        avg = avg_amounts[benefit]
        amount = max(50, rng.lognormal(np.log(avg), 0.8))
        amounts.append(round(amount, 2))

    # Age distribution of expats
    ages = rng.normal(42, 12, size=n_claims).clip(18, 75).astype(int)

    # Country cost multipliers
    cost_multipliers = {
        "United States": 1.8, "Switzerland": 1.6, "Singapore": 1.3,
        "Hong Kong": 1.3, "United Arab Emirates": 1.2,
        "United Kingdom": 1.1, "Germany": 1.0, "Australia": 1.1,
        "China": 0.7, "Thailand": 0.5, "Brazil": 0.6,
        "Saudi Arabia": 0.9, "Nigeria": 0.4, "South Africa": 0.5,
        "India": 0.3,
    }
    amounts = [
        round(a * cost_multipliers.get(c, 1.0), 2)
        for a, c in zip(amounts, countries)
    ]

    # Add some large claims
    large_claim_indices = rng.choice(n_claims, size=int(n_claims * 0.02), replace=False)
    for idx in large_claim_indices:
        amounts[idx] = round(amounts[idx] * rng.uniform(5, 15), 2)

    return pd.DataFrame({
        "Claim_ID": [f"CLM-{i+1:05d}" for i in range(n_claims)],
        "Claim_Date": dates,
        "Country": countries,
        "Benefit_Type": benefits,
        "Claim_Amount_USD": amounts,
        "Status": statuses,
        "Claimant_Age": ages,
        "Gender": rng.choice(["M", "F"], size=n_claims, p=[0.55, 0.45]),
    })


def generate_premiums(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    base_premiums = {
        "United Arab Emirates": 5500, "Singapore": 5000, "Hong Kong": 4800,
        "United Kingdom": 4200, "United States": 6000, "Switzerland": 5800,
        "Germany": 4000, "China": 2500, "Thailand": 2000, "Brazil": 2200,
        "Saudi Arabia": 3500, "Nigeria": 1800, "South Africa": 2000,
        "India": 1500, "Australia": 4500,
    }

    base_lives = {
        "United Arab Emirates": 320, "Singapore": 250, "Hong Kong": 210,
        "United Kingdom": 200, "United States": 180, "Switzerland": 150,
        "Germany": 140, "China": 120, "Thailand": 100, "Brazil": 90,
        "Saudi Arabia": 85, "Nigeria": 70, "South Africa": 65,
        "India": 60, "Australia": 55,
    }

    for year in [2022, 2023, 2024]:
        for quarter in [1, 2, 3, 4]:
            period = f"{year}-Q{quarter}"
            for country in COUNTRIES:
                growth = 1 + (year - 2022) * 0.05 + rng.normal(0, 0.02)
                lives = int(base_lives[country] * growth * (1 + rng.normal(0, 0.05)))
                premium_per_life = base_premiums[country] * (
                    1 + (year - 2022) * 0.08 + rng.normal(0, 0.03)
                )
                quarterly_premium = round(lives * premium_per_life / 4, 2)

                rows.append({
                    "Period": period,
                    "Country": country,
                    "Earned_Premium_USD": quarterly_premium,
                    "Lives_Covered": lives,
                    "Premium_Per_Life_USD": round(premium_per_life, 2),
                })

    return pd.DataFrame(rows)


def generate_reserves(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for year in [2022, 2023, 2024]:
        for quarter in [1, 2, 3, 4]:
            period = f"{year}-Q{quarter}"
            base_os = 800000 * (1 + (year - 2022) * 0.1)
            base_ibnr = 400000 * (1 + (year - 2022) * 0.12)

            rows.append({
                "Period": period,
                "Outstanding_Reserves_USD": round(
                    base_os * (1 + rng.normal(0, 0.1)), 2
                ),
                "IBNR_USD": round(base_ibnr * (1 + rng.normal(0, 0.15)), 2),
                "Total_Reserves_USD": 0,
            })

    df = pd.DataFrame(rows)
    df["Total_Reserves_USD"] = (
        df["Outstanding_Reserves_USD"] + df["IBNR_USD"]
    ).round(2)
    return df


def create_demo_excel() -> io.BytesIO:
    """Create an in-memory Excel file with all demo data sheets."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        generate_claims().to_excel(writer, sheet_name="Claims", index=False)
        generate_premiums().to_excel(writer, sheet_name="Premiums", index=False)
        generate_reserves().to_excel(writer, sheet_name="Reserves", index=False)
    buffer.seek(0)
    return buffer


def save_demo_excel(path: str = "demo_data.xlsx") -> str:
    """Save demo data to an Excel file on disk."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        generate_claims().to_excel(writer, sheet_name="Claims", index=False)
        generate_premiums().to_excel(writer, sheet_name="Premiums", index=False)
        generate_reserves().to_excel(writer, sheet_name="Reserves", index=False)
    return path


if __name__ == "__main__":
    path = save_demo_excel()
    print(f"Demo data saved to {path}")
