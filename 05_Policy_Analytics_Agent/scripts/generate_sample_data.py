"""Create the two sample datasets used by the demo, tests and evaluation.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FIRST = ["Omar", "Sara", "Ahmed", "Laila", "Youssef", "Nour", "Hana", "Karim", "Mona", "Tarek",
         "Salma", "Ali", "Dina", "Mostafa", "Rania", "Hassan", "Farah", "Adel", "Yara", "Ziad"]
LAST = ["Hassan", "Mahmoud", "Ibrahim", "Saleh", "Fawzy", "Nasser", "Khalil", "Samir", "Adel", "Mansour"]


def make_employees(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing"]
    base = {"Engineering": 9000, "Sales": 7000, "HR": 6000, "Finance": 8000, "Marketing": 6500}
    names, seen = [], set()
    while len(names) < n:
        full = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if full not in seen:
            seen.add(full)
            names.append(full)
    dept = rng.choice(departments, n)
    exp = rng.integers(0, 20, n)
    salary = np.array([base[d] for d in dept]) + exp * 320 + rng.normal(0, 500, n)
    return pd.DataFrame({
        "employee_id": np.arange(1001, 1001 + n),
        "name": names,
        "department": dept,
        "city": rng.choice(["Cairo", "Giza", "Alexandria", "Mansoura"], n),
        "years_experience": exp,
        "salary": salary.round(0).astype(int),
        "bonus": (salary * rng.uniform(0.03, 0.15, n)).round(0).astype(int),
        "performance_score": rng.uniform(2.5, 5.0, n).round(1),
    })


def make_sales(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = ["Electronics", "Furniture", "Clothing", "Books", "Sports", "Food"]
    price = {"Electronics": 900, "Furniture": 450, "Clothing": 60, "Books": 18, "Sports": 90, "Food": 12}
    cat = rng.choice(cats, n)
    qty = rng.integers(1, 8, n)
    unit = np.array([price[c] for c in cat]) * rng.uniform(0.8, 1.3, n)
    disc = rng.choice([0.0, 0.05, 0.1, 0.2], n)
    revenue = qty * unit * (1 - disc)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    return pd.DataFrame({
        "order_id": np.arange(50001, 50001 + n),
        "order_date": dates.strftime("%Y-%m-%d"),
        "customer_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "region": rng.choice(["North", "South", "East", "West", "Central"], n),
        "product_category": cat,
        "quantity": qty,
        "unit_price": unit.round(2),
        "discount": disc,
        "revenue": revenue.round(2),
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_employees().to_csv(DATA_DIR / "employees.csv", index=False)
    make_sales().to_csv(DATA_DIR / "sales_dataset.csv", index=False)
    print(f"Wrote sample data to {DATA_DIR}")


if __name__ == "__main__":
    main()
