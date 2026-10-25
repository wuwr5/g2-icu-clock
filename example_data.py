"""
example_data.py — generate a small synthetic dataset so the pipeline can be
smoke-tested without the (public, processed) GEO inputs.

Writes:
  data/parsed/GSE179325_beta.parquet / _clocks.csv
  data/parsed/GSE174818_beta.parquet  / _clocks.csv

The synthetic cohorts mimic the column contracts that g2_train.py and
g2_evaluate.py expect, with a weak severity/death signal injected so the
models train and evaluate without error. The numbers are NOT real effects.
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
BASE = os.path.dirname(os.path.abspath(__file__))
PARSED = os.path.join(BASE, "data", "parsed")
CLOCKS = ["horvath2013", "dnamphenoage", "grimage2",
          "dunedinpace", "altumage", "hannum"]
P = 300  # number of CpGs


def _make_cohort(n, with_death, n_severe=None):
    gsms = [f"GSM{i}" for i in range(1, n + 1)]
    age = RNG.integers(30, 90, n)
    sex = RNG.choice(["M", "F"], n)
    beta = RNG.beta(2, 2, size=(P, n)).astype("float32")  # ~U-shaped methylation
    if with_death:
        # GSE174818: in-hospital death
        y = RNG.binomial(1, 0.2, n)
        beta[:20, y == 1] = np.clip(beta[:20, y == 1] + 0.15, 0, 1)
        ph = pd.DataFrame({"gsm": gsms, "age": age, "sex": sex,
                           "death": y, "disease_state": np.nan,
                           "covid_status": "positive", "cohort": "GSE174818",
                           "subject_status": "positive"})
    else:
        # GSE179325: severe COVID-19
        n_sev = n_severe or int(n * 0.4)
        y = np.zeros(n, int)
        y[:n_sev] = 1
        RNG.shuffle(y)
        beta[:20, y == 1] = np.clip(beta[:20, y == 1] + 0.15, 0, 1)
        ph = pd.DataFrame({"gsm": gsms, "age": age, "sex": sex,
                           "death": np.nan,
                           "disease_state": np.where(y == 1, "SEVERE", "mild"),
                           "covid_status": "positive", "cohort": "GSE179325",
                           "subject_status": "positive"})
    for c in CLOCKS:
        ph[f"{c}_eaa"] = RNG.normal(0, 3, n)
    beta_df = pd.DataFrame(beta, index=[f"cg{i:07d}" for i in range(P)],
                           columns=gsms)
    return beta_df, ph


def main():
    os.makedirs(PARSED, exist_ok=True)
    b1, p1 = _make_cohort(250, with_death=False, n_severe=100)
    b2, p2 = _make_cohort(128, with_death=True)
    b1.to_parquet(os.path.join(PARSED, "GSE179325_beta.parquet"))
    p1.to_csv(os.path.join(PARSED, "GSE179325_clocks.csv"), index=False)
    b2.to_parquet(os.path.join(PARSED, "GSE174818_beta.parquet"))
    p2.to_csv(os.path.join(PARSED, "GSE174818_clocks.csv"), index=False)
    print("wrote synthetic data to", PARSED)


if __name__ == "__main__":
    main()
