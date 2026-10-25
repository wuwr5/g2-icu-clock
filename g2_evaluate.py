"""
10_g2_evaluate.py — G2: head-to-head evaluation of the ICU-trained clock
against six community-trained clocks under the G1 evaluation framework.

  GSE179325 (severe, n=574): ICU clock uses the honest out-of-fold score
  (icu_score_oof; it was trained on this cohort). Community clocks are
  evaluated exactly as in G1 (their EAA columns), for comparability.
  GSE174818 (death, n=128): ICU clock is fully external (full-model score)
  — this is the primary external validation.

Outputs:
  results/G2_incremental.csv
  results/G2_headtohead.csv   (DeLong: base+ICU vs base+clock)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_core as sc
from importlib import import_module

inc = import_module("incremental_framework")
CLOCKS = inc.CLOCKS
BASE = os.path.dirname(os.path.abspath(__file__))
PARSED = os.environ.get("PARSED", os.path.join(BASE, "data", "parsed"))
RESULTS = os.environ.get("RESULTS", os.path.join(BASE, "results"))
SEED = 42


def load_cohort(gse):
    ph = pd.read_csv(f"{PARSED}/{gse}_clocks.csv")
    sc_df = pd.read_csv(f"{RESULTS}/G2_scores.csv")
    sc_df = sc_df[sc_df["gse"] == gse][["gsm", "icu_score", "icu_score_oof"]]
    df = ph.merge(sc_df, on="gsm", how="inner")
    assert len(df) == len(ph), f"score merge failed for {gse}"
    return df


def oof_two_models(Xb, Xn, y, n_splits=5):
    p_base = np.full(len(y), np.nan)
    p_new = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in skf.split(Xb, y):
        p_base[te] = inc._glm_pred(Xb[tr], y[tr], Xb[te])
        p_new[te] = inc._glm_pred(Xn[tr], y[tr], Xn[te])
    return p_base, p_new


def evaluate(gse, df, y, score_col, label):
    Xb = inc.design(df)
    Xn = np.column_stack([Xb, df[score_col].fillna(0.0).to_numpy(float)])
    p_base, p_new = oof_two_models(Xb, Xn, y)

    out = {"gse": gse, "score": label, "n": len(y), "events": int(y.sum())}
    out["auc_base"] = sc.auc_mannwhitney(y, p_base)
    out["auc_new"] = sc.auc_mannwhitney(y, p_new)
    dl = sc.delong_test(y, p_new, p_base)
    out["delta_auc"] = dl["diff"]
    out["delta_auc_lo"] = dl["ci_low"]
    out["delta_auc_hi"] = dl["ci_high"]
    out["delong_p"] = dl["p"]
    ni = sc.nri_idi_bootstrap(y, p_base, p_new, n_boot=1000, seed=SEED)
    out["nri_cont"] = ni["nri_cont"]["value"]
    out["idi"] = ni["idi"]["value"]
    out["idi_p"] = ni["idi"]["p"]
    hl = sc.hosmer_lemeshow(y, p_new, g=max(4, min(10, int(y.sum()) // 5)))
    out["hl_stat"], out["hl_p"] = hl["stat"], hl["p"]
    cal_a, cal_b = sc.calibration_intercept_slope(y, p_new)
    out["cal_intercept"], out["cal_slope"] = cal_a, cal_b
    out["brier"] = sc.brier_score(y, p_new)
    lo, hi = sc.dca_net_benefit_range(y, p_new)
    out["dca_new_range_lo"], out["dca_new_range_hi"] = lo, hi
    out["auc_score_only"] = sc.auc_mannwhitney(y, df[score_col].fillna(0.0))
    return out, p_base, p_new


def main():
    rows, h2h = [], []
    for gse, score_col, note in [("GSE179325", "icu_score_oof", "OOF in-training"),
                                 ("GSE174818", "icu_score", "external")]:
        df = load_cohort(gse)
        if gse == "GSE179325":
            y, oname = inc.outcome_vector(df, gse)
        else:
            y = df["death"].fillna(0).astype(int).to_numpy()
            oname = "death"

        r_icu, _, p_icu = evaluate(gse, df, y, score_col, "icu_clock")
        r_icu["note"] = note
        rows.append(r_icu)

        for c in CLOCKS:
            col = f"{c}_eaa" if f"{c}_eaa" in df.columns else c
            r_c, _, p_clock = evaluate(gse, df, y, col, c)
            r_c["note"] = "as G1 (EAA)"
            rows.append(r_c)
            dl = sc.delong_test(y, p_icu, p_clock)
            h2h.append({"gse": gse, "outcome": oname, "vs": c,
                        "auc_icu": r_icu["auc_new"], "auc_clock": r_c["auc_new"],
                        "delta_auc_icu_minus_clock": dl["diff"],
                        "ci_lo": dl["ci_low"], "ci_hi": dl["ci_high"],
                        "delong_p": dl["p"]})

    pd.DataFrame(rows).to_csv(f"{RESULTS}/G2_incremental.csv", index=False)
    pd.DataFrame(h2h).to_csv(f"{RESULTS}/G2_headtohead.csv", index=False)
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(rows)[["gse", "score", "auc_base", "auc_new",
                                  "delta_auc", "delong_p", "nri_cont", "idi",
                                  "cal_slope"]].round(4).to_string(index=False))
        print()
        print(pd.DataFrame(h2h).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
