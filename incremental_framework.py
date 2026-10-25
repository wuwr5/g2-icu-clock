"""
04_incremental.py — G1: incremental predictive value of epigenetic age
acceleration (EAA) over routine predictors in critical illness.

Per cohort (public GEO), per clock:
  base model : age + sex                      (routine bedside predictors)
  new  model : age + sex + EAA (age residual)

Metrics (all self-implemented in stats_core.py):
  AUC (bootstrap CI), DeLong ΔAUC + CI, continuous & category NRI, IDI
  (bootstrap SE/CI), Hosmer-Lemeshow, calibration intercept/slope, Brier,
  DCA net benefit (model vs treat-all/treat-none) and the threshold range
  where the model is beneficial.

Cohorts:
  GSE174818  outcome = in-hospital death      (n=128, 28 events)
  GSE179325  outcome = severe COVID-19        (n=574, 113 events)

Output: results/G1_incremental.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import stats_core as sc

CLOCKS = ["horvath2013", "dnamphenoage", "grimage2", "dunedinpace", "altumage", "hannum"]


def outcome_vector(df: pd.DataFrame, gse: str):
    """Return (y, outcome_name)."""
    if gse == "GSE174818" and "death" in df.columns and df["death"].notna().any():
        return df["death"].fillna(0).astype(int).to_numpy(), "death"
    if gse == "GSE179325" and "disease_state" in df.columns:
        y = (df["disease_state"].astype(str).str.upper() == "SEVERE").astype(int)
        return y.to_numpy(), "severe"
    raise ValueError(f"no outcome defined for {gse}")


def design(df, eaa_col=None):
    age = df["age"].fillna(df["age"].median()).to_numpy(dtype=float)
    male = (df["sex"].astype(str).str.upper().str.startswith("M")).astype(float).to_numpy()
    X_base = np.column_stack([np.ones(len(df)), age, male])
    if eaa_col is None:
        return X_base
    eaa = df[eaa_col].fillna(0.0).to_numpy(dtype=float)
    return np.column_stack([X_base, eaa])


def fit_p(X, y):
    import statsmodels.api as sm
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0)
        p = np.asarray(res.predict(X), dtype=float)
    if not np.all(np.isfinite(p)):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
    return p


def oof_predictions(gse, df, eaa_col, n_splits=5, seed=42):
    """Pooled out-of-fold predicted probabilities for base and base+EAA models."""
    from sklearn.model_selection import StratifiedKFold
    y, _ = outcome_vector(df, gse)
    Xb, Xn = design(df), design(df, eaa_col)
    p_base = np.full(len(df), np.nan)
    p_new = np.full(len(df), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xb, y):
        p_base[te] = _glm_pred(Xb[tr], y[tr], Xb[te])
        p_new[te] = _glm_pred(Xn[tr], y[tr], Xn[te])
    return p_base, p_new


def _glm_pred(Xtr, ytr, Xte):
    import statsmodels.api as sm
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(ytr, Xtr, family=sm.families.Binomial()).fit(disp=0)
        p = np.asarray(res.predict(Xte), dtype=float)
    if not np.all(np.isfinite(p)):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
    return p


def evaluate(gse, df, clock):
    y, oname = outcome_vector(df, gse)
    out = dict(gse=gse, clock=clock, outcome=oname, n=len(df), events=int(y.sum()))

    p_base = fit_p(design(df), y)
    p_new = fit_p(design(df, f"{clock}_eaa"), y)
    out["auc_base"] = sc.auc_mannwhitney(y, p_base)
    out["auc_new"] = sc.auc_mannwhitney(y, p_new)

    dl = sc.delong_test(y, p_new, p_base)
    out["delta_auc"] = dl["diff"]
    out["delta_auc_lo"] = dl["ci_low"]
    out["delta_auc_hi"] = dl["ci_high"]
    out["delong_p"] = dl["p"]

    ni = sc.nri_idi_bootstrap(y, p_base, p_new, n_boot=1000, seed=42,
                              category_cutoffs=[0.1, 0.25, 0.5])
    out["nri_cont"] = ni["nri_cont"]["value"]
    out["nri_cont_lo"] = ni["nri_cont"]["ci_low"]
    out["nri_cont_hi"] = ni["nri_cont"]["ci_high"]
    out["nri_cat"] = ni["nri_cat"]["value"]
    out["idi"] = ni["idi"]["value"]
    out["idi_lo"] = ni["idi"]["ci_low"]
    out["idi_hi"] = ni["idi"]["ci_high"]
    out["idi_p"] = ni["idi"]["p"]

    hl = sc.hosmer_lemeshow(y, p_new, g=max(4, min(10, int(y.sum()) // 5)))
    out["hl_stat"] = hl["stat"]
    out["hl_p"] = hl["p"]
    out["hl_groups"] = max(4, min(10, int(y.sum()) // 5))
    cal_a, cal_b = sc.calibration_intercept_slope(y, p_new)
    out["cal_intercept"], out["cal_slope"] = cal_a, cal_b
    out["brier"] = sc.brier_score(y, p_new)

    # DCA: net benefit of base vs new at clinical thresholds
    for pt in [0.05, 0.10, 0.20, 0.30, 0.50]:
        dc_b = sc.decision_curve(y, p_base, thresholds=[pt])
        dc_n = sc.decision_curve(y, p_new, thresholds=[pt])
        out[f"nb_base@{pt:.2f}"] = float(dc_b.nb_model.iloc[0])
        out[f"nb_new@{pt:.2f}"] = float(dc_n.nb_model.iloc[0])
        out[f"nb_all@{pt:.2f}"] = float(dc_n.nb_all.iloc[0])
    # threshold range where the new model beats treat-all
    lo, hi = sc.dca_net_benefit_range(y, p_new)
    out["dca_new_range_lo"], out["dca_new_range_hi"] = lo, hi
    lo, hi = sc.dca_net_benefit_range(y, p_base)
    out["dca_base_range_lo"], out["dca_base_range_hi"] = lo, hi

    # univariate EAA-only AUC (reference)
    out["auc_eaa_only"] = sc.auc_mannwhitney(y, df[f"{clock}_eaa"].fillna(0.0))

    # out-of-fold (5-fold CV) sensitivity: pooled OOF predictions
    try:
        p_base_oof, p_new_oof = oof_predictions(gse, df, f"{clock}_eaa")
        out["cv_auc_base"] = sc.auc_mannwhitney(y, p_base_oof)
        out["cv_auc_new"] = sc.auc_mannwhitney(y, p_new_oof)
        out["cv_delta_auc"] = out["cv_auc_new"] - out["cv_auc_base"]
        dl_oof = sc.delong_test(y, p_new_oof, p_base_oof)
        out["cv_delong_p"] = dl_oof["p"]
        cal_a, cal_b = sc.calibration_intercept_slope(y, p_new_oof)
        out["cv_cal_slope"] = cal_b
    except Exception as e:
        print(f"    OOF failed for {gse}/{clock}: {repr(e)[:100]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["GSE174818", "GSE179325"])
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for gse in args.cohorts:
        df = pd.read_csv(f"{args.parsed}/{gse}_clocks.csv")
        for c in CLOCKS:
            if c not in df.columns or df[f"{c}_eaa"].notna().sum() < 20:
                continue
            try:
                rows.append(evaluate(gse, df, c))
                print(f"  {gse} {c}: ΔAUC={rows[-1]['delta_auc']:+.4f} "
                      f"IDI={rows[-1]['idi']:+.4f}")
            except Exception as e:
                print(f"  {gse} {c} FAILED: {repr(e)[:120]}")
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}/G1_incremental.csv", index=False)
    cols = ["gse", "clock", "outcome", "auc_base", "auc_new", "delta_auc",
            "delta_auc_lo", "delta_auc_hi", "delong_p", "nri_cont", "idi",
            "idi_lo", "idi_hi", "hl_p", "cal_slope", "nb_new@0.10", "nb_base@0.10"]
    print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
