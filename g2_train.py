"""
09_g2_train.py — G2: train a critical-illness (ICU) DNA methylation clock.

Design:
  Training label  : severe COVID-19 (GSE179325 COVID cases, n=473, 113 severe)
  Features        : 40,414 clock CpGs (beta values, z-scored on training set)
  Model           : logistic regression with elastic-net penalty, C chosen by
                    inner 5-fold CV (nested inside a 5-fold outer loop for
                    honest out-of-fold scores on the training cohort)
  Final model     : refit on all training samples with the selected C
  Scoring         : apply z-score + linear predictor to BOTH cohorts
                    (GSE179325 all 574 samples, GSE174818 all 128 samples);
                    CpGs missing in the target cohort are imputed with the
                    training mean (=0 in z-space)

Outputs:
  results/G2_icu_clock_coefs.csv   (cpg, coef, abs_rank)
  results/G2_scores.csv            (gse, gsm, icu_score)  — icu_score is the
                    raw linear predictor (z-space); higher = more "ICU-like"
  results/G2_train_cv.csv          (outer fold, best C, outer AUC)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
PARSED = os.environ.get("PARSED", os.path.join(BASE, "data", "parsed"))
RESULTS = os.environ.get("RESULTS", os.path.join(BASE, "results"))
SEED = 42

def load(gse):
    beta = pd.read_parquet(f"{PARSED}/{gse}_beta.parquet")      # index cpg, cols gsm
    ph = pd.read_csv(f"{PARSED}/{gse}_clocks.csv")
    ph = ph.rename(columns={"gsm": "gsm"})
    return beta, ph

def main():
    os.makedirs(RESULTS, exist_ok=True)
    np.random.seed(SEED)

    beta_tr, ph_tr = load("GSE179325")
    # training set: COVID-19 positive cases only, label = severe
    cases = ph_tr["subject_status"].astype(str).str.contains("positive")
    ph_t = ph_tr[cases].reset_index(drop=True)
    y = (ph_t["disease_state"].astype(str).str.upper() == "SEVERE").astype(int).to_numpy()
    gsm_t = ph_t["gsm"].to_numpy()
    print(f"[train] GSE179325 COVID cases: n={len(gsm_t)}, events={y.sum()}")

    # X: cpg x sample -> sample x cpg
    cols = [c for c in beta_tr.columns if c in set(gsm_t)]
    beta_tr = beta_tr[cols]
    X_all = beta_tr.T.to_numpy(dtype=np.float32)          # (n, p)
    assert list(cols) == list(gsm_t), "column/gsm order mismatch"

    # drop all-NaN CpGs, impute remaining NaN with column mean
    ok = np.isfinite(X_all).mean(axis=0) > 0.5
    X_all = X_all[:, ok]
    col_mean = np.nanmean(X_all, axis=0)
    nan_idx = np.where(np.isnan(X_all))
    X_all[nan_idx] = col_mean[nan_idx[1]]
    # z-score on training stats
    mu, sd = X_all.mean(0), X_all.std(0) + 1e-8
    Xz = (X_all - mu) / sd
    cpgs = beta_tr.index[ok].to_numpy()
    print(f"[train] CpGs used: {len(cpgs)}")

    Cs = np.logspace(-2, 1, 4)   # coarser grid: ~4x faster than 9-point grid

    def fit_enet(Xtr, ytr, cv=5):
        from sklearn.linear_model import LogisticRegressionCV
        lr = LogisticRegressionCV(
            Cs=list(Cs), cv=cv, scoring="roc_auc",
            penalty="elasticnet", solver="saga", l1_ratios=[0.5],
            max_iter=4000, tol=1e-3, n_jobs=-1, random_state=SEED)
        lr.fit(Xtr, ytr)
        return lr

    # ---- outer 5-fold for honest OOF scores on training cohort ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    cv_rows = []
    for k, (tr, te) in enumerate(skf.split(Xz, y), 1):
        lr = fit_enet(Xz[tr], y[tr])
        best_C = lr.C_[0]
        p = lr.predict_proba(Xz[te])[:, 1]
        oof[te] = p
        auc = roc_auc_score(y[te], p)
        cv_rows.append({"fold": k, "best_C": best_C, "n_train": len(tr), "outer_auc": auc})
        print(f"[cv] fold {k}: best_C={best_C:.3g}, outer AUC={auc:.3f}")
    pd.DataFrame(cv_rows).to_csv(f"{RESULTS}/G2_train_cv.csv", index=False)
    print(f"[cv] mean outer AUC = {np.mean([r['outer_auc'] for r in cv_rows]):.3f}")

    # ---- final model on all training samples ----
    final = fit_enet(Xz, y)
    best_C = final.C_[0]
    print(f"[final] C={best_C:.3g}, in-sample AUC={roc_auc_score(y, final.predict_proba(Xz)[:,1]):.3f}")

    coef = pd.DataFrame({"cpg": cpgs, "coef": final.coef_.ravel()})
    coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
    coef["abs_rank"] = np.arange(1, len(coef) + 1)
    coef.to_csv(f"{RESULTS}/G2_icu_clock_coefs.csv", index=False)
    nz = (coef["coef"] != 0).sum()
    print(f"[final] non-zero coefficients: {nz}/{len(coef)}")

    # ---- score both cohorts ----
    rows = []
    # training cohort: OOF score (honest) for COVID cases; full-model score for all
    score_oof = pd.DataFrame({"gsm": gsm_t, "icu_score_oof": oof})
    for gse, beta, ph in [("GSE179325", *load("GSE179325")),
                          ("GSE174818", *load("GSE174818"))]:
        common = [c for c in beta.index if c in set(cpgs)]
        M = beta.loc[common].T.to_numpy(dtype=np.float32)   # (n, len(common))
        # impute missing CpGs (absent from chip or NaN) with training mean (0 in z-space)
        Z = np.zeros((M.shape[0], len(cpgs)), dtype=np.float32)
        pos = {c: i for i, c in enumerate(cpgs)}
        idx = [pos[c] for c in common]
        Z[:, idx] = M
        nan = np.isnan(Z)
        Z[nan] = 0.0
        Z = (Z - mu) / sd
        lp = Z @ final.coef_.ravel() + final.intercept_[0]
        rows.append(pd.DataFrame({"gse": gse, "gsm": beta.columns,
                                  "icu_score": lp}))
    scores = pd.concat(rows, ignore_index=True)
    scores = scores.merge(score_oof, on="gsm", how="left")
    scores.to_csv(f"{RESULTS}/G2_scores.csv", index=False)

    print(f"[done] scored {scores.groupby('gse').size().to_dict()}")
    # quick AUC summary
    for gse in ["GSE179325", "GSE174818"]:
        ph = load(gse)[1]
        m = scores[scores["gse"] == gse].merge(ph[["gsm", "age", "sex", "death",
                                                  "disease_state"]], on="gsm")
        if gse == "GSE179325":
            yv = (m["disease_state"].astype(str).str.upper() == "SEVERE").astype(int)
            col = "icu_score_oof"
            sub = m[col].notna()
            print(f"[AUC] {gse} severe (OOF): {roc_auc_score(yv[sub], m.loc[sub, col]):.3f}")
        else:
            yv = m["death"].fillna(0).astype(int)
            print(f"[AUC] {gse} death (external, full-model): "
                  f"{roc_auc_score(yv, m['icu_score']):.3f}")

if __name__ == "__main__":
    main()
