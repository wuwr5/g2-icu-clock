"""
stats_core.py — Statistical core for G1+G4 critical-illness epigenetic clock evaluation.

Implements (pure numpy/scipy/statsmodels, no R dependency):
  - DeLong test for correlated ROC curves (AUC difference + 95% CI)
  - NRI (continuous & category-based) and IDI with bootstrap SEs
  - Hosmer-Lemeshow goodness-of-fit
  - Calibration slope / intercept (logistic recalibration)
  - Decision Curve Analysis (net benefit)
  - DerSimonian-Laird random-effects meta-analysis
  - Brier score, Spiegelhalter z, bootstrap utilities

All bootstrap routines accept a `seed` for reproducibility.
"""

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------------

def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def as_arrays(*arrs):
    return [np.asarray(a, dtype=float) for a in arrs]

def auc_mannwhitney(y, score):
    """AUC via Mann-Whitney U (handles ties by midranks)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    # midranks
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[y[order] == 1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# ----------------------------------------------------------------------------
# DeLong test for two correlated AUCs
# ----------------------------------------------------------------------------

def _delong_components(y, score):
    """Structural components V10 (per case) and V01 (per control) for one score."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    n1, n0 = len(pos), len(neg)
    # pairwise comparison matrix: 1 if pos > neg, 0.5 if equal
    diff = pos[:, None] - neg[None, :]
    comp = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
    v10 = comp.mean(axis=1)          # per case: average placement vs controls
    v01 = comp.mean(axis=0)          # per control: average placement vs cases
    return v10, v01

def delong_test(y, score_a, score_b):
    """
    DeLong test for AUC(score_a) - AUC(b) on correlated samples.

    Returns dict with auc_a, auc_b, diff, se, z, p, ci_low, ci_high.
    """
    y, score_a, score_b = as_arrays(y, score_a, score_b)
    y = y.astype(int)
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    v10_a, v01_a = _delong_components(y, score_a)
    v10_b, v01_b = _delong_components(y, score_b)
    auc_a = v10_a.mean()
    auc_b = v10_b.mean()
    # covariance of AUCs from structural components
    s10 = np.cov(np.vstack([v10_a, v10_b]))  # 2x2
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    if s10.shape == (0, 0):  # degenerate (all one class) - scipy returns scalar
        s10 = np.array([[np.var(v10_a, ddof=1), 0.0], [0.0, np.var(v10_b, ddof=1)]])
        s01 = np.array([[np.var(v01_a, ddof=1), 0.0], [0.0, np.var(v01_b, ddof=1)]])
    d = np.array([1.0, -1.0])
    var_diff = (d @ s10 @ d) / n1 + (d @ s01 @ d) / n0
    diff = auc_a - auc_b
    se = np.sqrt(max(var_diff, 0.0))
    if se == 0:
        return dict(auc_a=auc_a, auc_b=auc_b, diff=diff, se=0.0, z=np.nan, p=np.nan,
                    ci_low=np.nan, ci_high=np.nan)
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(auc_a=auc_a, auc_b=auc_b, diff=diff, se=se, z=z, p=p,
                ci_low=diff - 1.96 * se, ci_high=diff + 1.96 * se)

# ----------------------------------------------------------------------------
# NRI / IDI
# ----------------------------------------------------------------------------

def nri_continuous(y, p_old, p_new):
    """
    Continuous NRI (Pencina et al. 2011/2014).
    event NRI = P(p_new > p_old | event) - P(p_new > p_old | non-event)
    """
    y, p_old, p_new = as_arrays(y, p_old, p_new)
    y = y.astype(int)
    ev, ne = y == 1, y == 0
    up_e = np.mean(p_new[ev] > p_old[ev])
    dn_e = np.mean(p_new[ev] < p_old[ev])
    up_ne = np.mean(p_new[ne] > p_old[ne])
    dn_ne = np.mean(p_new[ne] < p_old[ne])
    # Pencina: events -> up - down ; non-events -> down - up
    nri_event = up_e - dn_e
    nri_nonevent = dn_ne - up_ne
    return nri_event + nri_nonevent, nri_event, nri_nonevent

def nri_category(y, p_old, p_new, cutoffs):
    """Category-based NRI with given cutoffs (e.g. [0.05, 0.2, 0.5] or clinical thresholds)."""
    y, p_old, p_new = as_arrays(y, p_old, p_new)
    y = y.astype(int)
    edges = [-np.inf] + list(cutoffs) + [np.inf]
    cat_old = np.digitize(p_old, edges[1:-1])
    cat_new = np.digitize(p_new, edges[1:-1])
    ev, ne = y == 1, y == 0
    nri_event = np.mean(cat_new[ev] > cat_old[ev]) - np.mean(cat_new[ev] < cat_old[ev])
    nri_nonevent = np.mean(cat_new[ne] < cat_old[ne]) - np.mean(cat_new[ne] > cat_old[ne])
    return nri_event + nri_nonevent, nri_event, nri_nonevent

def idi(y, p_old, p_new):
    """
    Integrated Discrimination Improvement.
    IDI = [mean(p_new|E) - mean(p_old|E)] - [mean(p_new|~E) - mean(p_old|~E)]
    """
    y, p_old, p_new = as_arrays(y, p_old, p_new)
    y = y.astype(int)
    ev, ne = y == 1, y == 0
    is_improve = np.mean(p_new[ev]) - np.mean(p_old[ev])
    ip_improve = np.mean(p_new[ne]) - np.mean(p_old[ne])
    return is_improve - ip_improve, is_improve, ip_improve

def nri_idi_bootstrap(y, p_old, p_new, n_boot=1000, seed=42, category_cutoffs=None):
    """Bootstrap SE / CI / p for continuous NRI, category NRI and IDI (stratified by outcome)."""
    y, p_old, p_new = as_arrays(y, p_old, p_new)
    y = y.astype(int)
    rng = np.random.default_rng(seed)
    idx_ev = np.where(y == 1)[0]
    idx_ne = np.where(y == 0)[0]
    stats_fn = lambda yy, po, pn: (
        nri_continuous(yy, po, pn)[0],
        idi(yy, po, pn)[0],
        (nri_category(yy, po, pn, category_cutoffs)[0] if category_cutoffs else np.nan),
    )
    point = stats_fn(y, p_old, p_new)
    boots = []
    for _ in range(n_boot):
        be = rng.choice(idx_ev, size=len(idx_ev), replace=True)
        bn = rng.choice(idx_ne, size=len(idx_ne), replace=True)
        b = np.concatenate([be, bn])
        boots.append(stats_fn(y[b], p_old[b], p_new[b]))
    boots = np.array(boots)
    out = {}
    names = ["nri_cont", "idi", "nri_cat"]
    for i, nm in enumerate(names):
        b = boots[:, i]
        b = b[~np.isnan(b)]
        se = b.std(ddof=1) if len(b) > 1 else np.nan
        z = point[i] / se if se and se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) > 1 else (np.nan, np.nan)
        out[nm] = dict(value=point[i], se=se, ci_low=lo, ci_high=hi, p=p)
    return out

# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------

def hosmer_lemeshow(y, p, g=10):
    """Hosmer-Lemeshow statistic (groups by predicted-probability deciles)."""
    y, p = as_arrays(y, p)
    y = y.astype(int)
    qs = np.quantile(p, np.linspace(0, 1, g + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    grp = np.digitize(p, qs[1:-1])
    hl = 0.0
    for k in range(g):
        m = grp == k
        n_k = m.sum()
        if n_k == 0:
            continue
        o1 = y[m].sum()
        e1 = p[m].sum()
        e0 = n_k - e1
        o0 = n_k - o1
        # guard against zero expected counts
        t1 = (o1 - e1) ** 2 / e1 if e1 > 0 else 0.0
        t0 = (o0 - e0) ** 2 / e0 if e0 > 0 else 0.0
        hl += t1 + t0
    df = g - 2
    pval = 1 - stats.chi2.cdf(hl, df) if df > 0 else np.nan
    return dict(stat=hl, df=df, p=pval)

def calibration_intercept_slope(y, p):
    """
    Logistic recalibration: logit(P(event)) = a + b * logit(p_hat).
    a = 0, b = 1 means perfect calibration. Returns (intercept, slope).
    Uses GLM binomial with logit link (NOT OLS, which attenuates the slope).
    Falls back to L2-penalised logistic regression when separation makes the
    MLE diverge (|slope| > 20).
    """
    import statsmodels.api as sm
    y, p = as_arrays(y, p)
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    X = np.column_stack([np.ones_like(lp), lp])
    try:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0)
        a, b = float(res.params[0]), float(res.params[1])
        if np.isfinite(a) and np.isfinite(b) and abs(b) < 20:
            return a, b
    except Exception:
        pass
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000)
    lr.fit(X, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][1])

def brier_score(y, p):
    y, p = as_arrays(y, p)
    return float(np.mean((np.asarray(y, dtype=float) - p) ** 2))

# ----------------------------------------------------------------------------
# Decision Curve Analysis
# ----------------------------------------------------------------------------

def decision_curve(y, p, thresholds=None):
    """
    Net benefit for the model and for treat-all / treat-none strategies.
    Returns DataFrame with columns: threshold, nb_model, nb_all, nb_none.
    """
    y, p = as_arrays(y, p)
    y = np.asarray(y, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    prev = y.mean()
    rows = []
    for pt in thresholds:
        pred_pos = (p >= pt).astype(float)
        tp = np.sum(pred_pos * y)
        fp = np.sum(pred_pos * (1 - y))
        n = len(y)
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append((pt, nb_model, nb_all, 0.0))
    import pandas as pd
    return pd.DataFrame(rows, columns=["threshold", "nb_model", "nb_all", "nb_none"])

def dca_net_benefit_range(y, p):
    """Net-benefit-positive threshold range for a model (for head-to-head summary)."""
    dc = decision_curve(y, p, thresholds=np.arange(0.01, 0.99, 0.005))
    pos = dc[dc.nb_model > np.maximum(dc.nb_all, 0.0)]
    if len(pos) == 0:
        return np.nan, np.nan
    return float(pos.threshold.min()), float(pos.threshold.max())

# ----------------------------------------------------------------------------
# meta-analysis (DerSimonian-Laird)
# ----------------------------------------------------------------------------

def meta_dl(effects, ses, labels=None):
    """
    Random-effects (DerSimonian-Laird) meta-analysis.
    effects/ses: arrays of per-study effect estimates and standard errors.
    Returns dict with pooled effect, 95% CI, z, p, tau2, Q, I2.
    """
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(effects)
    if k == 1:
        return dict(pooled=effects[0], ci_low=effects[0] - 1.96 * ses[0],
                    ci_high=effects[0] + 1.96 * ses[0],
                    z=effects[0] / ses[0] if ses[0] > 0 else np.nan,
                    p=2 * (1 - stats.norm.cdf(abs(effects[0] / ses[0]))) if ses[0] > 0 else np.nan,
                    tau2=0.0, Q=np.nan, I2=np.nan, k=1)
    w = 1.0 / ses ** 2
    fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - fixed) ** 2)
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = 1.0 / (ses ** 2 + tau2)
    pooled = np.sum(wr * effects) / np.sum(wr)
    se = np.sqrt(1.0 / np.sum(wr))
    z = pooled / se
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return dict(pooled=pooled, ci_low=pooled - 1.96 * se, ci_high=pooled + 1.96 * se,
                z=z, p=2 * (1 - stats.norm.cdf(abs(z))), tau2=tau2, Q=Q, I2=I2, k=k)

# ----------------------------------------------------------------------------
# logistic regression helpers
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1e-4, maxiter=1000):
    """
    Penalised logistic regression via statsmodels GLM (binomial, L2 via alpha).
    X: 2-D array (design matrix incl. intercept column).
    Returns fitted result object.
    """
    import statsmodels.api as sm
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    # statsmodels GLM supports fit_regularized (elastic net). Use small alpha ~ ridge.
    res = model.fit_regularized(alpha=l2, L1_wt=0.0, maxiter=maxiter)
    return res

def predict_logistic(res, X):
    X = np.asarray(X, dtype=float)
    return np.asarray(res.predict(X), dtype=float)

def roc_auc_ci(y, score, seed=42, n_boot=2000):
    """AUC with bootstrap percentile CI (stratified)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    idx_ev = np.where(y == 1)[0]
    idx_ne = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(idx_ev, len(idx_ev), replace=True),
                            rng.choice(idx_ne, len(idx_ne), replace=True)])
        vals.append(auc_mannwhitney(y[b], score[b]))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(auc=auc_mannwhitney(y, score), ci_low=lo, ci_high=hi)
