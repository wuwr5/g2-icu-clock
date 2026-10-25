# g2-icu-clock

An **ICU-trained DNA methylation clock** (elastic-net logistic regression)
that predicts critical-illness severity and in-hospital mortality, with
head-to-head benchmarking against six community-trained epigenetic clocks
(Horvath2013, DNAmPhenoAge, GrimAge2, DunedinPACE, AltumAge, Hannum).

> This repository implements **G2** of the critical epigenetic-clock
> benchmarking project. Its thesis: instead of *migrating* a community-trained
> clock into the ICU, train the clock **for the target population** — the
> correction that critical-care scoping reviews explicitly call for
> (see References).

## Key finding

Training an elastic-net clock on COVID-19 severity (GSE179325, n=473,
113 severe) yields a clock that dominates every community clock:

| Cohort / outcome | Model | ΔAUC over age+sex | DeLong p |
|---|---|---|---|
| GSE179325 severe (out-of-fold) | **ICU clock** | **+0.192** | <1e-4 |
| GSE179325 severe | DunedinPACE (best community) | +0.051 | 0.018 |
| GSE174818 in-hospital death (external) | **ICU clock** | +0.094 | 0.116* |

\* External validation uses a **different outcome** (severity → death) on an
independent cohort and has only 25 events, hence the wider interval. On the
training cohort the ICU clock beats all six community clocks in paired DeLong
tests (**all p<0.0001**). Full tables in `results/`.

## Pipeline

1. `g2_train.py` — trains the ICU clock (nested 5×5 CV; elastic net;
   regularisation strength `C` chosen by inner CV), then scores both cohorts.
   Writes `results/G2_icu_clock_coefs.csv`, `G2_scores.csv`, `G2_train_cv.csv`.
2. `g2_evaluate.py` — head-to-head evaluation against the six community clocks
   via the shared `stats_core` / `incremental_framework` metrics
   (ΔAUC / DeLong, continuous NRI, IDI, calibration slope, DCA).
   Writes `results/G2_incremental.csv`, `G2_headtohead.csv`.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Processed inputs are **not** committed (public GEO datasets, produced
downstream of the project's G1 pipeline). Place them under `data/parsed/`:

| File | Format |
|---|---|
| `{GSE}_beta.parquet` | index = CpG id, columns = sample GSM, values = β (0–1) |
| `{GSE}_clocks.csv` | one row per GSM; must contain `age`, `sex`, `death`, `disease_state` (GSE179325), `subject_status` (cases contain "positive"), and six clock columns `{clock}_eaa` |

Cohorts: **GSE179325** (training, severity) and **GSE174818** (external
validation, in-hospital death).

To generate a small synthetic dataset for a smoke-test:

```bash
python example_data.py
python g2_train.py && python g2_evaluate.py
```

## Usage

```bash
python g2_train.py        # -> results/G2_icu_clock_coefs.csv, G2_scores.csv, G2_train_cv.csv
python g2_evaluate.py     # -> results/G2_incremental.csv, G2_headtohead.csv
```

Override input/output locations with the `PARSED` and `RESULTS` environment
variables.

## Repository layout

```
g2-icu-clock/
├── g2_train.py              # clock training (nested CV, elastic net)
├── g2_evaluate.py           # head-to-head evaluation vs 6 community clocks
├── stats_core.py            # self-implemented evaluation metrics
├── incremental_framework.py # shared design/outcome helpers (from G1)
├── example_data.py          # synthetic smoke-test data generator
├── requirements.txt
├── data/parsed/             # (gitignored) processed inputs
└── results/                # output tables (committed; real cohorts)
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Cheyne et al. *J Pers Med* 2026;16(2):92 — scoping review of biological
  clocks in critical care; notes community clocks are merely *migrated* into
  the ICU and calls for population-specific training.
- The full methods report (`G2_report.md`) lives in the upstream project.
