# Thinking in Models

![Thinking in Models](figures/Thinking%20in%20Models.png)

Applied Bayesian statistics and causal inference across real-world domains: real estate markets, crime, forensic audio analysis, macroeconomics, and public governance.

Inspired by Richard McElreath's [Statistical Rethinking](https://www.youtube.com/playlist?list=PLDcUM9US4XdNOlqSyhe38US8mFgmqzI14) course and [book](https://www.amazon.com/Statistical-Rethinking-Bayesian-Examples-Chapman-dp-036713991X/dp/036713991X/ref=dp_ob_title_bk). The original course teaches Bayesian workflow through simulation and directed acyclic graphs. This project applies that framework to Slovenian public data, demonstrating how the same principles work when the data is messy, the stakes are real, and the models need to inform decisions.

All code is in Python. All data is public.

## Structure

```
revised_lectures/   Lecture notes with Python code and applied examples
notebooks/          Worked examples and exercises as Jupyter notebooks
data/               Teaching datasets derived from Slovenian public sources
thinkmod/           Python library for Bayesian workflow utilities (pip-installable)
```

## Datasets

Six datasets derived from Slovenian public sources (GURS, SURS, ERAR, Police). Full documentation in [`data/DATA_CATALOG.md`](data/DATA_CATALOG.md).

| Dataset | Rows | Source | Domain |
|---------|------|--------|--------|
| `sr2026_real_estate.csv` | 191,097 | GURS ETN (2007-2025) | Property transactions: price, area, location, type, construction year |
| `sr2026_municipality_panel.csv` | 424 | SURS, GURS | Municipality-level salary, prices, building permits (2023-2024) |
| `sr2026_crime.csv` | 915 | Slovenian Police (2010-2024) | Reported offenses by administrative unit and year |
| `sr2026_forensic_audio.csv` | 200 | Synthetic | Audio recordings with ENF, spectral, and pause features for authenticity classification |
| `sr2026_economic_pulse.csv` | 149 | SURS, Eurostat, UMAR | 13 macroeconomic indicators (1995-2026), mixed frequency |
| `sr2026_governance.csv` | 4,644 | ERAR (KPK RS) | Post-employment restrictions on public officials and business entities |

### Quick start

```python
import pandas as pd

re = pd.read_csv("data/sr2026_real_estate.csv", parse_dates=["date"])
lj_apt = re[(re["municipality"] == "LJUBLJANA") & (re["property_type"] == "apartment")]
print(f"Ljubljana apartments: {len(lj_apt):,} transactions")
print(f"Median price/m2: EUR {lj_apt['price_per_m2'].median():,.0f}")
```

## Lectures

| # | Topic |
|---|-------|
| A01 | Introduction to Bayesian Workflow |
| A02 | Bayesian Inference Foundations |
| A03 | Geocentric Models |
| A04 | Categories and Causes |
| A05 | Estimands and Estiplans |
| A06 | Elemental Confounds I |
| A07 | Good and Bad Controls |

## thinkmod

`thinkmod` is a Python library that packages the reusable components from this course: DAG utilities, data loaders, model templates, and plotting helpers. Install in development mode:

```bash
pip install -e .
```

## License

Code: MIT. Datasets: source data licensed under CC BY 4.0 (Slovenian public data). Forensic audio dataset is synthetic.

## Author

Niko Gamulin, PhD
