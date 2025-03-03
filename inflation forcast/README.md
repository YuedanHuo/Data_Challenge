# Inflation Forecasting Challenge: RAMP Starting Kit

![GitHub Actions](https://github.com/channdethsok/inflation_forecasting/actions/workflows/ci.yml/badge.svg)

This is the **RAMP starting kit** for the **Inflation Forecasting Challenge**, which focuses on predicting inflation using macroeconomic indicators from the **FRED dataset**.

## 📌 Challenge Overview

Inflation forecasting is a critical task in macroeconomics, impacting **monetary policy, financial markets, and business decision-making**. Accurate forecasts help:

- **Policymakers (e.g., central banks)** adjust interest rates and implement effective policies.
- **Investors** make informed decisions about asset allocation.
- **Businesses** develop pricing strategies and cost planning.

This challenge aims to **automate inflation prediction** by designing machine learning models that leverage **macroeconomic time-series data**.

---

## Getting Started

### Installation

To run a submission and the notebook, install the dependencies listed in `requirements.txt` using:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar usage.

```bash
conda env create -f environment.yml
conda activate ramp-inflation-forecast
```

### Challenge description

Get started on this RAMP with the
[dedicated notebook](inflation_forecast_notebook.ipynb).

This notebook guides you through:

- Loading and exploring macroeconomic data
- Time-series visualization
- Training a baseline model
- Evaluating model performance using MAE and RMSE

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
