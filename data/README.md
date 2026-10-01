# Input data contract

Use a UTF-8 CSV containing `Date` (ISO YYYY-MM-DD) and two or more asset columns.
Prices must be positive, complete, synchronous and ordered oldest first; dates must be unique.
At least 100 price rows are required (60 training and 20 holdout returns after splitting).

```csv
Date,Asset_A,Asset_B
2024-01-02,100.00,100.00
2024-01-03,101.20,99.90
2024-01-04,100.80,100.15
```

The short snippet only illustrates the format; it cannot be analysed on its own.

For real research, use **adjusted closing prices** or consistent total-return series in the same currency.
Check how the provider adjusts for splits and distributions. Raw closing prices can make corporate actions
look like investment losses. Do not silently fill missing prices or mix markets with incompatible calendars.
The program does not verify that observations are daily: the default 252 periods/year requires daily data.
For weekly/monthly series, change the frequency input and provide enough observations.

The default CLI generates fictional prices without a download. Its asset labels start with `Demo_`.
The process is a multivariate geometric Brownian model with constant parameters, a fixed random seed,
five fictional asset sleeves and 1,260 simulated daily returns. Dates are generic weekdays, not exchange sessions.

To supply provenance for a CSV called `prices.csv`, create `prices.metadata.json` alongside it:

```json
{
  "synthetic": false,
  "source": "Name of your authorised data source",
  "retrieved_on": "YYYY-MM-DD",
  "currency": "EUR",
  "price_type": "Adjusted close",
  "notes": "Describe adjustments, universe selection and missing-data handling"
}
```

Mark `synthetic: true` whenever the data are generated, even if loaded with `--prices`.
Do not commit licensed/private price files without permission. `data/private/` is ignored by Git.
