# used-car-regression

TECH-UB 24 individual regression assignment: **should the dealer buy this car?**

A used car dealer buys at wholesale auction and resells on its lot. This repo fits a linear regression of lot sale price on the car's features and turns it into a buy / pass rule, then compares that rule with what the dealer does today (guessing the average sale price for every car).

## Run

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python used_car_model.py
```

One script, `used_car_model.py`, reproduces every number and chart:

| Step | What it does | Where the result goes |
|---|---|---|
| 1 | Random 70/30 split of the 240 sold cars (seed 42): 168 training, 72 test | printed |
| 2 | Linear regression on the training cars, with and without `Num_Owners` | printed (full statsmodels summaries) |
| 3 | Baseline vs v1 on the test cars: R-squared and RMSE in dollars | printed, `charts/predicted_vs_actual_test.png` |
| 4 | Buy / pass rule (`should_buy`) and the baseline rule on the 40 auction lots | printed, `outputs/auction_lot_decisions.csv`, `charts/auction_baseline_vs_v1.png` |
| 5 | Every headline number in one file | `outputs/results.json` |

`outputs/run_output.txt` is the full printed output of the last run. `NOTES.md` explains how each number is calculated.

## Data

- `data/used_car_sales.csv` — 240 cars the dealer sold in the last year (fit and test).
- `data/auction_lots.csv` — 40 cars from one recent auction, with the winning bid and what each later sold for. Never used to fit the model.
