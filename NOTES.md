# How every number is calculated

Working notes behind `used_car_model.py`. All numbers come from `outputs/run_output.txt` and `outputs/results.json` (seed 42).

## 1. Data and split
- 240 sold cars. `sales.sample(frac=0.30, random_state=42)` picks 72 test cars at random. The other 168 are the training cars.
- The 40 auction lots are a separate file. They are only used in step 4.

## 2. The fit (training cars only)
Model with all five features:

| Term | Coefficient | p-value |
|---|---|---|
| Intercept | 21,698.6 | < 0.001 |
| Accident = yes | -2,171.8 | < 0.001 |
| Mileage_K | -78.6 | < 0.001 |
| Age_Years | -1,300.6 | < 0.001 |
| Horsepower | 75.0 | < 0.001 |
| Num_Owners | 62.9 | 0.865 |

Adjusted R-squared 0.8691.

Model without `Num_Owners` (this is **v1**):

| Term | Coefficient | p-value | Reading, holding the other features fixed |
|---|---|---|---|
| Intercept | 21,762.4 | < 0.001 | starting point of the line, not a real car |
| Accident = yes | -2,176.6 | < 0.001 | an accident on the history report: about $2,177 lower sale price |
| Mileage_K | -78.2 | < 0.001 | each extra 1,000 miles: about $78 lower |
| Age_Years | -1,291.6 | < 0.001 | each extra year of age: about $1,292 lower |
| Horsepower | 75.0 | < 0.001 | each extra horsepower: about $75 higher |

Adjusted R-squared 0.8698.

Why `Num_Owners` was dropped: its p-value is 0.865, so once mileage and age are in the model the data cannot tell its effect apart from zero. Owners, mileage and age move together (correlations of about 0.7 with each). Adjusted R-squared does not fall without it (0.8691 with, 0.8698 without). To hand over the five-feature version instead, change `v1 = reduced` to `v1 = full` in the script.

- **Adjusted R-squared** is the share of the variation in sale price explained on the training cars, with a penalty for each extra feature.

## 3. Fit on the 72 test cars
- **Baseline** predicts the same price for every car: the average training sale price, **$25,851**.
- **R-squared on the test cars** = 1 − (sum of squared prediction errors) ÷ (sum of squared gaps between each test car and the test-car average). The baseline scores slightly below zero (−0.014) because the training average is not exactly the test average.
- **RMSE** = square root of the average squared error. It is the typical miss in dollars.

| | R-squared (test) | RMSE (test) |
|---|---|---|
| Baseline | -0.014 | $6,494 |
| v1 | 0.823 | $2,716 |

Chart: `charts/predicted_vs_actual_test.png`.

## 4. The decision on the 40 auction lots
- v1 rule (`should_buy`): **buy** when `Auction_Price + 1,500 < predicted sale price`, otherwise pass.
- Baseline rule (`should_buy_baseline`): **buy** when `Auction_Price + 1,500 < 25,851`.
- Profit on a bought car = `Later_Sale_Price − Auction_Price − 1,500`. A car "loses money" when that is below zero.

| | Cars bought | Money-losing cars bought | Total profit |
|---|---|---|---|
| Baseline | 14 | 10 | -$6,000 |
| v1 | 17 | 6 | $15,800 |

Difference in total profit: $15,800 − (−$6,000) = **$21,800** on this one auction.
Reference only: buying all 40 lots loses money on 27 of them, for a total of −$35,500.

Worked example, lot 1: auction $21,500 + $1,500 = $23,000. v1 predicts $17,466, so pass. The baseline compares $23,000 with $25,851, so buy. The car later sold for $19,000, a $4,000 loss if bought.

Lot-by-lot table: `outputs/auction_lot_decisions.csv`. Chart: `charts/auction_baseline_vs_v1.png`.

## 5. How much the split matters
The seed is fixed so the slides can be reproduced. Rerunning with 100 other random splits (not in the script): test R-squared ranged 0.75 to 0.90, v1 profit ranged $10,400 to $17,700, and v1 beat the baseline rule in all 100. Seed 42 sits near the middle.

## What the assignment asks for on the slides
Title slide: name and the GitHub link. Then two slides covering: data (source, rows, period, the 70/30 split); model specification (outcome, features, family, the dropped feature, adjusted R-squared of the training fit); baseline vs v1 on the test cars (R-squared and RMSE, with the chart); the business decision and value (one-sentence headline, cars bought, money-losing cars, total profit, baseline vs v1). The assignment says the slides must be written by the student, not by AI.
