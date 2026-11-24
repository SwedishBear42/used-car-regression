"""Used car prices: should the dealer buy this car?
TECH-UB 24, individual regression assignment (Nico Lupin).

This one script reproduces every number and chart on the slides.

    python used_car_model.py

Steps (same order as the assignment):
  1. Split the 240 sold cars 70/30 (fixed random seed).
  2. Fit a linear regression of sale price on the training cars; compare a version
     without the feature that does not earn its place; pick v1.
  3. Evaluate baseline vs v1 on the test cars (R-squared and RMSE in dollars).
  4. Turn v1 into a buy/pass rule and run it, and the baseline rule, on the 40 auction lots.
  5. Save the charts (charts/) and all numbers (outputs/results.json).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter

SEED = 42           # fixed so the split, and therefore every number, is reproducible
TEST_SHARE = 0.30
PREP_COST = 1500    # inspection, reconditioning, detailing, lot time: dollars per car

HERE = Path(__file__).resolve().parent
(HERE / "charts").mkdir(exist_ok=True)
(HERE / "outputs").mkdir(exist_ok=True)

sales = pd.read_csv(HERE / "data" / "used_car_sales.csv")   # 240 cars the dealer sold (fit + test)
lots = pd.read_csv(HERE / "data" / "auction_lots.csv")      # 40 auction cars (decision test only)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def r_squared(actual, predicted):
    """1 - (squared error of the predictions) / (squared error of guessing the mean of these cars)."""
    actual = np.asarray(actual, dtype=float)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(actual, predicted):
    """Root mean squared error: the typical miss, in dollars."""
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=float) - predicted) ** 2)))


# ----------------------------------------------------------------------------
# 1. Split the data
# ----------------------------------------------------------------------------
section("1. DATA AND SPLIT")
test = sales.sample(frac=TEST_SHARE, random_state=SEED)
train = sales.drop(test.index)
print(f"used_car_sales.csv: {len(sales)} cars, columns {list(sales.columns)}")
print(f"training cars: {len(train)} ({len(train) / len(sales):.0%})   test cars: {len(test)} ({len(test) / len(sales):.0%})   seed {SEED}")
print(f"auction_lots.csv: {len(lots)} cars (never used to fit)")

# ----------------------------------------------------------------------------
# 2. Fit the model on the training cars
# ----------------------------------------------------------------------------
section("2. FIT ON THE TRAINING CARS")
FULL = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident + Num_Owners"
REDUCED = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident"   # drops Num_Owners

full = smf.ols(FULL, data=train).fit()
reduced = smf.ols(REDUCED, data=train).fit()

print("--- All five features ---")
print(full.summary())
print("\n--- Without Num_Owners ---")
print(reduced.summary())


def coef_table(model):
    return pd.DataFrame({"coefficient": model.params.round(1), "p_value": model.pvalues.round(4)})


print("\nAll five features:\n", coef_table(full).to_string())
print(f"adjusted R-squared {full.rsquared_adj:.4f}   R-squared {full.rsquared:.4f}")
print("\nWithout Num_Owners:\n", coef_table(reduced).to_string())
print(f"adjusted R-squared {reduced.rsquared_adj:.4f}   R-squared {reduced.rsquared:.4f}")

# The version handed over as v1.
# Num_Owners has a p-value far above 0.05 once mileage and age are in the model, and dropping it
# does not lower adjusted R-squared, so v1 is the model without it. Change this line to `full`
# to hand over the five-feature version instead; everything below follows.
v1 = reduced
V1_NAME = "v1 (no Num_Owners)" if v1 is reduced else "v1 (all five features)"
print(f"\nChosen {V1_NAME}:  Num_Owners p-value {full.pvalues['Num_Owners']:.3f};  "
      f"adjusted R-squared {full.rsquared_adj:.4f} with it, {reduced.rsquared_adj:.4f} without it")

# ----------------------------------------------------------------------------
# 3. Evaluate fit on the test cars: baseline vs v1
# ----------------------------------------------------------------------------
section("3. FIT ON THE TEST CARS (never seen by the model): BASELINE vs v1")
BASELINE_PRICE = float(train["Sale_Price"].mean())      # the no-model guess: one price for every car
test_actual = test["Sale_Price"].to_numpy(dtype=float)
test_pred_baseline = np.full(len(test), BASELINE_PRICE)
test_pred_v1 = v1.predict(test).to_numpy()

fit_rows = {
    "baseline (training average)": {"r2": r_squared(test_actual, test_pred_baseline), "rmse": rmse(test_actual, test_pred_baseline)},
    V1_NAME: {"r2": r_squared(test_actual, test_pred_v1), "rmse": rmse(test_actual, test_pred_v1)},
}
print(f"baseline prediction for every car = average training sale price = ${BASELINE_PRICE:,.0f}")
for name, m in fit_rows.items():
    print(f"{name:<30} R-squared {m['r2']:>7.3f}    typical error (RMSE) ${m['rmse']:>8,.0f}")


# ----------------------------------------------------------------------------
# 4. The decision rule
# ----------------------------------------------------------------------------
def predict_sale_price(mileage_k, age_years, horsepower, accident, num_owners):
    """v1's predicted lot sale price for one car. `accident` is "yes" or "no"."""
    car = pd.DataFrame([{"Mileage_K": mileage_k, "Age_Years": age_years, "Horsepower": horsepower,
                         "Accident": accident, "Num_Owners": num_owners}])
    return float(v1.predict(car).iloc[0])


def should_buy(auction_price, mileage_k, age_years, horsepower, accident, num_owners):
    """The dealer's rule with the model: buy when auction price + prep cost is below the predicted sale price."""
    predicted = predict_sale_price(mileage_k, age_years, horsepower, accident, num_owners)
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def should_buy_baseline(auction_price):
    """The dealer's rule with no model: buy when auction price + prep cost is below the average sale price."""
    return "buy" if auction_price + PREP_COST < BASELINE_PRICE else "pass"


section("4. DECISION RULE ON THE 40 AUCTION LOTS")
lots = lots.copy()
lots["Predicted_Sale_Price"] = [predict_sale_price(r.Mileage_K, r.Age_Years, r.Horsepower, r.Accident, r.Num_Owners)
                                for r in lots.itertuples()]
lots["Decision_v1"] = [should_buy(r.Auction_Price, r.Mileage_K, r.Age_Years, r.Horsepower, r.Accident, r.Num_Owners)
                       for r in lots.itertuples()]
lots["Decision_baseline"] = [should_buy_baseline(p) for p in lots["Auction_Price"]]
lots["Profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST


def rule_result(decisions):
    bought = lots[decisions == "buy"]
    return {"cars_bought": int(len(bought)),
            "money_losing_cars": int((bought["Profit_if_bought"] < 0).sum()),
            "total_profit": float(bought["Profit_if_bought"].sum())}


decision_rows = {"baseline": rule_result(lots["Decision_baseline"]), "v1": rule_result(lots["Decision_v1"])}
for name, d in decision_rows.items():
    print(f"{name:<9} buys {d['cars_bought']:>2} of {len(lots)} cars; {d['money_losing_cars']:>2} of those lose money; "
          f"total profit ${d['total_profit']:>9,.0f}")
every = {"cars_bought": len(lots), "money_losing_cars": int((lots["Profit_if_bought"] < 0).sum()),
         "total_profit": float(lots["Profit_if_bought"].sum())}
print(f"(reference, not a rule: buying all {len(lots)} lots -> {every['money_losing_cars']} lose money, total profit ${every['total_profit']:,.0f})")
print(f"\nv1 total profit minus baseline total profit = ${decision_rows['v1']['total_profit'] - decision_rows['baseline']['total_profit']:,.0f} on this one auction")

print("\nLot-by-lot detail:")
print(lots[["Lot", "Auction_Price", "Predicted_Sale_Price", "Later_Sale_Price", "Profit_if_bought",
            "Decision_baseline", "Decision_v1"]].round(0).to_string(index=False))
lots.round(2).to_csv(HERE / "outputs" / "auction_lot_decisions.csv", index=False)

# one worked example for the slide / for explaining the rule
ex = lots.iloc[0]
print(f"\nWorked example, lot {int(ex.Lot)}: auction ${ex.Auction_Price:,.0f} + prep ${PREP_COST:,} = ${ex.Auction_Price + PREP_COST:,.0f}; "
      f"v1 predicts ${ex.Predicted_Sale_Price:,.0f} -> {ex.Decision_v1};  baseline compares with ${BASELINE_PRICE:,.0f} -> {ex.Decision_baseline};  "
      f"it later sold for ${ex.Later_Sale_Price:,.0f} (profit if bought ${ex.Profit_if_bought:,.0f})")

# ----------------------------------------------------------------------------
# 5. Charts and a machine-readable copy of every number
# ----------------------------------------------------------------------------
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
C_BASELINE, C_V1 = "#2a78d6", "#eb6834"      # fixed: baseline is always blue, v1 always orange
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
                     "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
dollars_k = FuncFormatter(lambda v, _: "$0" if v == 0 else f"${v / 1000:,.0f}K")


def money(v):
    """$15,800 or -$6,000 (minus sign before the dollar sign)."""
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

# Chart 1: predicted vs actual sale price, test cars
fig, ax = plt.subplots(figsize=(7.2, 5.6))
lo, hi = 0, max(test_actual.max(), test_pred_v1.max()) * 1.06
ax.plot([lo, hi], [lo, hi], color="#9c9b96", linewidth=1, zorder=1)
ax.text(hi * 0.985, hi * 0.985, "perfect prediction ", ha="right", va="top", color=INK_2, fontsize=9.5, rotation=0)
ax.axhline(BASELINE_PRICE, color=C_BASELINE, linewidth=2, zorder=2, label=f"Baseline: ${BASELINE_PRICE:,.0f} for every car")
ax.scatter(test_actual, test_pred_v1, s=46, color=C_V1, edgecolor=SURFACE, linewidth=1.4, zorder=3,
           label="Model v1: one prediction per car")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual sale price"); ax.set_ylabel("Predicted sale price")
ax.xaxis.set_major_formatter(dollars_k); ax.yaxis.set_major_formatter(dollars_k)
ax.grid(color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_title("Predicted vs actual sale price (test set)", loc="left", fontsize=12.5, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "charts" / "predicted_vs_actual_test.png", dpi=200); plt.close(fig)

# Chart 2: the auction, baseline rule vs v1 rule (three measures -> three small panels, one axis each)
panels = [("Cars bought", "cars_bought", lambda v: f"{v:,.0f}"),
          ("Money-losing cars bought", "money_losing_cars", lambda v: f"{v:,.0f}"),
          ("Total profit", "total_profit", money)]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0))
for ax, (title, key, fmt) in zip(axes, panels):
    vals = [decision_rows["baseline"][key], decision_rows["v1"][key]]
    ax.bar(["Baseline", "Model v1"], vals, width=0.22, color=[C_BASELINE, C_V1], zorder=3)
    top = max(max(vals), 1)
    low = min(min(vals), 0)
    span = top - low
    # leave room above the tallest bar, and below a negative bar, for the value labels
    ax.set_ylim(low - span * 0.22 if low < 0 else 0, top + span * 0.18)
    for x, v in enumerate(vals):
        ax.text(x, v + (span * 0.03 if v >= 0 else -span * 0.03), fmt(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=12, color=INK, fontweight="bold")
    ax.axhline(0, color="#9c9b96", linewidth=1, zorder=2)
    ax.set_title(title, loc="left", fontsize=11.5, pad=10)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10.5); ax.margins(x=0.35)
fig.suptitle("Auction lots: baseline vs model v1", x=0.012, ha="left", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(HERE / "charts" / "auction_baseline_vs_v1.png", dpi=200); plt.close(fig)

results = {
    "seed": SEED, "n_train": int(len(train)), "n_test": int(len(test)), "n_auction_lots": int(len(lots)), "prep_cost": PREP_COST,
    "v1_formula": REDUCED if v1 is reduced else FULL,
    "v1_coefficients": {k: round(float(v), 2) for k, v in v1.params.items()},
    "v1_p_values": {k: round(float(v), 4) for k, v in v1.pvalues.items()},
    "v1_adj_r2_train": round(float(v1.rsquared_adj), 4),
    "full_model_adj_r2_train": round(float(full.rsquared_adj), 4),
    "num_owners_p_value_in_full_model": round(float(full.pvalues["Num_Owners"]), 4),
    "baseline_price": round(BASELINE_PRICE, 2),
    "test_fit": {k: {"r2": round(v["r2"], 4), "rmse": round(v["rmse"], 0)} for k, v in fit_rows.items()},
    "auction": decision_rows, "auction_buy_everything_reference": every,
}
(HERE / "outputs" / "results.json").write_text(json.dumps(results, indent=2))
print("\nsaved charts/predicted_vs_actual_test.png, charts/auction_baseline_vs_v1.png, outputs/results.json, outputs/auction_lot_decisions.csv")
