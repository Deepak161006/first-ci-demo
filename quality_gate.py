import json
import sys
from pathlib import Path

METRICS_PATH = Path("metrics.json")
MODEL_PATH = Path("house_price_model.pkl")

if not METRICS_PATH.exists():
    print("ERROR: metrics.json not found")
    sys.exit(1)

if not MODEL_PATH.exists():
    print("ERROR: house_price_model.pkl not found")
    sys.exit(1)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

r2_score = metrics.get("r2_score")
mae = metrics.get("mae")
rmse = metrics.get("rmse")

if r2_score is None or mae is None or rmse is None:
    print("ERROR: Required regression metrics are missing")
    sys.exit(1)

print("HOUSE PRICE MODEL QUALITY CHECK")
print("--------------------------------")
print("R2 Score:", round(r2_score, 4))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

if r2_score < 0:
    print("QUALITY GATE FAILED: R2 score is below zero")
    sys.exit(1)

print("QUALITY GATE PASSED")
