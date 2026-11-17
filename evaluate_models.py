from pathlib import Path
import pandas as pd
from app import load_dataset, score_models

ROOT = Path(__file__).resolve().parent
df = load_dataset()
metrics, predictions, fitted, split, features = score_models(df)
metrics.to_csv(ROOT / "model_evaluation.csv", index=False)
predictions.to_csv(ROOT / "holdout_predictions.csv", index=False)
print(metrics.to_string(index=False))
print("\nSaved model_evaluation.csv and holdout_predictions.csv")
