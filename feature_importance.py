from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from train_model import FEATURES, MODEL_PATH

# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
    )

model = joblib.load(MODEL_PATH)

# Importance values
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

# Sort
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

# Plot
plt.figure(figsize=(8,5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()

output_path = Path(__file__).resolve().parent / "feature_importance.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Feature importance chart saved to {output_path}")
