"""Train and evaluate the Spotify popularity model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "spotify_model.pkl"
FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "liveness",
]
TARGET = "popularity"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV and check that all columns used by the model exist."""
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df[required].isna().any().any():
        raise ValueError("Dataset has missing values in the model features or target.")
    return df


def evaluate_model(model, df: pd.DataFrame) -> tuple[float, float]:
    """Evaluate on the same held-out split used during training."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    # Refit a supplied model only when it has not already been fitted.
    if not hasattr(model, "estimators_"):
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions), mean_absolute_error(y_test, predictions)


def train_and_save_model(
    dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH
) -> tuple[RandomForestRegressor, float, float]:
    """Train the model, save it beside the app, and return holdout metrics."""
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, r2, mae


def main() -> None:
    model, r2, mae = train_and_save_model()
    print(f"R2 Score: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
