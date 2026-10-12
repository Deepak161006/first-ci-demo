
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "Housing.csv"
MODEL_PATH = "house_price_model.pkl"
METRICS_PATH = "metrics.json"


# --------------------------------------------------
# TRAIN HOUSE PRICE MODEL
# --------------------------------------------------

def train_model():

    # Load actual housing dataset
    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Dataset rows and columns:", data.shape)
    print("Columns:", data.columns.tolist())

    # Target and input features
    target = "price"

    if target not in data.columns:
        raise ValueError("Target column 'price' not found")

    X = data.drop(columns=[target])
    y = data[target]

    # Ensure Boolean features are handled numerically
    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Build regression pipeline
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate regression model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics
    metrics = {
        "model": "RandomForestRegressor",
        "dataset_rows": int(data.shape[0]),
        "dataset_columns": int(data.shape[1]),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nHOUSE PRICE MODEL RESULTS")
    print("-------------------------")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    train_model()
