from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET_COLUMN = "posted_rate"

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

BASE_FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]


def ensure_positive(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1.0, None)


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in BASE_FEATURES:
        if column not in result.columns:
            result[column] = np.nan

    parsed_date = pd.to_datetime(result["date"], errors="coerce")
    if parsed_date.isna().any():
        missing_count = int(parsed_date.isna().sum())
        raise ValueError(f"Found {missing_count} invalid date values in input data")

    iso_week = parsed_date.dt.isocalendar().week.astype(int)
    result["day_of_week"] = parsed_date.dt.dayofweek
    result["day_of_month"] = parsed_date.dt.day
    result["month"] = parsed_date.dt.month
    result["week_of_year"] = iso_week

    # Keep original raw features and append calendar signal columns.
    return result[BASE_FEATURES + ["day_of_week", "day_of_month", "month", "week_of_year"]]


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=450,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def temporal_train_validation_split(frame: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.copy()
    sorted_frame["_date"] = pd.to_datetime(sorted_frame["date"], errors="coerce")
    if sorted_frame["_date"].isna().any():
        raise ValueError("Training data contains invalid date values")

    unique_dates = sorted(sorted_frame["_date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates to create a temporal validation split")

    split_index = int(len(unique_dates) * train_ratio)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)

    train_dates = set(unique_dates[:split_index])
    train_frame = sorted_frame[sorted_frame["_date"].isin(train_dates)].drop(columns=["_date"])
    valid_frame = sorted_frame[~sorted_frame["_date"].isin(train_dates)].drop(columns=["_date"])

    return train_frame, valid_frame


def evaluate_model(pipeline: Pipeline, train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> dict[str, float]:
    x_train = add_date_features(train_frame)
    y_train = train_frame[TARGET_COLUMN].astype(float)

    x_valid = add_date_features(valid_frame)
    y_valid = valid_frame[TARGET_COLUMN].astype(float)

    pipeline.fit(x_train, y_train)
    predictions = ensure_positive(pipeline.predict(x_valid))

    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    mape = float(np.mean(np.abs((y_valid - predictions) / y_valid)) * 100)

    return {
        "validation_rows": float(len(valid_frame)),
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
    }


def write_validation_predictions(
    validation_frame: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
    template_path: Path | None = None,
) -> None:
    predicted = pd.DataFrame(
        {
            "load_id": validation_frame["load_id"].astype(str),
            "predicted_rate": predictions,
        }
    )

    if template_path is not None and template_path.is_file():
        template = pd.read_csv(template_path)
        if list(template.columns) != ["load_id", "predicted_rate"]:
            raise ValueError("Validation template must contain exactly: load_id,predicted_rate")

        template["load_id"] = template["load_id"].astype(str)
        pred_map = predicted.set_index("load_id")["predicted_rate"]
        template["predicted_rate"] = template["load_id"].map(pred_map)
        if template["predicted_rate"].isna().any():
            missing = int(template["predicted_rate"].isna().sum())
            raise ValueError(f"Template contains {missing} load_id values missing from validation input")
        template["predicted_rate"] = template["predicted_rate"].round(2)
        template.to_csv(output_path, index=False)
        return

    predicted["predicted_rate"] = predicted["predicted_rate"].round(2)
    predicted.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train freight rate model and create submission files")
    parser.add_argument("--train", default="train-test.csv", help="Path to train/test CSV")
    parser.add_argument("--validation", default="validation.csv", help="Path to validation feature CSV")
    parser.add_argument(
        "--validation-template",
        default="validation-predictions-template.csv",
        help="Path to validation template CSV",
    )
    parser.add_argument(
        "--validation-output",
        default="validation_predictions.csv",
        help="Where to write validation predictions",
    )
    parser.add_argument(
        "--december-input",
        default="december-chart-inputs.csv",
        help="Path to december chart input CSV",
    )
    parser.add_argument(
        "--metrics-output",
        default="model_metrics.json",
        help="Where to write validation metrics",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    validation_path = Path(args.validation)
    validation_template_path = Path(args.validation_template)
    validation_output_path = Path(args.validation_output)
    december_input_path = Path(args.december_input)
    metrics_output_path = Path(args.metrics_output)

    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)
    december_frame = pd.read_csv(december_input_path)

    if TARGET_COLUMN not in train_frame.columns:
        raise ValueError(f"Training CSV must contain target column: {TARGET_COLUMN}")
    if "load_id" not in validation_frame.columns:
        raise ValueError("Validation CSV must contain load_id column")

    train_split, valid_split = temporal_train_validation_split(train_frame, train_ratio=0.8)

    base_pipeline = build_pipeline()
    metrics = evaluate_model(base_pipeline, train_split, valid_split)

    final_pipeline = clone(base_pipeline)
    final_pipeline.fit(add_date_features(train_frame), train_frame[TARGET_COLUMN].astype(float))

    validation_predictions = ensure_positive(final_pipeline.predict(add_date_features(validation_frame)))
    write_validation_predictions(
        validation_frame=validation_frame,
        predictions=validation_predictions,
        output_path=validation_output_path,
        template_path=validation_template_path,
    )

    december_predictions = ensure_positive(final_pipeline.predict(add_date_features(december_frame)))
    december_output = december_frame.copy()
    december_output["predicted_rate"] = np.round(december_predictions, 2)
    december_output.to_csv(december_input_path, index=False)

    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"- {validation_output_path}")
    print(f"- {december_input_path}")
    print(f"- {metrics_output_path}")
    print("Validation summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
