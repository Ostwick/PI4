from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.mongo_repository import WeatherRepository
from src.settings import INMET_STATION_CODE
from src.transform import LOCAL_TZ


TARGETS = [
    "temperatura_media_c",
    "temperatura_minima_c",
    "temperatura_maxima_c",
    "umidade_media_pct",
    "precipitacao_total_mm",
]


def _load_daily_history():
    repo = WeatherRepository()
    try:
        docs = repo.get_daily_history(INMET_STATION_CODE)
        if not docs:
            raise RuntimeError("Sem histórico diário no MongoDB para comparar modelos.")
        frame = pd.DataFrame(docs)
        frame["data_local"] = pd.to_datetime(frame["data_local"], errors="coerce")
        frame = frame.dropna(subset=["data_local"]).sort_values("data_local")
        for column in TARGETS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.ffill().bfill()
        return frame.set_index("data_local")
    finally:
        repo.close()


def _features(frame):
    features = pd.DataFrame(index=frame.index)
    day_of_year = features.index.dayofyear
    features["dia_seno"] = (day_of_year * 2 * math.pi / 365.25).map(math.sin)
    features["dia_cosseno"] = (day_of_year * 2 * math.pi / 365.25).map(math.cos)
    for target in TARGETS:
        for lag in (1, 2, 3, 7, 14):
            features[f"{target}_lag_{lag}"] = frame[target].shift(lag)
        features[f"{target}_media_7"] = frame[target].rolling(7).mean().shift(1)
    return features


def _random_forest_score(frame):
    features = _features(frame)
    scores = {}
    rows = pd.concat([features, frame], axis=1).dropna()
    split = max(7, len(rows) // 5)
    train = rows.iloc[:-split]
    valid = rows.iloc[-split:]
    X_train = train[features.columns]
    X_valid = valid[features.columns]

    for target in TARGETS:
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, train[target])
        pred = model.predict(X_valid)
        scores[target] = float(mean_absolute_error(valid[target], pred))
    return scores


def _xgboost_score(frame):
    features = _features(frame)
    scores = {}
    rows = pd.concat([features, frame], axis=1).dropna()
    split = max(7, len(rows) // 5)
    train = rows.iloc[:-split]
    valid = rows.iloc[-split:]
    X_train = train[features.columns]
    X_valid = valid[features.columns]

    for target in TARGETS:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
        model.fit(X_train, train[target])
        pred = model.predict(X_valid)
        scores[target] = float(mean_absolute_error(valid[target], pred))
    return scores


def _sarimax_score(frame):
    scores = {}
    split = max(7, len(frame) // 5)
    train = frame.iloc[:-split]
    valid = frame.iloc[-split:]

    for target in TARGETS:
        try:
            train_series = train[target].asfreq("D").ffill().bfill().dropna()
            valid_series = valid[target].asfreq("D").ffill().bfill().dropna()
            if len(train_series) < 30 or len(valid_series) == 0:
                scores[target] = float("inf")
                continue
            train_series = train_series.asfreq("D")
            train_series.index = pd.DatetimeIndex(train_series.index, freq="D")
            model = SARIMAX(
                train_series,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, method_kwargs={"warn_convergence": False})
            pred = fitted.forecast(len(valid_series))
            scores[target] = float(mean_absolute_error(valid_series, pred))
        except Exception:
            scores[target] = float("inf")
    return scores


def _aggregate_scores(scores_by_model):
    aggregated = {}
    for name, metric_by_target in scores_by_model.items():
        aggregated[name] = float(sum(metric_by_target.values()) / max(len(metric_by_target), 1))
    return aggregated


def main():
    frame = _load_daily_history()
    models = {
        "RandomForest": _random_forest_score,
        "XGBoost": _xgboost_score,
        "SARIMAX": _sarimax_score,
    }

    scores_by_model = {}
    for name, scorer in models.items():
        scores = scorer(frame)
        scores_by_model[name] = scores
        print(f"\n=== {name} ===")
        for target, value in scores.items():
            print(f"{target}: MAE={value:.4f}")
        print(f"Média geral: {sum(scores.values()) / len(scores):.4f}")

    summary = _aggregate_scores(scores_by_model)
    print("\n=== Ranking ===")
    for name, value in sorted(summary.items(), key=lambda item: item[1]):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
