from datetime import datetime, timedelta
import math

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .settings import ML_MIN_TRAINING_DAYS, ML_MODEL, ML_RANDOM_STATE
from .transform import LOCAL_TZ

TARGETS = {
    "temperatura_media_c": "temperatura_media_c",
    "temperatura_minima_c": "temperatura_minima_c",
    "temperatura_maxima_c": "temperatura_maxima_c",
    "umidade_media_pct": "umidade_media_pct",
    "precipitacao_total_mm": "precipitacao_total_mm",
}
LAGS = (1, 2, 3, 7, 14)
RAIN_THRESHOLD_MM = 0.1


def _feature_frame(daily_docs, end_date=None):
    frame = pd.DataFrame(daily_docs)
    if frame.empty or "data_local" not in frame:
        raise ValueError("Não há histórico diário suficiente para treinar o modelo.")

    frame["data_local"] = pd.to_datetime(frame["data_local"], errors="coerce")
    frame = frame.dropna(subset=["data_local"]).sort_values("data_local")
    frame = frame.drop_duplicates("data_local").set_index("data_local")
    last_date = frame.index.max()
    if end_date is not None:
        last_date = max(last_date, pd.Timestamp(end_date))
    frame = frame.reindex(pd.date_range(frame.index.min(), last_date, freq="D"))

    for target in TARGETS.values():
        frame[target] = pd.to_numeric(frame.get(target), errors="coerce")
        frame[target] = frame[target].interpolate(limit_direction="both").ffill().bfill()

    features = pd.DataFrame(index=frame.index)
    day_of_year = features.index.dayofyear
    features["dia_seno"] = (day_of_year * 2 * math.pi / 365.25).map(math.sin)
    features["dia_cosseno"] = (day_of_year * 2 * math.pi / 365.25).map(math.cos)
    for target in TARGETS.values():
        for lag in LAGS:
            features[f"{target}_lag_{lag}"] = frame[target].shift(lag)
        features[f"{target}_media_7"] = frame[target].rolling(7).mean().shift(1)
    return frame, features


def _predict_random_forest(daily_docs, horizon_days, as_of_date):
    frame, features = _feature_frame(daily_docs, as_of_date)
    if len(frame) < ML_MIN_TRAINING_DAYS:
        raise ValueError(
            f"Histórico insuficiente: {len(frame)} dias; "
            f"mínimo configurado: {ML_MIN_TRAINING_DAYS}."
        )

    predictions = []
    working = frame.copy()
    feature_columns = list(features.columns)
    models = {}
    validation_mae = {}
    for target in TARGETS.values():
        train = pd.concat([features, frame[target].rename("target")], axis=1).dropna()
        validation_size = min(30, max(7, len(train) // 5))
        train_part = train.iloc[:-validation_size]
        validation_part = train.iloc[-validation_size:]
        validation_model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        )
        validation_model.fit(train_part[feature_columns], train_part["target"])
        validation_mae[target] = round(float(mean_absolute_error(
            validation_part["target"],
            validation_model.predict(validation_part[feature_columns]),
        )), 3)
        model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[feature_columns], train["target"])
        models[target] = model

    next_day = working.index.max() + timedelta(days=1)
    for offset in range(horizon_days):
        current_day = next_day + timedelta(days=offset)
        working.loc[current_day, :] = None
        for target, model in models.items():
            row = _future_features(working, current_day)
            value = float(model.predict(row[feature_columns])[0])
            if target == "precipitacao_total_mm":
                value = max(0.0, value)
            if target == "umidade_media_pct":
                value = min(100.0, max(0.0, value))
            working.loc[current_day, target] = value
        values = {
            target: round(float(working.loc[current_day, target]), 2)
            for target in TARGETS.values()
        }
        predictions.append({
            "data_local": current_day.date().isoformat(),
            **values,
            "chuva_provavel": values["precipitacao_total_mm"] >= RAIN_THRESHOLD_MM,
        })
    return predictions, validation_mae


def _future_features(frame, current_day):
    row = pd.DataFrame(index=[current_day])
    day_of_year = current_day.dayofyear
    angle = day_of_year * 2 * math.pi / 365.25
    row["dia_seno"] = math.sin(angle)
    row["dia_cosseno"] = math.cos(angle)
    for target in TARGETS.values():
        for lag in LAGS:
            row[f"{target}_lag_{lag}"] = frame[target].shift(lag).loc[current_day]
        row[f"{target}_media_7"] = frame[target].rolling(7).mean().shift(1).loc[current_day]
    return row


def predict(daily_docs, horizon_days):
    if ML_MODEL != "random_forest":
        raise ValueError(
            f"ML_MODEL={ML_MODEL!r} ainda não está implementado; use 'random_forest'."
        )
    today = datetime.now(LOCAL_TZ).date().isoformat()
    usable_docs = [
        doc for doc in daily_docs
        if str(doc.get("data_local", "")) <= today
    ]
    predictions, validation_mae = _predict_random_forest(usable_docs, horizon_days, today)
    return {
        "modelo": "random_forest",
        "data_referencia": today,
        "variaveis": list(TARGETS),
        "horizonte_dias": horizon_days,
        "previsoes": predictions,
        "validacao": {"mae_por_variavel": validation_mae},
        "treinado_com_dias": len(usable_docs),
    }