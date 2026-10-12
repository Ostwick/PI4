from datetime import datetime, timedelta, timezone
import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .settings import ML_RANDOM_STATE

LAGS = (1, 2, 3, 6, 12, 24)
RAIN_THRESHOLD_MM = 0.1


def _features(frame):
    features = pd.DataFrame(index=frame.index)
    hour = features.index.hour
    day_of_year = features.index.dayofyear
    features["hora_seno"] = (hour * 2 * math.pi / 24).map(math.sin)
    features["hora_cosseno"] = (hour * 2 * math.pi / 24).map(math.cos)
    features["dia_seno"] = (day_of_year * 2 * math.pi / 365.25).map(math.sin)
    features["dia_cosseno"] = (day_of_year * 2 * math.pi / 365.25).map(math.cos)
    for target in ("temperatura_c", "umidade_pct", "precipitacao_mm"):
        for lag in LAGS:
            features[f"{target}_lag_{lag}"] = frame[target].shift(lag)
    return features


def predict_rain_next_hours(hourly_docs, hours=24):
    frame = pd.DataFrame(hourly_docs)
    if frame.empty or "data_hora_utc" not in frame:
        raise ValueError("Não há histórico horário para prever chuva.")

    frame["data_hora_utc"] = pd.to_datetime(frame["data_hora_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["data_hora_utc"])
    frame = frame[frame["data_hora_utc"] <= pd.Timestamp.now(tz="UTC")]
    frame = frame.sort_values("data_hora_utc").drop_duplicates("data_hora_utc")
    frame = frame.set_index("data_hora_utc")
    frame = frame.reindex(pd.date_range(frame.index.min(), pd.Timestamp.now(tz="UTC").floor("h"), freq="h"))
    for target in ("temperatura_c", "umidade_pct", "precipitacao_mm"):
        frame[target] = pd.to_numeric(frame.get(target), errors="coerce")
        frame[target] = frame[target].interpolate(limit_direction="both").ffill().bfill()

    features = _features(frame)
    target = (frame["precipitacao_mm"] > RAIN_THRESHOLD_MM).astype(int)
    training = features.assign(chuva=target).dropna()
    if len(training) < 100 or training["chuva"].nunique() < 2:
        raise ValueError("Histórico horário insuficiente para classificar chuva.")

    model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=ML_RANDOM_STATE,
        n_jobs=-1,
    )
    feature_columns = list(features.columns)
    model.fit(training[feature_columns], training["chuva"])

    regressors = {}
    for target_name in ("temperatura_c", "umidade_pct"):
        regression_training = features.assign(target=frame[target_name]).dropna()
        regressor = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        )
        regressor.fit(regression_training[feature_columns], regression_training["target"])
        regressors[target_name] = regressor

    working = frame.copy()
    next_hour = working.index.max() + timedelta(hours=1)
    predictions = []
    for offset in range(hours):
        current_hour = next_hour + timedelta(hours=offset)
        working.loc[current_hour, :] = None
        row = _features(working).loc[[current_hour], feature_columns]
        probability = float(model.predict_proba(row)[0][1])
        rain = probability >= 0.35
        temperature = float(regressors["temperatura_c"].predict(row)[0])
        humidity = min(100.0, max(0.0, float(regressors["umidade_pct"].predict(row)[0])))
        working.loc[current_hour, "precipitacao_mm"] = RAIN_THRESHOLD_MM if rain else 0.0
        working.loc[current_hour, "temperatura_c"] = temperature
        working.loc[current_hour, "umidade_pct"] = humidity
        predictions.append({
            "data_hora_utc": current_hour.to_pydatetime().replace(tzinfo=timezone.utc),
            "temperatura_c": round(temperature, 2),
            "umidade_pct": round(humidity, 2),
            "probabilidade_chuva": round(probability, 3),
            "chuva_provavel": rain,
        })
    return {
        "modelo": "random_forest_horario",
        "variaveis": ["temperatura_c", "umidade_pct", "chuva"],
        "limiar_mm": RAIN_THRESHOLD_MM,
        "previsoes": predictions,
        "treinado_com_horas": len(training),
    }
