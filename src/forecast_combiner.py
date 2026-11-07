from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ML_WEIGHT = 0.4
OFFICIAL_WEIGHT = 0.6

FIELD_ALIASES = {
    "temperatura_minima_c": ("temperatura_minima_c", "temp_min", "temp_minima", "minima"),
    "temperatura_maxima_c": ("temperatura_maxima_c", "temp_max", "temp_maxima", "maxima"),
    "umidade_media_pct": ("umidade_media_pct", "umidade", "umidade_media", "humidity"),
    "precipitacao_total_mm": ("precipitacao_total_mm", "precipitacao", "chuva", "rain"),
}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", ".").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _date_value(record: dict[str, Any]) -> str | None:
    for key in ("data_local", "data", "date", "dt", "dia"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if len(text) >= 10:
            if "/" in text[:10]:
                parts = text[:10].split("/")
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            return text[:10]
    return None


def _records(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    records = payload.get("previsao_10dias", [])
    return [record for record in records if isinstance(record, dict)]


def _value(record: dict[str, Any], field: str) -> float | None:
    for key in FIELD_ALIASES[field]:
        if key in record:
            value = _as_float(record[key])
            if value is not None:
                return value
    return None


def combine_forecasts(
    ml_forecast: dict[str, Any],
    official_forecast: dict[str, Any] | None,
    ml_weight: float = ML_WEIGHT,
    official_weight: float = OFFICIAL_WEIGHT,
) -> dict[str, Any]:
    if ml_weight < 0 or official_weight < 0 or ml_weight + official_weight <= 0:
        raise ValueError("Os pesos das previsões devem ser positivos.")

    total_weight = ml_weight + official_weight
    ml_weight /= total_weight
    official_weight /= total_weight
    ml_by_date = {
        item.get("data_local"): item
        for item in ml_forecast.get("previsoes", [])
        if isinstance(item, dict) and item.get("data_local")
    }
    official_by_date = {
        _date_value(item): item
        for item in _records(official_forecast)
        if _date_value(item)
    }

    combined = []
    for day in sorted(set(ml_by_date) | set(official_by_date)):
        ml_item = ml_by_date.get(day, {})
        official_item = official_by_date.get(day, {})
        item: dict[str, Any] = {
            "data_local": day,
            "fontes": {
                "modelo_ml": bool(ml_item),
                "inmet": bool(official_item),
            },
        }
        for field in FIELD_ALIASES:
            ml_value = _as_float(ml_item.get(field))
            official_value = _value(official_item, field)
            if ml_value is not None and official_value is not None:
                item[field] = round(ml_value * ml_weight + official_value * official_weight, 2)
                item[f"{field}_ml"] = round(ml_value, 2)
                item[f"{field}_inmet"] = round(official_value, 2)
            elif ml_value is not None:
                item[field] = round(ml_value, 2)
                item[f"{field}_ml"] = round(ml_value, 2)
            elif official_value is not None:
                item[field] = round(official_value, 2)
                item[f"{field}_inmet"] = round(official_value, 2)

        precipitation = item.get("precipitacao_total_mm")
        if precipitation is not None:
            item["chuva_provavel"] = precipitation >= 0.1
        combined.append(item)

    return {
        "metodo": "media_ponderada",
        "pesos": {"modelo_ml": ml_weight, "inmet": official_weight},
        "gerado_em": datetime.now(timezone.utc),
        "status": "atualizada" if official_by_date else "somente_modelo_ml",
        "previsoes": combined,
    }