from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import re
import pandas as pd

LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

# INMET API keys and common CSV headers. Extend these aliases if your CSV differs.
ALIASES = {
    "date": [
        "data", "data medicao", "data_medicao", "dt_medicao", "date", "datahora",
        "data e hora", "data_hora"
    ],
    "hour": ["hora", "hora medicao", "hora_medicao", "hr_medicao", "hour"],
    "temperature": [
        "temperatura do ar bulbo seco horaria c", "temperatura", "temp", "tem_ins",
        "temperatura c", "temp_c", "temperatura_ar"
    ],
    "humidity": [
        "umidade relativa do ar horaria", "umidade relativa", "umidade", "umd_ins",
        "humidity", "umidade_pct"
    ],
    "precipitation": [
        "precipitacao total horaria mm", "precipitacao", "chuva", "precipitation",
        "chuva mm", "precip_mm"
    ],
    "station": [
        "codigo da estacao", "codigo_estacao", "estacao_codigo", "cd_estacao",
        "cod_estacao", "station_code"
    ],
}

def norm_col(value):
    import unicodedata
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return value

def find_column(df, kind):
    normalized = {norm_col(c): c for c in df.columns}
    aliases = [norm_col(a) for a in ALIASES[kind]]
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    # Permit partial matches for long INMET CSV labels.
    for norm, original in normalized.items():
        if any(alias and (alias in norm or norm in alias) for alias in aliases if len(alias) >= 5):
            return original
    return None

def parse_hour(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper().replace(" UTC", "").replace("UTC", "").strip()
    # INMET CSVs often use 0000, 0100, ... 2300, sometimes HH:MM.
    digits = re.sub(r"\D", "", s)
    if len(digits) == 4:
        hour, minute = int(digits[:2]), int(digits[2:])
    elif ":" in s:
        parts = s.split(":")
        hour, minute = int(parts[0]), int(parts[1][:2])
    elif len(digits) in (1, 2):
        hour, minute = int(digits), 0
    else:
        return None
    if hour > 23 or minute > 59:
        return None
    return hour, minute

def parse_datetime(date_value, hour_value=None):
    if pd.isna(date_value):
        return None
    raw_date = str(date_value).strip()
    # ISO dates are unambiguous; otherwise prefer Brazilian day-first notation.
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", raw_date):
        dt = pd.to_datetime(raw_date, dayfirst=False, errors="coerce")
    else:
        dt = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return None
    dt = dt.to_pydatetime().replace(tzinfo=None)

    if hour_value is not None:
        parsed = parse_hour(hour_value)
        if parsed is None:
            return None
        dt = dt.replace(hour=parsed[0], minute=parsed[1], second=0, microsecond=0)

    # INMET station timestamps are generally reported in UTC.
    return dt.replace(tzinfo=timezone.utc)

def numeric(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or n in (9999, -9999, 999, -999):
        return None
    return n

def normalize_frame(df, station_code, station_name="Presidente Prudente"):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = find_column(df, "date")
    hour_col = find_column(df, "hour")
    temp_col = find_column(df, "temperature")
    hum_col = find_column(df, "humidity")
    precip_col = find_column(df, "precipitation")
    station_col = find_column(df, "station")

    if date_col is None:
        raise ValueError(
            "Não encontrei a coluna de data. Colunas disponíveis: "
            + ", ".join(map(str, df.columns))
        )
    if hour_col is None and not any(
        token in norm_col(date_col) for token in ["datahora", "data e hora", "data hora"]
    ):
        raise ValueError(
            "Não encontrei a coluna de hora. Ajuste ALIASES em src/transform.py. "
            f"Colunas disponíveis: {', '.join(map(str, df.columns))}"
        )

    output = []
    for _, row in df.iterrows():
        dt = parse_datetime(row[date_col], row[hour_col] if hour_col else None)
        if dt is None:
            continue

        def val(col):
            return numeric(row[col]) if col is not None else None

        code = str(row[station_col]).strip() if station_col and pd.notna(row[station_col]) else station_code
        if not code or code.lower() == "nan":
            code = station_code

        doc = {
            "estacao_codigo": code,
            "estacao_nome": station_name,
            "data_hora_utc": dt,
            "temperatura_c": val(temp_col),
            "umidade_pct": val(hum_col),
            "precipitacao_mm": val(precip_col),
            "origem": "INMET",
        }
        # Keep a record only if at least one target variable is present.
        if any(doc[k] is not None for k in ("temperatura_c", "umidade_pct", "precipitacao_mm")):
            output.append(doc)

    # Deduplicate inside the same file before bulk writing.
    unique = {}
    for doc in output:
        key = (doc["estacao_codigo"], doc["data_hora_utc"])
        unique[key] = doc
    return list(unique.values())

def build_daily_docs(hourly_docs):
    groups = {}
    for doc in hourly_docs:
        dt_utc = doc["data_hora_utc"]
        # PyMongo commonly returns BSON datetimes as naive UTC unless tz_aware=True.
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(LOCAL_TZ)
        day = dt_local.date().isoformat()
        key = (doc["estacao_codigo"], day)
        groups.setdefault(key, []).append(doc)

    daily_docs = []
    for (station, day), docs in groups.items():
        temps = [d["temperatura_c"] for d in docs if d.get("temperatura_c") is not None]
        hums = [d["umidade_pct"] for d in docs if d.get("umidade_pct") is not None]
        precips = [d["precipitacao_mm"] for d in docs if d.get("precipitacao_mm") is not None]
        # Date key is a local calendar date; keep it as ISO YYYY-MM-DD.
        daily_docs.append({
            "estacao_codigo": station,
            "data_local": day,
            "temperatura_media_c": round(sum(temps) / len(temps), 2) if temps else None,
            "temperatura_minima_c": round(min(temps), 2) if temps else None,
            "temperatura_maxima_c": round(max(temps), 2) if temps else None,
            "umidade_media_pct": round(sum(hums) / len(hums), 2) if hums else None,
            "umidade_minima_pct": round(min(hums), 2) if hums else None,
            "umidade_maxima_pct": round(max(hums), 2) if hums else None,
            "precipitacao_total_mm": round(sum(precips), 2) if precips else None,
            "quantidade_registros_horarios": len(docs),
            "origem": "INMET_agregado",
        })
    return daily_docs
