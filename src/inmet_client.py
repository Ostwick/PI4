from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
from .settings import INMET_STATION_CODE, INMET_STATION_NAME, INMET_LOOKBACK_DAYS
from .transform import normalize_frame

BASE_URL = "https://apitempo.inmet.gov.br/estacao/dados"

def fetch_recent_observations(days=None):
    if not INMET_STATION_CODE:
        raise RuntimeError(
            "Defina INMET_STATION_CODE com o código exato da estação automática."
        )
    days = days or INMET_LOOKBACK_DAYS
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"{BASE_URL}/{start.isoformat()}/{end.isoformat()}/{INMET_STATION_CODE}"

    response = requests.get(
        url,
        timeout=(10, 90),
        headers={"User-Agent": "ProjetoMeteorologiaUNIVESP/1.0"}
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Resposta inesperada da API do INMET: {type(payload).__name__}")

    if not payload:
        print(f"API do INMET retornou zero registros para {start} a {end}.")
        return []

    frame = pd.DataFrame(payload)
    return normalize_frame(frame, INMET_STATION_CODE, INMET_STATION_NAME)
