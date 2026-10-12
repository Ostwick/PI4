from datetime import datetime, timezone

import requests

from .settings import INMET_STATION_CODE

BASE_URL = "https://apialert-as.inmet.gov.br/dados/mobile"


def fetch_official_forecast(station_code=None):
    station_code = station_code or INMET_STATION_CODE
    if not station_code:
        raise RuntimeError("Defina INMET_STATION_CODE antes de buscar a previsão oficial.")

    response = requests.get(
        f"{BASE_URL}/{station_code}",
        timeout=(10, 30),
        headers={
            "User-Agent": "ProjetoMeteorologiaUNIVESP/1.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://vitral.inmet.gov.br/",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("A previsão oficial do INMET não retornou um objeto JSON.")

    forecast = payload.get("10 dias", [])
    if not isinstance(forecast, list):
        raise ValueError("O campo '10 dias' da previsão oficial não é uma lista.")

    return {
        "fonte": "INMET",
        "estacao_codigo": station_code,
        "consultado_em": datetime.now(timezone.utc),
        "previsao_10dias": forecast,
    }