from datetime import datetime, timedelta, timezone
from src.settings import INMET_LOOKBACK_DAYS
from src.inmet_client import fetch_recent_observations
from src.transform import build_daily_docs
from src.mongo_repository import WeatherRepository
from src.forecast_client import fetch_official_forecast
from src.ml_forecaster import predict
from src.hourly_forecaster import predict_rain_next_hours
from src.settings import (
    FORECAST_HORIZON_DAYS,
    INMET_STATION_CODE,
)


def run_pipeline():
    started = datetime.now(timezone.utc)
    print(f"[{started.isoformat()}] Iniciando atualização meteorológica.")

    try:
        hourly_docs = fetch_recent_observations(days=INMET_LOOKBACK_DAYS)
        observation_status = "atualizadas" if hourly_docs else "sem_dados"
    except Exception as error:
        print(f"Observações recentes indisponíveis; usando o histórico salvo: {error}")
        hourly_docs = []
        observation_status = "indisponiveis"

    repo = WeatherRepository()
    try:
        changed_hourly = 0
        changed_daily = 0
        daily_docs = []
        if hourly_docs:
            changed_hourly = repo.upsert_many(
                repo.hourly, hourly_docs, ["estacao_codigo", "data_hora_utc"]
            )

            # Rebuild daily summaries for the dates in this run, using the full
            # hourly history already stored in MongoDB, so overlap can correct a day.
            from src.transform import LOCAL_TZ
            from datetime import date
            dates_by_station = {}
            for doc in hourly_docs:
                local_day = doc["data_hora_utc"].astimezone(LOCAL_TZ).date().isoformat()
                dates_by_station.setdefault(doc["estacao_codigo"], set()).add(local_day)

            for station, days in dates_by_station.items():
                for day in sorted(days):
                    # Convert local calendar-day boundaries to UTC.
                    local_date = date.fromisoformat(day)
                    start_local = datetime(local_date.year, local_date.month, local_date.day, tzinfo=LOCAL_TZ)
                    end_local = start_local + timedelta(days=1)
                    cursor = repo.hourly.find({
                        "estacao_codigo": station,
                        "data_hora_utc": {
                            "$gte": start_local.astimezone(timezone.utc),
                            "$lt": end_local.astimezone(timezone.utc)
                        }
                    })
                    daily_docs.extend(build_daily_docs(list(cursor)))

            changed_daily = repo.upsert_many(
                repo.daily, daily_docs, ["estacao_codigo", "data_local"]
            )

        history = repo.get_daily_history(INMET_STATION_CODE)
        if not history:
            raise RuntimeError("Não há histórico diário suficiente para publicar previsões.")
        ml_forecast = predict(history, FORECAST_HORIZON_DAYS)
        hourly_forecast = predict_rain_next_hours(
            repo.get_hourly_history(INMET_STATION_CODE), hours=24
        )
        prediction_document = {
            "estacao_codigo": INMET_STATION_CODE,
            "estacao_nome": history[-1].get("estacao_nome") if history else None,
            "gerado_em": datetime.now(timezone.utc),
            "status_observacoes": observation_status,
            "observacoes_recebidas": len(hourly_docs),
            "modelo_ml": ml_forecast,
            "chuva_proximas_horas": hourly_forecast,
            "previsao_oficial": None,
            "status_previsao_oficial": "indisponivel",
        }
        try:
            prediction_document["previsao_oficial"] = fetch_official_forecast()
            prediction_document["status_previsao_oficial"] = "atualizada"
        except Exception as error:
            print(f"Previsão oficial indisponível; mantendo o ML: {error}")

        repo.replace_prediction(INMET_STATION_CODE, prediction_document)

        print(f"Observações horárias recebidas: {len(hourly_docs):,}")
        print(f"Documentos horários inseridos/alterados: {changed_hourly:,}")
        print(f"Resumo diário calculado: {len(daily_docs):,}")
        print(f"Documentos diários inseridos/alterados: {changed_daily:,}")
        print(f"Previsões ML publicadas: {len(ml_forecast['previsoes']):,}")
        print(f"Previsão oficial: {prediction_document['status_previsao_oficial']}")
        print("Atualização concluída com sucesso.")
    finally:
        repo.close()


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
