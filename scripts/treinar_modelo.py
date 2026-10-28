from pathlib import Path

from src.mongo_repository import WeatherRepository
from src.ml_forecaster import train_and_save_model
from src.settings import FORECAST_HORIZON_DAYS, INMET_STATION_CODE, MODEL_ARTIFACT_PATH


def main():
    repo = WeatherRepository()
    try:
        history = repo.get_daily_history(INMET_STATION_CODE)
        if not history:
            raise RuntimeError("Não há histórico diário para treinar o modelo.")

        result = train_and_save_model(history, FORECAST_HORIZON_DAYS, output_path=MODEL_ARTIFACT_PATH)
        print(f"Modelo salvo em: {result['arquivo']}")
        print(f"MAE: {result['validacao']['mae_por_variavel']}")
    finally:
        repo.close()


if __name__ == "__main__":
    main()
