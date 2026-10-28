import os
from dotenv import load_dotenv

load_dotenv()

def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {name}")
    return value

MONGODB_URI = required_env("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "meteorologia")
MONGODB_HOURLY_COLLECTION = os.getenv("MONGODB_HOURLY_COLLECTION", "observacoes_horarias")
MONGODB_DAILY_COLLECTION = os.getenv("MONGODB_DAILY_COLLECTION", "observacoes_diarias")
MONGODB_PREDICTIONS_COLLECTION = os.getenv("MONGODB_PREDICTIONS_COLLECTION", "previsoes")
INMET_STATION_CODE = os.getenv("INMET_STATION_CODE", "").strip()
INMET_STATION_NAME = os.getenv("INMET_STATION_NAME", "Presidente Prudente").strip()
INMET_LOOKBACK_DAYS = int(os.getenv("INMET_LOOKBACK_DAYS", "5"))
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "10"))
HISTORICAL_CSV = os.getenv("HISTORICAL_CSV", "data/historico_inmet.csv")
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "10"))
ML_MODEL = os.getenv("ML_MODEL", "random_forest").strip().lower()
ML_MIN_TRAINING_DAYS = int(os.getenv("ML_MIN_TRAINING_DAYS", "90"))
ML_RANDOM_STATE = int(os.getenv("ML_RANDOM_STATE", "42"))
ML_N_ESTIMATORS = int(os.getenv("ML_N_ESTIMATORS", "80"))
ML_MAX_DEPTH = int(os.getenv("ML_MAX_DEPTH", "10"))
ML_MIN_SAMPLES_LEAF = int(os.getenv("ML_MIN_SAMPLES_LEAF", "3"))
ML_VALIDATION_SAMPLES = int(os.getenv("ML_VALIDATION_SAMPLES", "12"))
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "models/random_forest_daily.joblib")
