import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
from src.settings import HISTORICAL_CSV, INMET_STATION_CODE, INMET_STATION_NAME, HISTORY_YEARS
from src.transform import normalize_frame, build_daily_docs
from src.mongo_repository import WeatherRepository

def main():
    parser = argparse.ArgumentParser(description="Importa histórico CSV do INMET para o MongoDB.")
    parser.add_argument("--csv", default=HISTORICAL_CSV, help="Caminho do CSV histórico.")
    parser.add_argument("--years", type=int, default=HISTORY_YEARS, help="Quantidade de anos a importar.")
    parser.add_argument("--all", action="store_true", help="Importa todas as datas do CSV, sem filtro de anos.")
    args = parser.parse_args()

    if not INMET_STATION_CODE:
        raise RuntimeError("Defina INMET_STATION_CODE no .env.")
    frame = pd.read_csv(args.csv, sep=";", encoding_errors="replace")
    print(f"Arquivo lido: {args.csv} | linhas brutas: {len(frame):,}")
    docs = normalize_frame(frame, INMET_STATION_CODE, INMET_STATION_NAME)

    if not args.all:
        cutoff = datetime.now(timezone.utc) - timedelta(days=365.25 * args.years)
        docs = [d for d in docs if d["data_hora_utc"] >= cutoff]

    if not docs:
        raise RuntimeError(
            "Nenhum registro válido foi identificado. Confira os nomes das colunas e o formato do CSV."
        )

    repo = WeatherRepository()
    try:
        changed = repo.upsert_many(
            repo.hourly, docs, ["estacao_codigo", "data_hora_utc"]
        )
        daily = build_daily_docs(docs)
        daily_changed = repo.upsert_many(
            repo.daily, daily, ["estacao_codigo", "data_local"]
        )
        print(f"Observações horárias válidas: {len(docs):,}")
        print(f"Documentos horários inseridos/alterados: {changed:,}")
        print(f"Documentos diários calculados: {len(daily):,}")
        print(f"Documentos diários inseridos/alterados: {daily_changed:,}")
        print("Importação concluída.")
    finally:
        repo.close()

if __name__ == "__main__":
    main()
