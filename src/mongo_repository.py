from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from .settings import (
    MONGODB_URI, MONGODB_DATABASE,
    MONGODB_HOURLY_COLLECTION, MONGODB_DAILY_COLLECTION,
    MONGODB_PREDICTIONS_COLLECTION
)

def utc_now():
    return datetime.now(timezone.utc)

class WeatherRepository:
    def __init__(self):
        self.client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            appname="pipeline-meteorologico-inmet"
        )
        self.client.admin.command("ping")
        db = self.client[MONGODB_DATABASE]
        self.hourly = db[MONGODB_HOURLY_COLLECTION]
        self.daily = db[MONGODB_DAILY_COLLECTION]
        self.predictions = db[MONGODB_PREDICTIONS_COLLECTION]

        # A composite unique key makes repeated imports idempotent.
        self.hourly.create_index(
            [("estacao_codigo", ASCENDING), ("data_hora_utc", ASCENDING)],
            unique=True,
            name="uq_estacao_data_hora"
        )
        self.daily.create_index(
            [("estacao_codigo", ASCENDING), ("data_local", ASCENDING)],
            unique=True,
            name="uq_estacao_data_local"
        )
        self.predictions.create_index(
            [("estacao_codigo", ASCENDING), ("gerado_em", ASCENDING)],
            name="ix_previsoes_gerado_em"
        )

    def get_daily_history(self, station_code):
        return list(self.daily.find(
            {"estacao_codigo": station_code},
            {"_id": 0}
        ).sort("data_local", ASCENDING))

    def get_hourly_history(self, station_code):
        return list(self.hourly.find(
            {"estacao_codigo": station_code},
            {"_id": 0}
        ).sort("data_hora_utc", ASCENDING))

    def replace_prediction(self, station_code, document):
        self.predictions.replace_one(
            {"estacao_codigo": station_code},
            document,
            upsert=True
        )

    def upsert_many(self, collection, docs, key_fields, batch_size=500):
        total = 0
        now = utc_now()
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            ops = []
            for doc in batch:
                filt = {key: doc[key] for key in key_fields}
                payload = dict(doc)
                payload["atualizado_em"] = now
                ops.append(UpdateOne(filt, {"$set": payload}, upsert=True))
            if ops:
                result = collection.bulk_write(ops, ordered=False)
                total += result.upserted_count + result.modified_count
        return total

    def close(self):
        self.client.close()
