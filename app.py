
import os
import secrets
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from main import run_pipeline

app = Flask(__name__)

API_TOKEN = os.environ.get("API_TOKEN", "").strip()


def _requested_token() -> str:
    header_token = request.headers.get("X-API-Token", "")
    if header_token:
        return header_token

    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]

    return request.args.get("token", "")


@app.get("/")
@app.post("/")
def health():
    return jsonify({"status": "online"})


@app.get("/atualizar")
@app.post("/atualizar")
def atualizar():
    token = _requested_token()

    if API_TOKEN and (not token or not secrets.compare_digest(token, API_TOKEN)):
        return jsonify({"status": "unauthorized"}), 401

    try:
        run_pipeline()
        return jsonify({
            "status": "ok",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        app.logger.exception("Erro no pipeline")
        return jsonify({"status": "error", "message": "pipeline_failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
