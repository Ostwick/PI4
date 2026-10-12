
import os
import secrets
from flask import Flask, jsonify, request
from main import run_pipeline

app = Flask(__name__)

API_TOKEN = os.environ.get("API_TOKEN")

@app.get("/")
def health():
    return jsonify({"status": "online"})

@app.post("/atualizar")
def atualizar():
    token = request.headers.get("X-API-Token", "")

    if not API_TOKEN or not secrets.compare_digest(token, API_TOKEN):
        return jsonify({"erro": "Não autorizado"}), 401

    try:
        run_pipeline()
        return jsonify({"status": "sucesso"})
    except Exception:
        app.logger.exception("Erro no pipeline")
        return jsonify({"erro": "Falha no pipeline"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
