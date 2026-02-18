from fastapi import FastAPI
import pandas as pd
import numpy as np

from engine import find_similar, backtest

app = FastAPI()

# data.parquet repo root’ta olmalı
df = pd.read_parquet("data.parquet")

@app.get("/")
def root():
    return {"status": "Betting API Running"}

@app.get("/meta")
def meta():
    teams = sorted(list(set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))))
    leagues = sorted(df["Div"].dropna().astype(str).unique().tolist()) if "Div" in df.columns else []
    return {"teams": teams, "leagues": leagues}

@app.post("/analyze")
def analyze(payload: dict):
    ch = float(payload["ch"])
    cd = float(payload["cd"])
    ca = float(payload["ca"])
    topk = int(payload.get("topk", 100))
    limit = int(payload.get("limit", 50))

    similar = find_similar(df, ch, cd, ca, topk=topk)
    bt = backtest(similar)

    # ✅ JSON uyumluluk: NaN/Inf -> null
    out = similar.head(limit).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)

    return {
        "matches": out.to_dict(orient="records"),
        "backtest": bt
    }
