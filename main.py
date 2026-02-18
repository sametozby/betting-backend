from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np

from engine import find_similar, backtest

app = FastAPI()

df = pd.read_parquet("data.parquet")

@app.get("/")
def root():
    return {"status": "Betting API Running"}

@app.get("/meta")
def meta():
    teams = sorted(list(set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))))
    leagues = sorted(df["Div"].dropna().astype(str).unique().tolist()) if "Div" in df.columns else []
    return {"teams": teams, "leagues": leagues}

def clean_for_json(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    # Tarih varsa ISO string yap
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # NaN/Inf -> None
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)

    return out

def clean_dict_for_json(d):
    # dict içindeki NaN/Inf varsa temizle
    def fix(x):
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x
    return fix(d)

@app.post("/analyze")
def analyze(payload: dict):
    ch = float(payload["ch"])
    cd = float(payload["cd"])
    ca = float(payload["ca"])
    topk = int(payload.get("topk", 100))
    limit = int(payload.get("limit", 50))

    similar = find_similar(df, ch, cd, ca, topk=topk)

    bt = backtest(similar)
    bt = clean_dict_for_json(bt)

    out = clean_for_json(similar.head(limit))
    matches = out.to_dict(orient="records")

    # ✅ En sağlam JSON encoder (datetime/Decimal/np types hepsini çözer)
    response = {"matches": matches, "backtest": bt}
    return jsonable_encoder(response)
