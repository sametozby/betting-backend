from fastapi import FastAPI
import pandas as pd
import numpy as np
import json

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

@app.post("/analyze")
def analyze(payload: dict):
    ch = float(payload["ch"])
    cd = float(payload["cd"])
    ca = float(payload["ca"])
    topk = int(payload.get("topk", 100))
    limit = int(payload.get("limit", 50))

    similar = find_similar(df, ch, cd, ca, topk=topk)
    bt = backtest(similar)

    # ✅ Matches: NaN/Inf temizliği + JSON-safe dönüşüm (en sağlam yöntem)
    out = similar.head(limit).copy()

    # inf -> nan
    out = out.replace([np.inf, -np.inf], np.nan)

    # Date varsa ISO string
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # pandas to_json NaN -> null yapar, sonra json.loads ile listeye çeviriyoruz
    matches = json.loads(out.to_json(orient="records", date_format="iso"))

    # ✅ Backtest içinde NaN varsa None yap
    def fix_nan(x):
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: fix_nan(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix_nan(v) for v in x]
        return x

    bt = fix_nan(bt)

    return {"matches": matches, "backtest": bt}
