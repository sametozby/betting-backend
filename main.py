from fastapi import FastAPI
import pandas as pd
from engine import find_similar, backtest

app = FastAPI()

df = pd.read_parquet("data.parquet")

@app.get("/")
def root():
    return {"status": "Betting API Running"}

@app.get("/meta")
def meta():
    return {
        "teams": sorted(list(set(df["HomeTeam"]) | set(df["AwayTeam"]))),
        "leagues": sorted(df["Div"].dropna().unique().tolist())
    }

@app.post("/analyze")
def analyze(payload: dict):

    ch = float(payload["ch"])
    cd = float(payload["cd"])
    ca = float(payload["ca"])
    topk = int(payload.get("topk", 100))

    similar = find_similar(df, ch, cd, ca, topk=topk)

    bt = backtest(similar)

    return {
        "matches": similar.head(50).to_dict(orient="records"),
        "backtest": bt
    }
