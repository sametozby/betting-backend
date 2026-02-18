import numpy as np
import pandas as pd

def norm_1x2(ch, cd, ca):
    pH, pD, pA = 1/ch, 1/cd, 1/ca
    s = pH + pD + pA
    return np.array([pH/s, pD/s, pA/s], float)

def odds_distance(df, ch, cd, ca):
    target = norm_1x2(ch, cd, ca)
    pH = 1/df["B365CH"]
    pD = 1/df["B365CD"]
    pA = 1/df["B365CA"]
    s  = pH+pD+pA
    dist = np.sqrt(((pH/s)-target[0])**2 +
                   ((pD/s)-target[1])**2 +
                   ((pA/s)-target[2])**2)
    return dist

def find_similar(df, ch, cd, ca, topk=100):
    df = df.copy()
    df["dist"] = odds_distance(df, ch, cd, ca)
    return df.sort_values("dist").head(topk)

def backtest(df):
    results = {}

    def calc(profits):
        bets = len(profits)
        wins = (profits > 0).sum()
        net = profits.sum()
        roi = net / bets if bets > 0 else 0
        return {"bets": int(bets), "winrate": float(wins/bets) if bets>0 else 0,
                "net_profit": float(net), "roi": float(roi)}

    # MS1
    profits = np.where(df["FTR"]=="H",
                       df["B365CH"]-1,
                       -1)
    results["MS1"] = calc(profits)

    # MS2
    profits = np.where(df["FTR"]=="A",
                       df["B365CA"]-1,
                       -1)
    results["MS2"] = calc(profits)

    # Over 2.5
    if "B365C>2.5" in df.columns:
        goals = df["FTHG"] + df["FTAG"]
        profits = np.where(goals>2.5,
                           df["B365C>2.5"]-1,
                           -1)
        results["Over2.5"] = calc(profits)

    return results
