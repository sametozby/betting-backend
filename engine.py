import numpy as np
import pandas as pd

def norm_1x2(ch, cd, ca):
    pH, pD, pA = 1 / ch, 1 / cd, 1 / ca
    s = pH + pD + pA
    return np.array([pH / s, pD / s, pA / s], float)

def odds_distance(df, ch, cd, ca):
    target = norm_1x2(ch, cd, ca)

    pH = 1 / df["B365CH"]
    pD = 1 / df["B365CD"]
    pA = 1 / df["B365CA"]
    s = pH + pD + pA

    dist = np.sqrt(
        ((pH / s) - target[0]) ** 2 +
        ((pD / s) - target[1]) ** 2 +
        ((pA / s) - target[2]) ** 2
    )
    return dist

def find_similar(df, ch, cd, ca, topk=100):
    tmp = df.copy()
    tmp["dist"] = odds_distance(tmp, ch, cd, ca)
    tmp = tmp.sort_values("dist").head(int(topk))
    return tmp

def backtest(sim_df):
    """
    1 birim stake varsayımıyla net kâr ve ROI hesaplar.
    """
    df = sim_df.copy()

    results = {}

    def calc(profits: pd.Series):
        profits = pd.to_numeric(profits, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        bets = int(len(profits))
        if bets == 0:
            return {"bets": 0, "winrate": None, "net_profit": 0.0, "roi": None}
        wins = int((profits > 0).sum())
        net = float(profits.sum())
        roi = float(net / bets)
        return {"bets": bets, "winrate": float(wins / bets), "net_profit": net, "roi": roi}

    # MS1
    profits_h = np.where(df["FTR"] == "H", df["B365CH"] - 1, -1)
    results["MS1"] = calc(pd.Series(profits_h))

    # MS0
    profits_d = np.where(df["FTR"] == "D", df["B365CD"] - 1, -1)
    results["MS0"] = calc(pd.Series(profits_d))

    # MS2
    profits_a = np.where(df["FTR"] == "A", df["B365CA"] - 1, -1)
    results["MS2"] = calc(pd.Series(profits_a))

    # Over 2.5
    if "B365C>2.5" in df.columns:
        goals = (pd.to_numeric(df["FTHG"], errors="coerce") + pd.to_numeric(df["FTAG"], errors="coerce"))
        profits_o = np.where(goals > 2.5, df["B365C>2.5"] - 1, -1)
        results["Over2.5"] = calc(pd.Series(profits_o))

    # Under 2.5
    if "B365C<2.5" in df.columns:
        goals = (pd.to_numeric(df["FTHG"], errors="coerce") + pd.to_numeric(df["FTAG"], errors="coerce"))
        profits_u = np.where(goals < 2.5, df["B365C<2.5"] - 1, -1)
        results["Under2.5"] = calc(pd.Series(profits_u))

    return results
