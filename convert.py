import pandas as pd
import numpy as np

IN_FILE = "merged_no_AH_with_filter_div_fixed.xlsx"
SHEET = "DATA"
OUT_FILE = "data.parquet"

print("Excel okunuyor:", IN_FILE)
df = pd.read_excel(IN_FILE, sheet_name=SHEET, engine="openpyxl")
print("Satır:", len(df), "Kolon:", len(df.columns))

# BOM kolon adı düzeltme
if "ï»¿Div" in df.columns and "Div" not in df.columns:
    df = df.rename(columns={"ï»¿Div": "Div"})

# Tarih parse
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# ---- SAYISAL DÖNÜŞÜM (pyarrow hatasını çözer) ----
# Oran/stats gibi kolonlarda sayı beklenir. Object olanları sayıya çevirelim.
# Önce bariz metin kolonlarını hariç tutalım.
text_cols = {"Div", "HomeTeam", "AwayTeam", "Referee", "source_file", "Time", "Season"}
for col in df.columns:
    if col in text_cols:
        continue

    # object/string ise: nokta/virgül gibi durumları normalize edip sayıya çevir
    if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
        s = df[col].astype(str).str.strip()

        # boşlar -> NaN
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})

        # bazı dosyalarda virgül ondalık olabiliyor:  "1,85" -> "1.85"
        s = s.str.replace(",", ".", regex=False)

        # sayıya çevir (olmayanlar NaN)
        df[col] = pd.to_numeric(s, errors="coerce")

# Gereksiz unnamed kolonlar varsa at
df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

print("Parquet yazılıyor...")
df.to_parquet(OUT_FILE, index=False)
print("OK ->", OUT_FILE)
