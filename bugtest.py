import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import ADXIndicator

# Instellingen
epsilonneg = 10.0
epsilonpos = 30.0
window = 14

# Tickers om te testen
tickers = ["ASML.AS", "SMCI", "MSFT"]

st.title("🔎 SAMD Debug View voor Meerdere Tickers")

# Functie om SAMD te berekenen per ticker
def bereken_samd(ticker):
    st.markdown(f"### 📊 {ticker}")

    # 📥 Data ophalen
    df = yf.download(ticker, period="120mo", interval="1d")
    df = df[(df["Volume"] > 0) & ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))]

    # Herstel ontbrekende waarden
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    if df.empty:
        st.error(f"❌ Geen geldige data gevonden voor {ticker}")
        return

    # SAMD op basis van DI+ en DI-
    high_series = df["High"]
    low_series = df["Low"]
    close_series = df["Close"]

    adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=window)

    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()
    df["SAMD"] = 0.0

    # 1️⃣ Sterke positieve richting
    df.loc[(df["DI_PLUS"] > epsilonpos) & (df["DI_MINUS"] <= epsilonneg), "SAMD"] = 1.0
    # 2️⃣ Sterke negatieve richting
    df.loc[(df["DI_MINUS"] > epsilonpos) & (df["DI_PLUS"] <= epsilonneg), "SAMD"] = -1.0
    # 3️⃣ Lichte positieve richting
    df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > epsilonneg), "SAMD"] = 0.5
    # 4️⃣ Lichte negatieve richting
    df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > epsilonneg), "SAMD"] = -0.5

    # 🔍 Toon tabel
    st.dataframe(df[["Close", "High", "Low", "DI_PLUS", "DI_MINUS", "SAMD"]].tail(40).round(2))

# ➕ Run voor elke ticker
for ticker in tickers:
    bereken_samd(ticker)




















# wit
#high_series = df["High"].squeeze()
#low_series = df["Low"].squeeze()
#close_series = df["Close"].squeeze()
