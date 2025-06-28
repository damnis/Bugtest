import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import ADXIndicator

# 📥 Data ophalen
st.write("Haal data op van Yahoo Finance...")
df = yf.download("ASML.AS", period="12mo", interval="1d")
df = df[(df["Volume"] > 0) & ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))]

if df.empty:
    st.error("Geen geldige data gevonden voor uw keuze.")
else:
    # ✅ Fix: Zorg dat de inputs eendimensionale Series zijn
    high_series = df["High"].squeeze()
    low_series = df["Low"].squeeze()
    close_series = df["Close"].squeeze()

    # ✅ SAMD op basis van DI+ en DI-
    adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)

    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()
    df["SAMD"] = 0.0

    df.loc[(df["DI_PLUS"] > 0) & (df["DI_MINUS"] == 0), "SAMD"] = 1.0
    df.loc[(df["DI_MINUS"] > 0) & (df["DI_PLUS"] == 0), "SAMD"] = -1.0
    df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > 0), "SAMD"] = 0.5
    df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > 0), "SAMD"] = -0.5

    st.subheader("📈 Laatste xx rijen met SAMD-berekening")
    st.write(df[["Close", "High", "Low", "DI_PLUS", "DI_MINUS", "SAMD"]].tail(120).round(2))



# wit
#high_series = df["High"].squeeze()
#low_series = df["Low"].squeeze()
#close_series = df["Close"].squeeze()
