import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import ADXIndicator

# Data ophalen
df = yf.download("ASML.AS", period="1mo", interval="1d")
df = df[(df["Volume"] > 0) & ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))]

# ADX indicatoren berekenen
adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
df["DI_PLUS"] = adx.adx_pos()
df["DI_MINUS"] = adx.adx_neg()

# SAMD logica toepassen
df["SAMD"] = 0.0
df.loc[(df["DI_PLUS"] > 0) & (df["DI_MINUS"] == 0), "SAMD"] = 1.0
df.loc[(df["DI_MINUS"] > 0) & (df["DI_PLUS"] == 0), "SAMD"] = -1.0
df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > 0), "SAMD"] = 0.5
df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > 0), "SAMD"] = -0.5

# Resultaten tonen
print(df[["Close", "DI_PLUS", "DI_MINUS", "SAMD"]].tail(10).round(2))
