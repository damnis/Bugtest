import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import ADXIndicator

# Data ophalen
df = yf.download("ASML.AS", period="1mo", interval="1d")
df = df[(df["Volume"] > 0) & ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))]

high_series = df["High"].squeeze()
low_series = df["Low"].squeeze()
close_series = df["Close"].squeeze()

adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)

df["DI_PLUS"] = adx.adx_pos()
df["DI_MINUS"] = adx.adx_neg()
df["SAMD"] = 0.0

df.loc[(df["DI_PLUS"] > 0) & (df["DI_MINUS"] == 0), "SAMD"] = 1.0
df.loc[(df["DI_MINUS"] > 0) & (df["DI_PLUS"] == 0), "SAMD"] = -1.0
df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > 0), "SAMD"] = 0.5
df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > 0), "SAMD"] = -0.5

# Resultaten tonen
print(df[["Close", "DI_PLUS", "DI_MINUS", "SAMD"]].tail(10).round(2))
