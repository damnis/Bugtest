import streamlit as st
import yfinance as yf

st.title("Shift Test met AAPL")

# Haal recente data op
df = yf.download("AAPL", period="1mo", interval="1d")

# Voeg shifts toe
df["Close_shift1"] = df["Close"].shift(1)
df["Close_shift2"] = df["Close"].shift(2)

# Toon laatste 10 rijen
st.dataframe(df[["Close", "Close_shift1", "Close_shift2"]].tail(10))
