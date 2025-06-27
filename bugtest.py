import yfinance as yf
import streamlit as st

st.title("Test Streamlit App")
st.write("Hallo wereld!")
# Haal recente data op
df = yf.download("AAPL", period="1mo", interval="1d")

# Bekijk Close en shifts
df["Close_shift1"] = df["Close"].shift(1)
df["Close_shift2"] = df["Close"].shift(2)

print(df[["Close", "Close_shift1", "Close_shift2"]].tail(10))
print(df[["Close", "Close_shift1", "Close_shift1"]].tail(10))
