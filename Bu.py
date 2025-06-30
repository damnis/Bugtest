import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator
#from datetime import date
#from datetime import timedelta
#from ta.momentum import TRIXIndicator

# --- Functie om data op te halen ---
# 📥 Cachen van data per combinatie van ticker/interval (15 minuten geldig)
@st.cache_data(ttl=900)
def fetch_data_cached(ticker, interval, period):
    return yf.download(ticker, interval=interval, period=period)
def weighted_moving_average(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)



# ✅ Wrapper-functie met schoonmaak + fallback
def fetch_data(ticker, interval):
    # 📅 Interval naar periode (maximale periode per interval volgens Yahoo Finance)
    if interval == "15m":
        period = "30d"     # Max voor 15m = 60d, maar 30d is veiliger/snelle laadtijd
    elif interval == "1h":
        period = "720d"    # Max voor 1h = ±730d (2 jaar)
    elif interval == "4h":
        period = "360d"    # Max voor 4h = ±730d (gedeeld over 6 candles per dag)
    elif interval == "1d":
        period = "20y"     # Max voor 1d = 20y
    elif interval == "1wk":
        period = "20y"     # Max voor 1wk = 20y
    elif interval == "1mo":
        period = "25y"  # maximaal bij maanddata = 25y
    else:
        period = "25y"     # Fallback (bijv. voor '1mo' of onbekend)

        # 📥 Ophalen via gecachete functie
    df = fetch_data_cached(ticker, interval, period)

    # 🛡️ Check op geldige data
    if df.empty or "Close" not in df.columns or "Open" not in df.columns:
        return pd.DataFrame()

    # 🧹 Verwijder irrelevante of foutieve rijen
    df = df[
        (df["Volume"] > 0) &
        ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))
    ]

    # 🕓 Zorg dat index datetime is
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # 🔁 Vul NaN's op per kolom
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    return df

# 📆 Periode voor SAM-grafiek op basis van interval
def bepaal_grafiekperiode(interval):
    if interval == "15m":
        return timedelta(days=7)        # 7 dagen à ~96 candles per dag = ±672 punten
    elif interval == "1h":
        return timedelta(days=5)        # 5 dagen à ~7 candles = ±35 punten
    elif interval == "4h":
        return timedelta(days=90)       # 3 maanden à ~6 candles per week
    elif interval == "1d":
        return timedelta(days=180)      # 180=6 maanden à 1 candle per dag
    elif interval == "1wk":
        return timedelta(weeks=104)     # 104=2 jaar aan weekly candles (104 candles)
    elif interval == "1mo":
        return timedelta(weeks=520)     # 520=0 jaar aan monthly candles (120 candles)
    else:
        return timedelta(weeks=260)     # Fallback = 5 jaar
# periode voor koersgrafiek 
#def bepaal_grafiekperiode2(interval):
#    if interval == "15m":
#        return timedelta(days=7)
 #   elif interval == "1h":
 #       return timedelta(days=5)
 #   elif interval == "4h":
 #       return timedelta(days=90)
#    elif interval == "1d":
#        return timedelta(days=180)
 #   else:
  #      return timedelta(weeks=260)  # bijv. bij weekly/monthly data


# --- SAM Indicatorberekeningen ---
def calculate_sam(df):
    df = df.copy()

    # Basiskolommen
    # --- SAMG op basis van Weighted Moving Averages + Crossovers nodig ---
  #  def weighted_moving_average(series, window):  # reeds gedefinieerd 
  #      weights = np.arange(1, window + 1)
  #      return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    
    # --- SAMK: candlestick score op basis van patronen Open/Close ---
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"] > df["Close"].shift(1)
    df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
    df["c5"] = df["Close"] < df["Open"]
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
    df["c7"] = df["Close"] < df["Close"].shift(1)
    df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

    df["SAMK"] = 0.0  # standaard
    # Positieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"]).fillna(False),
        "SAMK"
    ] = 0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] | df["c3"]).fillna(False),
        "SAMK"
    ] = 0.25

    # Negatieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"]).fillna(False),
        "SAMK"
    ] = -0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] | df["c7"]).fillna(False),
        "SAMK"
    ] = -0.25
    

    # SAMK oud
#    df["SAMK"] = 0
#    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
 #   df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    # --- SAMG (WMA-based trendanalyse met crossovers) ---
    df["WMA18"] = weighted_moving_average(df["Close"], 18)
    df["WMA35"] = weighted_moving_average(df["Close"], 35)
    df["WMA18_shifted"] = df["WMA18"].shift(1)
    df["WMA35_shifted"] = df["WMA35"].shift(1)

    df["SAMG"] = 0.0

    # Kleine trendbewegingen omhoog en omlaag
    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    # Grote crossovers tussen WMA18 en WMA35
    df.loc[
        (df["WMA18_shifted"] < df["WMA35_shifted"]) & (df["WMA18"] > df["WMA35"]),
        "SAMG"
    ] = 0.75 # oorspronkelijk 1.0, uit in nieuwere sam versie
    df.loc[
        (df["WMA18_shifted"] > df["WMA35_shifted"]) & (df["WMA18"] < df["WMA35"]),
        "SAMG"
    ] = -0.75 # zie vorige

#  samg oud
#    df["Change"] = df["Close"].pct_change()
#    df["SAMG"] = 0
#    df.loc[df["Change"] > 0.03, "SAMG"] = 1
#    df.loc[df["Change"] < -0.03, "SAMG"] = -1

    # SAMT oud
#    df["SMA5"] = df["Close"].rolling(window=5).mean()
#    df["SMA20"] = df["Close"].rolling(window=20).mean()
#    df["SAMT"] = 0
 #   df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1
#    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1
# --- SAMT op basis van Weighted Moving Averages 6 en 80 ---
#    def weighted_moving_average(series, window): --> al eerder gedefineerd
 #       weights = np.arange(1, window + 1)
  #      return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

    df["WMA6"] = weighted_moving_average(df["Close"], 6)
    df["WMA6_shifted"] = df["WMA6"].shift(1)
    df["WMA80"] = weighted_moving_average(df["Close"], 80)

    df["SAMT"] = 0.0  # standaardwaarde

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = 0.5

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = 0.25

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = -0.75

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = -0.5
    
    # SAMD
    # --- SAMD op basis van DI+ en DI- ---
    high_series = df["High"].squeeze()
    low_series = df["Low"].squeeze()
    close_series = df["Close"].squeeze()

    adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)

    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()
  #  df["SAMD"] = 0.0  # begin met 0.0

    # Epsilon-drempels instellen
    epsilonneg = 10.0  # vrijwel afwezig andere richting
    epsilonpos = 30.0  # sterke richting
    df["SAMD"] = 0.0
    # 1️⃣ Sterke positieve richting
    df.loc[(df["DI_PLUS"] > epsilonpos) & (df["DI_MINUS"] <= epsilonneg), "SAMD"] = 0.75 # was 1.0
    # 2️⃣ Sterke negatieve richting
    df.loc[(df["DI_MINUS"] > epsilonpos) & (df["DI_PLUS"] <= epsilonneg), "SAMD"] = -0.75 # was -1.0
    # 3️⃣ Lichte positieve richting
    df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > epsilonneg), "SAMD"] = 0.5
    # 4️⃣ Lichte negatieve richting
    df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > epsilonneg), "SAMD"] = -0.5


    # samd oud
#    df["daily_range"] = df["High"] - df["Low"]
 #   avg_range = df["daily_range"].rolling(window=14).mean()
#    df["SAMD"] = 0
#    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1
#    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1

    # SAMM
    # ✅ Correcte MACD-berekening met ta
    close_series = df["Close"].squeeze()  # Zorg dat het een Series is
    macd_ind = ta.trend.MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)

    df["MACD"] = macd_ind.macd()
    df["SIGNAL"] = macd_ind.macd_signal()

    # ✅ Detecteer MACD crossovers voor SAMM
    prev_macd = df["MACD"].shift(1)
    prev_signal = df["SIGNAL"].shift(1)

    conditions = [
        (prev_macd < prev_signal) & (df["MACD"] > df["SIGNAL"]),
        (df["MACD"] > df["SIGNAL"]),
        (prev_macd > prev_signal) & (df["MACD"] < df["SIGNAL"]),
        (df["MACD"] <= df["SIGNAL"])
    ]
    keuzes = [1.0, 0.5, -1.0, -0.5]

    df["SAMM"] = np.select(conditions, keuzes, default=0.0)

    # samm oud   
  #  df["SMA10"] = df["Close"].rolling(window=10).mean()
  #  df["SMA50"] = df["Close"].rolling(window=50).mean()
 #   df["SAMM"] = 0
  #  df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1
 #   df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1

    # SAMX
#    from ta.momentum import TRIXIndicator

    # --- SAMX op basis van TRIX ---
    # --- SAMX: handmatige TRIX-berekening en interpretatie ---
    def calculate_trix(series, period=15):
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
        return trix

    df["TRIX"] = calculate_trix(df["Close"], period=15)
    df["TRIX_PREV"] = df["TRIX"].shift(1)
    df["SAMX"] = 0.0

    # Sterke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] > df["TRIX_PREV"]), "SAMX"] = 0.75
    # Zwakke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] <= df["TRIX_PREV"]), "SAMX"] = 0.5
    # Sterke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] < df["TRIX_PREV"]), "SAMX"] = -0.75
    # Zwakke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] >= df["TRIX_PREV"]), "SAMX"] = -0.5

    # --- SAMX op basis van TRIX - werkt niet met ta
#    close_series = df["Close"].squeeze()
 #   trix_ind = TRIXIndicator(close=close_series, window=15)

  #  df["TRIX"] = trix_ind.trix()
#    df["TRIX_PREV"] = df["TRIX"].shift(1)
#    df["SAMX"] = 0.0  # Standaardwaarde

    # Sterke opwaartse trend
#    df.loc[(df["TRIX"] > 0) & (df["TRIX"] > df["TRIX_PREV"]), "SAMX"] = 0.75
    # Zwakke opwaartse trend
#    df.loc[(df["TRIX"] > 0) & (df["TRIX"] <= df["TRIX_PREV"]), "SAMX"] = 0.5
    # Sterke neerwaartse trend
#    df.loc[(df["TRIX"] < 0) & (df["TRIX"] < df["TRIX_PREV"]), "SAMX"] = -0.75
    # Zwakke neerwaartse trend
#    df.loc[(df["TRIX"] < 0) & (df["TRIX"] >= df["TRIX_PREV"]), "SAMX"] = -0.5

    # SAMX OUD
#    df["Momentum"] = df["Close"] - df["Close"].shift(3)
#    df["SAMX"] = 0
#    df.loc[df["Momentum"] > 0, "SAMX"] = 1
 #   df.loc[df["Momentum"] < 0, "SAMX"] = -1

    # Totale SAM
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df
    

# --- Advies en rendementen ---
def determine_advice(df, threshold):
    df = df.copy()

    # 🧮 Trendberekening over SAM
    df["Trend"] = weighted_moving_average(df["SAM"], 12)  # wma zoals hoort
 #   df["Trend"] = df["SAM"].rolling(window=12).mean()  # hier Trend sam ingeven default:12
    df["TrendChange"] = df["Trend"] - df["Trend"].shift(1)
    df["Richting"] = np.sign(df["TrendChange"])
    df["Trail"] = 0
    df["Advies"] = np.nan

    # 🔁 Bereken Trail (opeenvolgende richting-versterking)
    huidige_trend = 0
    for i in range(1, len(df)):
        huidige = df["Richting"].iloc[i]
        vorige = df["Richting"].iloc[i - 1]

        if huidige == vorige and huidige != 0:
            huidige_trend += 1
        elif huidige != 0:
            huidige_trend = 1
        else:
            huidige_trend = 0

        df.at[df.index[i], "Trail"] = huidige_trend

    # ✅ Eerst advies op basis van sterke trendwaarde
#    df.loc[df["Trend"] > 2, "Advies"] = "Kopen"
 #   df.loc[df["Trend"] < -2, "Advies"] = "Verkopen"

 #   df.loc[df["SAM"] > 2, "Advies"] = "Kopen"
 #   df.loc[df["SAM"] < -2, "Advies"] = "Verkopen"

    # ✅ Daarna alleen nog trailing advies als er nog geen advies is
    mask_koop = (df["Richting"] == 1) & (df["Trail"] >= threshold) & (df["Advies"].isna())
    mask_verkoop = (df["Richting"] == -1) & (df["Trail"] >= threshold) & (df["Advies"].isna())

    df.loc[mask_koop, "Advies"] = "Kopen"
    df.loc[mask_verkoop, "Advies"] = "Verkopen"

    # 🔄 Advies forward fillen
    df["Advies"] = df["Advies"].ffill()

    # 📊 Bereken rendementen op basis van adviesgroepering
    df["AdviesGroep"] = (df["Advies"] != df["Advies"].shift()).cumsum()
    rendementen = []
    sam_rendementen = []

    groepen = list(df.groupby("AdviesGroep"))

    for i in range(len(groepen)):
        _, groep = groepen[i]
        advies = groep["Advies"].iloc[0]

        start = groep["Close"].iloc[0]
        if i < len(groepen) - 1:
            eind = groepen[i + 1][1]["Close"].iloc[0]
        else:
            eind = groep["Close"].iloc[-1]

        try:
            start = float(start)
            eind = float(eind)
            if start != 0.0:
                markt_rendement = (eind - start) / start
                sam_rendement = markt_rendement if advies == "Kopen" else -markt_rendement
            else:
                markt_rendement = 0.0
                sam_rendement = 0.0
        except Exception:
            markt_rendement = 0.0
            sam_rendement = 0.0

        rendementen.extend([markt_rendement] * len(groep))
        sam_rendementen.extend([sam_rendement] * len(groep))

    # ✅ Controle
    if len(rendementen) != len(df):
        raise ValueError(f"Lengte mismatch: rendementen={len(rendementen)}, df={len(df)}")

    df["Markt-%"] = rendementen
    df["SAM-%"] = sam_rendementen

    # 🏁 Huidig advies bepalen
    if "Advies" in df.columns and df["Advies"].notna().any():
        huidig_advies = df["Advies"].dropna().iloc[-1]
    else:
        huidig_advies = "Niet beschikbaar"

    return df, huidig_advies

   
# --- Streamlit UI ---
#st.title("SAM Trading Indicator")
# Titel met kleur en grootte tonen
# Kleur bepalen op basis van advies
#advies_kleur = "green" if huidig_advies == "Kopen" else "red" if huidig_advies == "Verkopen" else "gray"

# SAM TITLE
st.markdown(
    f"""
    <h1>SAM Trading Indicator<span style='color:#3366cc'>   </span></h1>
    """,
    unsafe_allow_html=True
)
# Simple Alert Monitor met responsive uitleg
st.markdown("""
<style>
.sam-uitleg details[open] {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 90vw;
  max-width: 700px;
  z-index: 999;
  background-color: #f9f9f9;
  padding: 1em;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  border-radius: 10px;
}
</style>

<div class="sam-uitleg" style='display: flex; justify-content: space-between; align-items: top;'>
  <div style='flex: 1;'>
    <h5 style='margin: 0;'>Simple Alert Monitor</h5>
  </div>
  <div style='flex: 1; text-align: right;'>
    <details>
      <summary style='cursor: pointer; font-weight: bold; color: #555;text-align: right;'>ℹ️ Uitleg SAM Trading Indicator</summary>
      <div style='margin-top: 10px;'>
        <p style='font-size: 13px; color: #333; text-align: left'>
        Gebruik de <strong>SAM Trading Indicator</strong> door voornamelijk te sturen op de blauwe lijn in de SAM en Trend grafiek,
        de trendlijn. De groene en rode SAM waarden (vaak perioden) geven het momentum weer...<br><br>
        Het advies is hiervan afgeleid en kan bijgesteld worden door de gevoeligheid aan te passen.<br>
        De indicator is oorspronkelijk bedoeld voor de <strong>middellange termijn belegger</strong>.
        </p>
      </div>
    </details>
  </div>
</div>
""", unsafe_allow_html=True)

# Simple Alert Monitor 
#st.markdown("""
#<div style='display: flex; justify-content: space-between; align-items: top;'>
#  <div style='flex: 1;'>
#    <h5 style='margin: 0;'>Simple Alert Monitor</h5>
#  </div>
#  <div style='flex: 1; text-align: right;'>
  #  <details>
  #    <summary style='cursor: pointer; font-weight: bold; color: #555;text-align: right;'>ℹ️ Uitleg SAM Trading Indicator</summary>
   #   <div style='margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 8px;'>
   #     <p style='font-size: 13px; color: #333; text-align: left'>
     #   Gebruik de <strong>SAM Trading Indicator</strong> door voornamelijk te sturen op de blauwe lijn in de SAM en Trend grafiek,
   #     de trendlijn. De groene en rode SAM waarden (vaak perioden) geven het momentum weer...
  #      <br><br>
   #     Het advies is hiervan afgeleid en kan bijgesteld worden door de gevoeligheid aan te passen.<br>
   #     De indicator is oorspronkelijk bedoeld voor de <strong>middellange termijn belegger</strong>.
 #       </p>
#      </div>
  #  </details>
#  </div>
#</div>
#""", unsafe_allow_html=True)

#st.markdown(
#    f"""
#    <h1>SAM Trading Indicator<span style='color:#3366cc'>   </span></h1>
#    """,
#    unsafe_allow_html=True
##)
# Simple Alert Monitor 
#col1, col2 = st.columns([9, 6])  # Pas verhouding aan als je wilt

#with col1:
#    st.markdown(
  #      f"""
  #      <h5>Simple Alert Monitor</h5>
   #     """,
 #       unsafe_allow_html=True 
#    )    
#with col2:
#    with st.expander("ℹ️ Uitleg SAM Trading Indicator"):
  #      st.markdown(
  #          """
   #         <div style='color:#444; font-size:12px;'>
    #        Gebruik de SAM Trading Indicator door voornamelijk te sturen op de blauwe lijn in de SAM en Trend grafiek,
    #        de trendlijn. De groene en rode SAM waarden (vaak perioden) geven het momentum weer, dus pas op voor aankoopbeslissingen
    #        in een rode periode en wees niet te snel met verkopen als het duidelijk groen is. Kleine trend wijzigingen
    #        zouden anders uw resultaat negatief kunnen beïnvloeden.<br>
   #         Het advies is hiervan afgeleid en kan bijgesteld worden door de gevoeligheid aan te passen. Maar dit blijkt een model en slechts een benadering
  #          van de juiste beslissing. Blijf de blauwe lijn (die vrij geleidelijk omhoog en omlaag gaat) zelf goed volgen.<br>
    #        De indicator is oorspronkelijk bedoeld voor de middellange termijn belegger en beslissingen op 'week' basis,
   #         maar kan ook voor korte intervallen gebruikt worden.
     #       </div>
    #        """,
    #        unsafe_allow_html=True
   #     )


# --- Volledige tickerlijsten ---
aex_tickers = {
"ABN.AS": "ABN AMRO", "ADYEN.AS": "Adyen", "AGN.AS": "Aegon", "AD.AS": "Ahold Delhaize", 
"AKZA.AS": "Akzo Nobel", "MT.AS": "ArcelorMittal", "ASM.AS": "ASMI", "ASML.AS": "ASML", "ASRNL.AS": "ASR Nederland",
"BESI.AS": "BESI", "DSFIR.AS": "DSM-Firmenich", "GLPG.AS": "Galapagos", "HEIA.AS": "Heineken", 
"IMCD.AS": "IMCD", "INGA.AS": "ING Groep", "TKWY.AS": "Just Eat Takeaway", "KPN.AS": "KPN",
"NN.AS": "NN Group", "PHIA.AS": "Philips", "PRX.AS": "Prosus", "RAND.AS": "Randstad",
"REN.AS": "Relx", "SHELL.AS": "Shell", "UNA.AS": "Unilever", "WKL.AS": "Wolters Kluwer"
}

dow_tickers = {
    'MMM': '3M', 'AXP': 'American Express', 'AMGN': 'Amgen', 'AAPL': 'Apple', 'BA': 'Boeing',
    'CAT': 'Caterpillar', 'CVX': 'Chevron', 'CSCO': 'Cisco', 'KO': 'Coca-Cola', 'DIS': 'Disney',
    'GS': 'Goldman Sachs', 'HD': 'Home Depot', 'HON': 'Honeywell', 'IBM': 'IBM', 'INTC': 'Intel',
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'MCD': 'McDonaldâ€™s', 'MRK': 'Merck',
    'MSFT': 'Microsoft', 'NKE': 'Nike', 'PG': 'Procter & Gamble', 'CRM': 'Salesforce',
    'TRV': 'Travelers', 'UNH': 'UnitedHealth', 'VZ': 'Verizon', 'V': 'Visa', 'WMT': 'Walmart',
    'DOW': 'Dow', 'RTX': 'RTX Corp.', 'WBA': 'Walgreens Boots'
}
nasdaq_tickers = {
    'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'AMZN': 'Amazon', 'META': 'Meta',
    'NFLX': 'Netflix', 'GOOG': 'Google', 'GOOGL': 'Alphabet', 'TSLA': 'Tesla', 'CSCO': 'Cisco',
    'INTC': 'Intel', 'ADBE': 'Adobe', 'CMCSA': 'Comcast', 'PEP': 'PepsiCo', 'COST': 'Costco',
    'AVGO': 'Broadcom', 'QCOM': 'Qualcomm', 'TMUS': 'T-Mobile', 'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials'
}

ustech_tickers = {
    "SMCI": "Super Micro Computer", "PLTR": "Palantir", "ORCL": "Oracle", "SNOW": "Snowflake",
    "NVDA": "NVIDIA", "AMD": "AMD", "MDB": "MongoDB", "DDOG": "Datadog", "CRWD": "CrowdStrike",
    "ZS": "Zscaler", "TSLA": "Tesla", "AAPL": "Apple", "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft"
}
eurostoxx_tickers = {
    'ASML.AS': 'ASML Holding', 'AIR.PA': 'Airbus', 'BAS.DE': 'BASF', 'BAYN.DE': 'Bayer',
    'BNP.PA': 'BNP Paribas', 'MBG.DE': 'Mercedes-Benz Group', 'ENEL.MI': 'Enel',
    'ENGI.PA': 'Engie', 'IBE.MC': 'Iberdrola', 'MC.PA': 'LVMH', 'OR.PA': 'L’Oréal',
    'PHIA.AS': 'Philips', 'SAN.PA': 'Sanofi', 'SAP.DE': 'SAP', 'SIE.DE': 'Siemens',
    'SU.PA': 'Schneider Electric', 'TTE.PA': 'TotalEnergies', 'VIV.PA': 'Vivendi',
    'AD.AS': 'Ahold Delhaize', 'CRH.L': 'CRH', 'DPW.DE': 'Deutsche Post', 'IFX.DE': 'Infineon',
    'ITX.MC': 'Inditex', 'MT.AS': 'ArcelorMittal', 'RI.PA': 'Pernod Ricard', 'STLA.MI': 'Stellantis',
    'UN01.DE': 'Uniper'
}
# --- Toevoeging tickers AMX & Crypto ---
amx_tickers = {
    "AMG.AS": "AMG", "ARCAD.AS": "Arcadis", "BAMNB.AS": "BAM Groep",
    "BPOST.AS": "BPost", "FAGR.AS": "Fagron", "FUR.AS": "Fugro", "KENDR.AS": "Kendrion",
    "SBMO.AS": "SBM Offshore", "TKWY.AS": "Just Eat", "VASTN.AS": "Vastned Retail"
}

crypto_tickers = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin"
}
# --- Update tab labels en bijbehorende mapping ---
tabs_mapping = {
    "🇺🇸 Dow Jones": dow_tickers,
    "🇺🇸 Nasdaq": nasdaq_tickers,
    "🇺🇸 US Tech": ustech_tickers,
    "🇪🇺 Eurostoxx": eurostoxx_tickers,
    "🇳🇱 AEX": aex_tickers,
    "🇳🇱 AMX": amx_tickers,
    "🌐 Crypto": crypto_tickers
}

tab_labels = list(tabs_mapping.keys())
selected_tab = st.radio("Kies beurs", tab_labels, horizontal=True)

valutasymbool = {
    "🇳🇱 AEX": "€ ",
    "🇳🇱 AMX": "€ ",
    "🇺🇸 Dow Jones": "$ ",
    "🇺🇸 Nasdaq": "$ ",
    "🇪🇺 Eurostoxx": "€ ",
    "🇺🇸 US Tech": "$ ",
    "🌐 Crypto": "",  # Geen symbool
}.get(selected_tab, "")

#def get_live_ticker_data(tickers_dict):
# --- Data ophalen voor dropdown live view ---
def get_live_ticker_data(tickers_dict):
    tickers = list(tickers_dict.keys())
    data = yf.download(tickers, period="1d", interval="1d", progress=False, group_by='ticker')
    result = []

    for ticker in tickers:
        try:
            last = data[ticker]['Close'].iloc[-1]
            prev = data[ticker]['Open'].iloc[-1]
            change = (last - prev) / prev * 100
            kleur = "#00FF00" if change > 0 else "#FF0000" if change < 0 else "#808080"
            naam = tickers_dict[ticker]
            result.append((ticker, naam, last, change, kleur))
        except Exception:
            continue

    return result

# --- Weergave dropdown met live info ---
live_info = get_live_ticker_data(tabs_mapping[selected_tab])
# --- Dropdown dictionary voorbereiden ---
dropdown_dict = {}  # key = ticker, value = (display_tekst, naam)

for t, naam, last, change, kleur in live_info:
    emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
    display = f"{t} - {naam} | {valutasymbool}{last:.2f} {emoji} {change:+.2f}%"
    dropdown_dict[t] = (display, naam)

# --- Bepalen van de juiste default key voor selectie
# Herstel vorige selectie als deze nog bestaat
default_ticker_key = st.session_state.get(f"ticker_select_{selected_tab}")
if default_ticker_key not in dropdown_dict:
    default_ticker_key = list(dropdown_dict.keys())[0]  # fallback naar eerste

# --- Dropdown zelf ---
selected_ticker = st.selectbox(
    f"Selecteer {selected_tab} ticker:",
    options=list(dropdown_dict.keys()),  # alleen tickers als stabiele optie-key
    format_func=lambda x: dropdown_dict[x][0],  # toon koers etc.
    key=f"ticker_select_{selected_tab}",
    index=list(dropdown_dict.keys()).index(default_ticker_key)
)

# --- Ophalen ticker info ---
ticker = selected_ticker
ticker_name = dropdown_dict[ticker][1]

# --- Live koers opnieuw ophalen voor de geselecteerde ticker ---
try:
    live_data = yf.download(ticker, period="1d", interval="1d", progress=False)
    last = live_data["Close"].iloc[-1]
except Exception:
    last = 0.0  # fallback

# --- Andere instellingen ---
# --- Intervalopties ---
interval_optie = st.selectbox(
    "Kies de interval",
    ["Dagelijks", "Wekelijks", "4-uur", "1-uur", "15-minuten"]
)

# Vertaal gebruikerskeuze naar Yahoo Finance intervalcode
interval_mapping = {
    "Dagelijks": "1d",
    "Wekelijks": "1wk",
    "4-uur": "4h",
    "1-uur": "1h",
    "15-minuten": "15m"
}

interval = interval_mapping[interval_optie]
# -------

# 📌 Titel en uitleg als toggle (zelfde stijl als eerder)
st.markdown("""
<style>
.sam-uitleg details[open] {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 90vw;
  max-width: 700px;
  z-index: 999;
  background-color: #f9f9f9;
  padding: 1em;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  border-radius: 10px;
}
</style>

<div class="sam-uitleg" style='display: flex; justify-content: space-between; align-items: top;'>
  <div style='flex: 1;'>
    <h4 style='margin-bottom: 10px;'>⚙️ Adviesgevoeligheid</h4>
  </div>
  <div style='flex: 1; text-align: right;'>
    <details>
      <summary style='cursor: pointer; font-weight: bold; color: #555;text-align: right;'>ℹ️ Uitleg SAM Trading Indicator</summary>
      <div style='margin-top: 10px;'>
        <p style='font-size: 13px; color: #333; text-align: left'>
        De gevoeligheidsslider bepaalt hoeveel opeenvolgende perioden met dezelfde trendrichting
        nodig zijn voordat een advies wordt afgegeven.<br><br>
        - Een lagere waarde (**1 of 2**) geeft sneller advieswijzigingen, maar is gevoeliger voor ruis.<br>
        - Een hogere waarde (**3 t/m 5**) geeft minder maar betrouwbaardere signalen.<br><br>
        De standaardwaarde is <strong>2</strong>.
        </p>   
      </div>
    </details>
  </div>
</div>
""", unsafe_allow_html=True)
# 📌 Titel en uitleg als toggle (zelfde stijl als eerder) werkt niet
#st.markdown("""
#<div style='display: flex; justify-content: space-between; align-items: flex-start; max-width: 900px; margin-bottom: 10px;'>
#  <div style='flex: 1; padding-right: 20px;'>
 #   <h4 style='margin-bottom: 10px;'>⚙️ Adviesgevoeligheid</h4>
#  </div>
#  <div style='flex: 1;'>
#    <details>
#      <summary style='cursor: pointer; font-weight: bold; color: #555;'>ℹ️ Uitleg Adviesgevoeligheid</summary>
#      <div style='margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 8px;'>
 #       <p style='font-size: 13px; color: #333; text-align: left;'>
 #       De gevoeligheidsslider bepaalt hoeveel opeenvolgende perioden met dezelfde trendrichting
  #      nodig zijn voordat een advies wordt afgegeven.<br><br>
 #       - Een lagere waarde (**1 of 2**) geeft sneller advieswijzigingen, maar is gevoeliger voor ruis.<br>
 #       - Een hogere waarde (**3 t/m 5**) geeft minder maar betrouwbaardere signalen.<br><br>
  #      De standaardwaarde is <strong>2</strong>.
 #       </p>
 #     </div>
#    </details>
#  </div>
#</div>
#""", unsafe_allow_html=True)

# 📌 Slider in kolommen, links met max 50% breedte
col1, col2 = st.columns([1, 1])
with col1:
    thresh = st.slider("Aantal perioden met dezelfde richting voor advies", 1, 5, 2, step=1)
with col2:
    pass  # lege kolom, zodat slider links blijft

# oude
#col1, col2 = st.columns([9, 6])  # Pas verhouding aan als je wilt


#with col1:
 #   st.markdown("### ⚙️ Adviesgevoeligheid")
#    thresh = st.slider("Aantal perioden met dezelfde richting voor advies", 1, 5, 2, step=1)
    
#with col2:
#    with st.expander("ℹ️ Uitleg Adviesgevoeligheid"):
  #      st.markdown(
   #         """
#            <div style='color:#444; font-size:12px;'>
 #           De gevoeligheidsslider bepaalt hoeveel opeenvolgende perioden met dezelfde trendrichting
 #           nodig zijn voordat een advies wordt afgegeven.<br>
  #          Een lagere waarde (1 of 2) geeft sneller advies wijzigingen, maar is gevoeliger voor ruis.
 #           Een hogere waarde (3 of 4) geeft dus minder maar vaak betrouwbaardere adviezen.<br>
   #         De standaardwaarde is 2.
   #         </div>
  #          """,
 #           unsafe_allow_html=True
  #      )

  #

#thresh = st.slider("Aantal perioden met dezelfde richting voor advies", 1, 5, 2, step=1)
#thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 0.5, 0.1, step=0.02)

# Berekening
# ✅ Gecombineerde functie met cache
@st.cache_data(ttl=900)  # Cache 15 minuten
def advies_wordt_geladen(ticker, interval, threshold):
    df = fetch_data(ticker, interval)
    if df.empty or "Close" not in df.columns or "Open" not in df.columns:
        return None, None  # Signaal dat data niet bruikbaar is
    df = calculate_sam(df)
    df, huidig_advies = determine_advice(df, threshold=threshold)
    return df, huidig_advies

# ✅ Gebruik en foutafhandeling
df, huidig_advies = advies_wordt_geladen(ticker, interval, thresh)
#df, huidig_advies = get_sam_met_advies(ticker, interval, thresh)

if df is None or df.empty:
    st.error("❌ Geen geldige data opgehaald. Kies een andere ticker of interval.")
    st.stop()

#df = fetch_data(ticker, interval)

#if df.empty:
#    st.error("❌ Geen geldige data opgehaald. Kies een andere ticker of interval.")
#    st.stop()
#df = calculate_sam(df)
#df = determine_advice(df, threshold=thresh)
#df, huidig_advies = determine_advice(df, threshold=thresh)

# debugging tools
#st.subheader("🔍 SAM Debug-tabel (laatste 8 rijen)")
#st.dataframe(
#    df[["Close", "SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX", "SAM"]].tail(180),
#    use_container_width=True
#)
#st.caption(f"SAM-componenten gemiddeld: "
#           f"SAMK={df['SAMK'].mean():+.2f}, "
 #          f"SAMG={df['SAMG'].mean():+.2f}, "
#           f"SAMT={df['SAMT'].mean():+.2f}, "
#           f"SAMD={df['SAMD'].mean():+.2f}, "
 #          f"SAMM={df['SAMM'].mean():+.2f}, "
 #          f"SAMX={df['SAMX'].mean():+.2f}, "
  #         f"SAM totaal={df['SAM'].mean():+.2f}")

# Grafieken

# Kleur bepalen op basis van advies
advies_kleur = "green" if huidig_advies == "Kopen" else "red" if huidig_advies == "Verkopen" else "gray"

# Titel met kleur en grootte tonen - indicator
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Advies voor:")
#    st.markdown(
#    f"""
#    Advies voor:
#    """,
#    unsafe_allow_html=True
#)
with col2:
    st.markdown(
    f"""
    <h3><span style='color:#3366cc'>{ticker_name}</span</h3>
    """,
    unsafe_allow_html=True
   )

# Titel met kleur en grootte tonen - advies
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Huidig advies:")
#    st.markdown(
#    f"""
#    <h5 style='color:{advies_kleur}'>Huidig advies:</h5>
#    """,
 #   unsafe_allow_html=True
#)
with col2:
    st.markdown(
    f"""
    <h2 style='color:{advies_kleur}'>{huidig_advies}</h2>
    """,
    unsafe_allow_html=True
    )

# ⏱ gecompliceerde koersgrafiek werkt niet geheel
# bepaal data weeergaveperiode op basis van interval
#grafiek_periode = bepaal_grafiekperiode(interval)

# Bepaal cutoff-datum
#cutoff_datum = df.index.max() - grafiek_periode

# Filter alleen grafiekdata
#df_grafiek = df[df.index >= cutoff_datum].copy()

#cutoff_datum = datetime.now() - bepaal_grafiekperiode(interval)
#df_filtered = df[df.index >= cutoff_datum]

# 🖼️ Toggle voor grafiek
#if st.toggle("📊 Toon koersgrafiek"):
 #   fig = go.Figure(data=[
#        go.Candlestick(
 #           x=df_filtered.index,
#            open=df_filtered["Open"],
 #           high=df_filtered["High"],
  #          low=df_filtered["Low"],
  #          close=df_filtered["Close"],
 #           increasing_line_color='green',
 #           decreasing_line_color='red',
#            name='Koers'
#        )
#    ])

 #   fig.update_layout(
 #       xaxis_title="Datum",
 #       yaxis_title="Koers",
 #       xaxis_rangeslider_visible=False,
 #       height=400,
 #       margin=dict(l=10, r=10, t=10, b=10)
#    )

#    st.plotly_chart(fig, use_container_width=True)

# simpele koersgrafiek
# ⏳ Toggle voor koersgrafiekb>📈 "📉  Voorbeeld:</b>
toon_koersgrafiek = st.toggle("📈 Toon koersgrafiek", value=False)

if toon_koersgrafiek:
    # 📅 Bepaal grafiekperiode
    grafiek_periode = bepaal_grafiekperiode(interval)
    cutoff_datum = df.index.max() - grafiek_periode
    df_koers = df[df.index >= cutoff_datum].copy()  # Alleen koers in periode

    # ✅ Bereken MA's op de volledige dataset
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA150"] = df["Close"].rolling(window=150).mean()

    # 📊 Plot met lijnen
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot koers alleen voor gekozen periode
    ax.plot(df_koers.index, df_koers["Close"], color="black", linewidth=2.0, label="Koers")

    # Plot MA-lijnen vanuit volledige dataset
    ax.plot(df.index, df["MA30"], color="orange", linewidth=1.0, label="MA(30)")
    ax.plot(df.index, df["MA150"], color="pink", linewidth=1.0, label="MA(150)")

    # Beperk x-as op koersperiode
    ax.set_xlim(df_koers.index.min(), df_koers.index.max())

    # ➕ y-as: bepaal min/max + marge (veilig en robuust)
    try:
        close_series = df_koers["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]  # neem eerste kolom als DataFrame
        koers_values = close_series.astype(float).dropna()

        if not koers_values.empty:
            koers_min = koers_values.min()
            koers_max = koers_values.max()
            marge = (koers_max - koers_min) * 0.05
            ax.set_ylim(koers_min - marge, koers_max + marge)
    except Exception as e:
        st.warning(f"Kon y-as limieten niet instellen: {e}")

    # Opmaak
    ax.set_title(f"Koersgrafiek van {ticker_name}")
    ax.set_ylabel("Close")
    ax.set_xlabel("Datum")
    ax.legend()
    fig.tight_layout()
    st.subheader("Koersgrafiek")

    st.pyplot(fig)

# --- Grafiek met SAM en Trend ---
st.subheader("Grafiek met SAM en Trend")

# Bepaal de weergaveperiode op basis van interval
grafiek_periode = bepaal_grafiekperiode(interval)

# Bepaal cutoff-datum
cutoff_datum = df.index.max() - grafiek_periode

# Filter alleen grafiekdata
df_grafiek = df[df.index >= cutoff_datum].copy()

# --- Grafiek met SAM en Trend (aangepast) ---
fig, ax = plt.subplots(figsize=(10, 4))

# ✅ Kleuren voor SAM afhankelijk van positief/negatief
kleuren = ["green" if val >= 0 else "red" for val in df_grafiek["SAM"]]
# ✅ Bars voor SAM
ax.bar(df_grafiek.index, df_grafiek["SAM"], color=kleuren, label="SAM")
ax.set_xlim(df_grafiek.index.min(), df_grafiek.index.max())
# ✅ Trendlijn (zelfde as)
ax.plot(df_grafiek.index, df_grafiek["Trend"], color="blue", linewidth=2, label="Trend")
# ✅ Nullijn
ax.axhline(y=0, color="black", linewidth=1, linestyle="--")
# ✅ Geforceerde y-as
ax.set_ylim(-4.5, 4.5)
# ✅ Titel en labels
ax.set_title("SAM-indicator en Trendlijn")
ax.set_ylabel("Waarde")
# ✅ Legenda toevoegen
ax.legend()

fig.tight_layout()
st.pyplot(fig)

# --- Grafiek met SAM en Trend  - oud ---
#fig, ax1 = plt.subplots(figsize=(10, 4))
#ax1.bar(df_grafiek.index, df_grafiek["SAM"], color="lightblue", label="SAM")
#ax1.axhline(y=0, color="black", linewidth=1, linestyle="--")  # nullijn
#ax2 = ax1.twinx()
#ax2.plot(df_grafiek.index, df_grafiek["Trend"], color="red", label="Trend")
#ax1.set_ylabel("SAM")
#ax2.set_ylabel("Trend")
#fig.tight_layout()
#st.pyplot(fig)

# --- Tabel met signalen en rendement ---
st.subheader("Laatste signalen en rendement")

# Kolommen selecteren en formatteren
kolommen = ["Close", "Advies", "SAM", "Trend", "Markt-%", "SAM-%"]
#tabel = df[kolommen].dropna().tail(30).round(3).copy()
tabel = df[kolommen].dropna().copy()
tabel = tabel.sort_index(ascending=False).head(30) # lengte tabel hier!

# Datumkolom aanmaken vanuit index
if not isinstance(tabel.index, pd.DatetimeIndex):
    tabel.index = pd.to_datetime(tabel.index, errors="coerce")
tabel = tabel[~tabel.index.isna()]
tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")

# Zet kolomvolgorde
tabel = tabel[["Datum"] + kolommen]

# Afronding en formatting
if selected_tab == "🌐 Crypto":
    tabel["Close"] = tabel["Close"].map("{:.3f}".format)
else:
    tabel["Close"] = tabel["Close"].map("{:.2f}".format)
# ✅ Eerst de juiste berekening behouden als float (NIET afronden!)
tabel["Markt-%"] = tabel["Markt-%"].astype(float) * 100
tabel["SAM-%"] = tabel["SAM-%"].astype(float) * 100

# ✅ Daarna afzonderlijke kolommen voor weergave formatteren
tabel["Markt-% weergave"] = tabel["Markt-%"].map("{:+.2f}%".format)
tabel["SAM-% weergave"] = tabel["SAM-%"].map("{:+.2f}%".format)
tabel["Trend Weergave"] = tabel["Trend"].map("{:+.3f}".format)

tabel = tabel[["Datum", "Close", "Advies", "SAM", "Trend Weergave", "Markt-% weergave", "SAM-% weergave"]]
tabel = tabel.rename(columns={
    "Markt-% weergave": "Markt-%",
    "SAM-% weergave": "SAM-%",
    "Trend Weergave": "Trend"
})

# HTML-rendering
html = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    th {
        background-color: #004080;
        color: white;
        padding: 6px;
        text-align: center;
    }
    td {
        border: 1px solid #ddd;
        padding: 6px;
        text-align: right;
        background-color: #f9f9f9;
        color: #222222;
    }
    tr:nth-child(even) td {
        background-color: #eef2f7;
    }
    tr:hover td {
        background-color: #d0e4f5;
    }
</style>
<table>
    <thead>
        <tr>
            <th style='width: 110px;'>Datum</th>
            <th style='width: 80px;'>Close</th>
            <th style='width: 90px;'>Advies</th>
            <th style='width: 60px;'>SAM</th>
            <th style='width: 70px;'>Trend</th>
            <th style='width: 90px;'>Markt-%</th>
            <th style='width: 90px;'>SAM-%</th>
        </tr>
    </thead>
    <tbody>
"""

# Voeg rijen toe aan de tabel
for _, row in tabel.iterrows():
    html += "<tr>"
    for value in row:
        html += f"<td>{value}</td>"
    html += "</tr>"

html += "</tbody></table>"

#Weergave in Streamlit
st.markdown(html, unsafe_allow_html=True)



## 📊 Backtestfunctie: sluit op close van nieuw signaal
# ✅ 0.Data voorbereiden voor advies')
df_signalen = df.copy()
if "Advies" not in df_signalen.columns:
    st.error("Kolom 'Advies' ontbreekt in de data.")
    st.stop()

                
st.subheader("Vergelijk Marktrendement en SAM-rendement")

# 📅 1. Datumkeuze
current_year = date.today().year
default_start = date(current_year, 1, 1)
default_end = df.index.max().date()

start_date = st.date_input("Startdatum analyse", default_start)
end_date = st.date_input("Einddatum analyse", default_end)

# 📆 2. Filter op periode
df = df.copy()
df.index = pd.to_datetime(df.index)
df_period = df.loc[
    (df.index.date >= start_date) & (df.index.date <= end_date)
].copy()

# 🧹 Flatten MultiIndex indien nodig
if isinstance(df_period.columns, pd.MultiIndex):
    df_period.columns = ["_".join([str(i) for i in col if i]) for col in df_period.columns]

# 🔍 Zoek geldige 'Close'-kolom
close_col = next((col for col in df_period.columns if col.lower().startswith("close")), None)

if not close_col:
    st.error("❌ Geen geldige 'Close'-kolom gevonden in deze dataset.")
    st.stop()

# 📈 Marktrendement (Buy & Hold)
df_period[close_col] = pd.to_numeric(df_period[close_col], errors="coerce")
df_valid = df_period[close_col].dropna()

marktrendement = None
if len(df_valid) >= 2 and df_valid.iloc[0] != 0.0:
    koers_start = df_valid.iloc[0]
    koers_eind = df_valid.iloc[-1]
    marktrendement = ((koers_eind - koers_start) / koers_start) * 100

# ✅ Signaalkeuze geforceerd op Beide
signaalkeuze = "Beide"
advies_col = "Advies"

# Vind eerste rij waar 'Trail' >= threshold en dan pas beginnen
eerste_valid_index = df_period.index[df_period["Trail"] >= thresh][0]
df_signalen = df_period.loc[eerste_valid_index:]
df_signalen = df_signalen[df_signalen[advies_col].isin(["Kopen", "Verkopen"])].copy()

# 🔄 Backtestfunctie

#def bereken_sam_rendement(df_signalen, signaal_type="Beide", close_col="Close"):
def bereken_sam_rendement(df_signalen, signaal_type="Beide", close_col="Close"):
    rendementen = []
    trades = []
    entry_price = None
    entry_date = None
    entry_type = None

    type_map = {"Koop": "Kopen", "Verkoop": "Verkopen", "Beide": "Beide"}
    mapped_type = type_map.get(signaal_type, "Beide")

    for i in range(len(df_signalen)):
        advies = df_signalen["Advies"].iloc[i]
        close = df_signalen[close_col].iloc[i]
        datum = df_signalen.index[i]

        # Alleen sluiten als er een open positie is
        if entry_type is not None:
            if advies != entry_type and (mapped_type == "Beide" or entry_type == mapped_type):
                sluit_datum = datum
                sluit_close = close

                if entry_type == "Kopen":
                    rendement = (sluit_close - entry_price) / entry_price * 100
                else:
                    rendement = (entry_price - sluit_close) / entry_price * 100

                # Nieuw: filter dummy-trade
                if entry_price != sluit_close and entry_date != sluit_datum:
                    rendementen.append(rendement)
                    trades.append({
                        "Type": entry_type,
                        "Open datum": entry_date.date(),
                        "Open prijs": entry_price,
                        "Sluit datum": sluit_datum.date(),
                        "Sluit prijs": sluit_close,
                        "Rendement (%)": rendement,
                        "SAM": df.loc[entry_date, "SAM"] if entry_date in df.index else None,
                        "Trend": df.loc[entry_date, "Trend"] if entry_date in df.index else None,
                    })

                # Mogelijk nieuwe trade openen
                if mapped_type == "Beide" or advies == mapped_type:
                    entry_type = advies
                    entry_price = close
                    entry_date = datum
                else:
                    entry_type = None
                    entry_price = None
                    entry_date = None

        # Start een nieuwe trade, zonder geforceerd te zijn
        elif advies in ["Kopen", "Verkopen"] and (mapped_type == "Beide" or advies == mapped_type):
            entry_type = advies
            entry_price = close
            entry_date = datum

    # Eventueel open trade afsluiten op laatste koers
    if entry_type and entry_price is not None:
        laatste_datum = df_signalen.index[-1]
        laatste_koers = df_signalen[close_col].iloc[-1]

        if entry_type == "Kopen":
            rendement = (laatste_koers - entry_price) / entry_price * 100
        else:
            rendement = (entry_price - laatste_koers) / entry_price * 100

        # Nieuw: filter dummy-trade (ook laatste)
        if entry_price != laatste_koers and entry_date != laatste_datum:
            rendementen.append(rendement)
            trades.append({
                "Type": entry_type,
                "Open datum": entry_date.date(),
                "Open prijs": entry_price,
                "Sluit datum": laatste_datum.date(),
                "Sluit prijs": laatste_koers,
                "Rendement (%)": rendement
            })

    sam_rendement = sum(rendementen) if rendementen else 0.0
    return sam_rendement, trades, rendementen

# ✅ 4. Berekening
#sam_rendement, trades, rendementen = bereken_sam_rendement(df, signaalkeuze, close_col)

sam_rendement, trades, rendementen = bereken_sam_rendement(df_signalen, signaalkeuze, close_col)

# ✅ 5. Visueel weergeven
col1, col2 = st.columns(2)
col1.metric("Marktrendement (Buy & Hold)", f"{marktrendement:+.2f}%" if marktrendement is not None else "n.v.t.")
col2.metric("📊 SAM-rendement", f"{sam_rendement:+.2f}%" if isinstance(sam_rendement, (int, float)) else "n.v.t.")

if trades:
    df_trades = pd.DataFrame(trades)

    df_trades["SAM-% Koop"] = df_trades.apply(lambda row: row["Rendement (%)"] if row["Type"] == "Kopen" else None, axis=1)
    df_trades["SAM-% Verkoop"] = df_trades.apply(lambda row: row["Rendement (%)"] if row["Type"] == "Verkopen" else None, axis=1)
    df_trades["Markt-%"] = df_trades.apply(
        lambda row: ((row["Sluit prijs"] - row["Open prijs"]) / row["Open prijs"]) * 100, axis=1)

    # Kopie voor weergave
    df_display = df_trades.copy()

   # Formatteringskolommen
    for col in ["Markt-%", "Rendement (%)", "SAM-% Koop", "SAM-% Verkoop"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].astype(float)

    df_display = df_display.rename(columns={"Rendement (%)": "SAM-% tot."})
    df_display = df_display[[
        "Open datum", "Open prijs", "Sluit datum", "Sluit prijs",
        "Markt-%", "SAM-% tot.", "SAM-% Koop", "SAM-% Verkoop"]]

    # Succes-analyses
    aantal_trades = len(df_display)
    aantal_koop = df_display["SAM-% Koop"].notna().sum()
    aantal_verkoop = df_display["SAM-% Verkoop"].notna().sum()
    rendement_totaal = df_display["SAM-% tot."].sum()
    rendement_koop = df_display["SAM-% Koop"].sum(skipna=True)
    rendement_verkoop = df_display["SAM-% Verkoop"].sum(skipna=True)
    aantal_succesvol = (df_display["SAM-% tot."] > 0).sum()
    aantal_succesvol_koop = (df_display["SAM-% Koop"] > 0).sum()
    aantal_succesvol_verkoop = (df_display["SAM-% Verkoop"] > 0).sum()

    st.caption(f"Aantal afgeronde **trades**: **{aantal_trades}**, totaal resultaat SAM-%: **{rendement_totaal:+.2f}%**, aantal succesvol: **{aantal_succesvol}**")
    st.caption(f"Aantal **koop** trades: **{aantal_koop}**, SAM-% koop: **{rendement_koop:+.2f}%**, succesvol: **{aantal_succesvol_koop}**")
    st.caption(f"Aantal **verkoop** trades: **{aantal_verkoop}**, SAM-% verkoop: **{rendement_verkoop:+.2f}%**, succesvol: **{aantal_succesvol_verkoop}**")

    def kleur_positief_negatief(val):
        if pd.isna(val): return "color: #808080"
        if val > 0: return "color: #008000"
        if val < 0: return "color: #FF0000"
        return "color: #808080"

    kleurbare_kolommen = ["Markt-%", "SAM-% tot.", "SAM-% Koop", "SAM-% Verkoop"]
    toon_alle = st.toggle("Toon alle trades", value=False)
    df_display = df_display if toon_alle or len(df_display) <= 12 else df_display.iloc[-12:]

    # Afronding van 'Open prijs' en 'Sluit prijs' op basis van type asset
    if selected_tab == "🌐 Crypto":
        df_display["Open prijs"] = df_display["Open prijs"].map("{:.3f}".format)
        df_display["Sluit prijs"] = df_display["Sluit prijs"].map("{:.3f}".format)
    else:
        df_display["Open prijs"] = df_display["Open prijs"].map("{:.2f}".format)
        df_display["Sluit prijs"] = df_display["Sluit prijs"].map("{:.2f}".format)
    
    # ✅ Laatste stap: toon als tabel
    geldige_kolommen = [col for col in kleurbare_kolommen if df_display[col].notna().any()]
    
    # ✅ Eerst kleuren toepassen, dan formatteren (anders krijg je een TypeError)
    styler = df_display.style.applymap(kleur_positief_negatief, subset=geldige_kolommen)
    styler = styler.format({col: "{:+.2f}%" for col in geldige_kolommen})

    st.dataframe(styler, use_container_width=True)

else:
    st.info("ℹ️ Geen trades gevonden binnen de geselecteerde periode.")
    
    













# wit
#with st.container():
#    st.markdown(
#        """
#        <div style='
 #           background-color: #f0f2f6;
 #           border-radius: 12px;
   #         padding: 20px 25px;
  #          margin-top: 25px;
  #          margin-bottom: 25px;
  #          box-shadow: 0 4px 8px rgba(0,0,0,0.05);
 #       '>
  #          <h3 style='margin-bottom:10px; color:#2c3e50;'>⚙️ Instellingen voor Adviesgevoeligheid</h3>
  #          <p style='margin-top:0; color:#6c757d;'>Kies hoe sterk de trend moet zijn voordat een advies volgt. Hogere waarde betekent meer bevestiging vereist.</p>
  #      </div>
  #      """,
   #     unsafe_allow_html=True
 #   )









    



# wit
#  if entry_type is None and advies in ["Kopen", "Verkopen"]:
#            if mapped_type == "Beide" or advies == mapped_type:
#                entry_type = advies
 #               entry_price = close
 #               entry_date = datum
 #               continue
#  else:  




# wit




                    
