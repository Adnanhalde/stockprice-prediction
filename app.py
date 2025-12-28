import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="centered"
)

# ----------------------------------
# CUSTOM CSS (PREMIUM UI)
# ----------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

/* Signals */
.buy {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    color: white;
}

.sell {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    color: white;
}

.hold {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------
st.title("📈 Stock Price Prediction App")
st.caption("Live stock price • Trend analysis • ML-based prediction")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.header("🔍 Select Stock")

stocks_df = pd.read_csv("stocks.csv")
stocks_df.columns = stocks_df.columns.str.strip()
stocks_df["display"] = stocks_df["Name"] + " (" + stocks_df["Symbol"] + ")"

selected_stock = st.sidebar.selectbox(
    "Search stock by name or symbol",
    stocks_df["display"]
)

ticker = stocks_df.loc[
    stocks_df["display"] == selected_stock, "Symbol"
].values[0]

# ----------------------------------
# FETCH DATA
# ----------------------------------
with st.spinner("Fetching stock data..."):
    data = yf.download(ticker, period="3y", progress=False)

if data.empty:
    st.error("❌ No data found")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ----------------------------------
# CURRENT PRICE
# ----------------------------------
current_price = float(data["Close"].iloc[-1])
previous_price = float(data["Close"].iloc[-2])

price_change = current_price - previous_price
percent_change = (price_change / previous_price) * 100

st.markdown('<div class="card">', unsafe_allow_html=True)
st.metric(
    "💰 Current Price",
    f"${current_price:.2f}",
    f"{price_change:.2f} ({percent_change:.2f}%)"
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------
# MOVING AVERAGES
# ----------------------------------
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# ----------------------------------
# SIGNAL LOGIC
# ----------------------------------
ma20_today = data["MA20"].iloc[-1]
ma50_today = data["MA50"].iloc[-1]
ma20_yesterday = data["MA20"].iloc[-2]
ma50_yesterday = data["MA50"].iloc[-2]

signal = "HOLD"

if ma20_yesterday < ma50_yesterday and ma20_today > ma50_today:
    signal = "BUY"
elif ma20_yesterday > ma50_yesterday and ma20_today < ma50_today:
    signal = "SELL"

# ----------------------------------
# DISPLAY SIGNAL
# ----------------------------------
st.subheader("📌 Trading Signal")

if signal == "BUY":
    st.markdown('<div class="buy">✅ BUY — Uptrend Started</div>', unsafe_allow_html=True)
elif signal == "SELL":
    st.markdown('<div class="sell">❌ SELL — Downtrend Started</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="hold">⏸️ HOLD — No Clear Trend</div>', unsafe_allow_html=True)

# ----------------------------------
# CHART
# ----------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Price Chart with Moving Averages")
st.line_chart(data[["Close", "MA20", "MA50"]])
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------
# MACHINE LEARNING
# ----------------------------------
st.subheader("🤖 Machine Learning Prediction")

data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Close"]]
y = data["Next_Close"]

model = LinearRegression()
model.fit(X, y)

if st.button("🔮 Predict Next Day Price"):
    prediction = model.predict([[current_price]])
    st.success(f"📈 Predicted Next Day Price: ${prediction[0]:.2f}")
    st.caption("Prediction uses Linear Regression on historical prices")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption("⚠️ Educational purpose only. Not financial advice.")
