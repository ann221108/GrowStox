import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="AI Stock WebApp", layout="wide")
st.image("growstox_logo.png", width=120)
st.title("GrowStox — AI-Powered Stock Market WebApp")
# Define tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict 5-Day Prices", "📊 Compare Stocks", "🤖 AI Stock Predictor", "📰 Financial Newsletter"])

# ---------------- Tab 1: 5-Day Forecast ---------------- #
with tab1:
    st.header("🔮 Predict Stock Prices for Next 5 Days")

    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS):", "AAPL")

    if st.button("Predict 5 Days"):
        data = yf.download(symbol, period="1y")

        if data.empty:
            st.error("No data found. Please check the symbol.")
        else:
            df = data[['Close']].copy()
            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Close']].values
            y = df['Target'].values

            model = LinearRegression()
            model.fit(X, y)

            last_price = float(df['Close'].iloc[-1])
            future_prices = []

            for _ in range(5):
                input_array = np.array([[last_price]])
                predicted = model.predict(input_array)[0]
                future_prices.append(predicted)
                last_price = float(predicted)

            st.subheader("📅 Predicted Prices")
            for i, price in enumerate(future_prices, 1):
                st.write(f"Day {i}: **${price:.2f}**")

            # Plot
            st.subheader("📈 Forecast Plot")
            fig, ax = plt.subplots()
            ax.plot(range(1, 6), future_prices, marker='o', linestyle='--', color='green')
            ax.set_xlabel("Day")
            ax.set_ylabel("Predicted Price")
            ax.set_title(f"5-Day Forecast for {symbol}")
            st.pyplot(fig)

# ---------------- Tab 2: Compare Stocks ---------------- #
with tab2:
    st.header("📊 Compare Stocks")

    stock_options = {
        "Apple (AAPL)": "AAPL",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "Google (GOOGL)": "GOOGL",
        "Microsoft (MSFT)": "MSFT",
        "Infosys (INFY.NS)": "INFY.NS",
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS"
    }

    selections = st.multiselect("Select up to 3 stocks to compare:", list(stock_options.keys()), default=["Apple (AAPL)"])
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

    if 1 <= len(selections) <= 3:
        fig, ax = plt.subplots()
        latest_data = {}

        for name in selections:
            ticker = stock_options[name]
            data = yf.download(ticker, start=start_date, end=end_date)

            if not data.empty:
                ax.plot(data['Close'], label=name)
                latest_data[name] = data
            else:
                st.warning(f"No data for {name}")

        ax.set_title("Stock Closing Price Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Table
        st.subheader("📄 Latest Stock Prices")
        columns = st.columns(len(latest_data))
        for idx, (name, df) in enumerate(latest_data.items()):
            with columns[idx]:
                st.markdown(f"**{name}**")
                st.dataframe(df.tail())

    else:
        st.info("Please select between 1 to 3 stocks.")

# ---------------- Tab 3: AI Stock Predictor ---------------- #
with tab3:
    st.header("🤖 Predict Tomorrow’s Closing Price")

    ai_symbol = st.text_input("Enter Stock Symbol for AI Prediction:", "TSLA")

    if st.button("Run AI Predictor"):
        data = yf.download(ai_symbol, period="6mo")
        if data.empty:
            st.error("Data not available for that ticker.")
        else:
            df = data[['Close']].copy()
            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Close']].values
            y = df['Target'].values

            model = LinearRegression()
            model.fit(X, y)

            today_price = float(df['Close'].iloc[-1])
            predicted_price = model.predict(np.array([[today_price]]))[0]

            st.success(f"📌 Predicted Closing Price for Tomorrow: **${predicted_price:.2f}**")

            # Visual
            st.subheader("📉 Trend + Prediction")
            fig, ax = plt.subplots()
            ax.plot(df.index[-50:], df['Close'].tail(50), label="Actual")
            ax.scatter(df.index[-1], predicted_price, color='red', label="Prediction")
            ax.set_title(f"Recent Trend + Prediction for {ai_symbol}")
            ax.legend()
            st.pyplot(fig)
with tab4:
    st.header("📰 Weekly Digest - Stock Market Insights")

    st.markdown("### 📅 This Week at a Glance")
    st.write("""
    **Monday**  
    📉 Markets opened slightly lower as investors awaited inflation data.  
    **Tip:** Don't react to every Monday dip — zoom out, not in!

    **Tuesday**  
    🧠 AI stocks boosted the tech sector after Nvidia’s earnings beat expectations.  
    **Term to Know:** *Volatility* — How much a stock price swings.

    **Wednesday**  
    💬 The Fed held interest rates steady. No hike = market calm.  
    **Tip:** Rate decisions can influence stock & bond prices.

    **Thursday**  
    🚀 Mid-cap stocks led gains, driven by optimism in manufacturing.  
    **Did You Know?** Mid-caps often outperform in recovery cycles.

    **Friday**  
    📊 Markets closed the week higher. Energy stocks rebounded.  
    **Quick Tip:** Check earnings season calendars to stay ahead.
    """)

    st.markdown("### 💡 Beginner’s Corner: What Should You Focus On?")
    st.success("""
    - Start with **index funds** or ETFs if you're new.
    - Don’t try to time the market — focus on consistency.
    - Learn basic indicators like *P/E ratio*, *moving average*, and *volume*.
    """)

    st.markdown("### 💬 Quote of the Week")
    st.info("“An investment in knowledge pays the best interest.” – Benjamin Franklin")

    st.markdown("### 📘 Financial Term of the Week")
    st.info("**Bear Market** — A market condition where prices fall 20% or more from recent highs.")

    st.markdown("### 🔗 Trusted Resources to Explore")
    st.markdown("""
    - [📈 Bloomberg Markets](https://www.bloomberg.com/markets)
    - [💼 CNBC Finance](https://www.cnbc.com/finance/)
    - [📚 Investopedia Education](https://www.investopedia.com/)
    """)

    st.markdown("### 🧠 Quick Fun Fact")
    st.info("Apple Inc. was the first U.S. company to reach a $3 trillion market cap — in 2022!")

