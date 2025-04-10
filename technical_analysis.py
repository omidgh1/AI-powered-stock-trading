from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pytz
import math
from utils import scale_dataframe


def financial_overview_info(df):
    df_scaled = scale_dataframe(df, 0.0000001, 4)
    result = {}
    for var in df.index:
        result[var] = {}
        for year in df.columns:
            date_value = str(year.strftime('%Y-%m'))
            value = df_scaled.loc[var][year]
            result[var][date_value] = float(value)

    return result


def major_holders_info(df):
    cleaned_major_holders = {key: val['Value'] for key, val in df.to_dict(orient="index").items()}
    return cleaned_major_holders


def institutional_holders_info(df):
    df[['Shares', 'Value']] = scale_dataframe(df[['Shares', 'Value']], 0.0000001, 4)
    data = df[['Holder', 'pctHeld', 'Shares', 'Value', 'pctChange']].to_dict(orient="index")
    institutional_holders = {}
    for index, entry in data.items():
        holder_name = entry['Holder']
        # Remove the 'Holder' key and round the other values
        cleaned_entry = {key: round(val, 3) for key, val in entry.items() if key != 'Holder'}
        institutional_holders[holder_name] = cleaned_entry
    return institutional_holders


def history_technical_indicators(df):
    """Calculate multiple technical indicators and generate investment signals."""
    df = df.copy()
    today = datetime.now(pytz.timezone('America/New_York'))

    # Moving Averages
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # RSI Calculation
    delta = df["Close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Williams %R
    df["Williams %R"] = ((df["High"].rolling(14).max() - df["Close"]) /
                         (df["High"].rolling(14).max() - df["Low"].rolling(14).min())) * -100

    # Parabolic SAR
    df["PSAR"] = df["Close"].shift(1) + 0.02 * (df["High"].shift(1) - df["Low"].shift(1))

    # Momentum Indicator
    df["Momentum"] = df["Close"].diff(10)

    # Rate of Change (ROC)
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

    # Fibonacci Retracement Levels (key support/resistance)
    high_price = df["High"].max()
    low_price = df["Low"].min()
    df["Fib_0.236"] = low_price + 0.236 * (high_price - low_price)
    df["Fib_0.382"] = low_price + 0.382 * (high_price - low_price)
    df["Fib_0.5"] = low_price + 0.5 * (high_price - low_price)
    df["Fib_0.618"] = low_price + 0.618 * (high_price - low_price)

    # Ichimoku Cloud
    df["Tenkan"] = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    df["Kijun"] = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    df["Senkou A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    df["Senkou B"] = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)

    # Compute buy/sell signals
    latest = df.iloc[-1]
    signals = {"datetime": str(today.strftime('%Y-%m-%dT%H:%M')),
               "Golden Cross": bool(latest["SMA_50"] > latest["SMA_200"]),
               "Death Cross": bool(latest["SMA_50"] < latest["SMA_200"]),
               "RSI Signal": "Overbought (Sell)" if latest["RSI"] > 70 else "Oversold (Buy)" if latest[
                                                                                                    "RSI"] < 30 else "Neutral",
               "MACD Signal": "Bullish (Buy)" if latest["MACD"] > latest["Signal_Line"] else "Bearish (Sell)",
               "Williams %R Signal": "Overbought (Sell)" if latest["Williams %R"] > -20 else "Oversold (Buy)" if latest[
                                                                                                                     "Williams %R"] < -80 else "Neutral",
               "Parabolic SAR Signal": "Bullish (Buy)" if latest["PSAR"] < latest["Close"] else "Bearish (Sell)",
               "Momentum Signal": "Bullish (Buy)" if latest["Momentum"] > 0 else "Bearish (Sell)",
               "ROC Signal": "Bullish (Buy)" if latest["ROC"] > 0 else "Bearish (Sell)",
               "Fib Resistance": "Above 0.618 Level" if latest["Close"] > latest[
                   "Fib_0.618"] else "Below 0.236 Level (Support)" if latest["Close"] < latest[
                   "Fib_0.236"] else "In Between",
               "Ichimoku Signal": "Bullish (Buy)" if latest["Close"] > latest["Senkou A"] else "Bearish (Sell)"}

    decision = {"Short Term": None, "Mid Term": None, "Long Term": None}

    # Short-term Decision
    if signals["RSI Signal"] == "Oversold (Buy)" and signals["MACD Signal"] == "Bullish (Buy)" and signals[
        "Momentum Signal"] == "Bullish (Buy)":
        decision["Short Term"] = "Good for short-term investment"
    elif signals["RSI Signal"] == "Overbought (Sell)" or signals["MACD Signal"] == "Bearish (Sell)" or signals[
        "Momentum Signal"] == "Bearish (Sell)":
        decision["Short Term"] = "Not good for short-term investment"
    else:
        decision["Short Term"] = "Neutral short-term outlook"

    # Mid-term Decision
    if signals["Golden Cross"] and signals["Fib Resistance"] == "Above 0.618 Level":
        decision["Mid Term"] = "Good for mid-term investment"
    elif signals["Death Cross"] or signals["Fib Resistance"] == "Below 0.236 Level (Support)":
        decision["Mid Term"] = "Not good for mid-term investment"
    else:
        decision["Mid Term"] = "Neutral mid-term outlook"

    # Long-term Decision
    if signals["Ichimoku Signal"] == "Bullish (Buy)" and signals["SMA_200"] > signals["SMA_50"]:
        decision["Long Term"] = "Good for long-term investment"
    elif signals["Ichimoku Signal"] == "Bearish (Sell)" or signals["SMA_200"] < signals["SMA_50"]:
        decision["Long Term"] = "Not good for long-term investment"
    else:
        decision["Long Term"] = "Neutral long-term outlook"

    signals.update(decision)

    return signals


def analyze_recommendations(df):
    """
    Analyze stock recommendations over different periods and return a dictionary with insights.
    """
    # Extract the latest row and previous row for comparison
    today = datetime.now(pytz.timezone('America/New_York'))
    latest = df.iloc[0]
    mid = df.iloc[1]
    long_term = df.iloc[3]

    previous = df.iloc[1]

    # Calculate additional insights

    # 1. **Consistency of Recommendations** (Compare current with previous periods)
    strong_buy_consistency = latest["strongBuy"] == previous["strongBuy"]
    buy_consistency = latest["buy"] == previous["buy"]
    hold_consistency = latest["hold"] == previous["hold"]
    sell_consistency = latest["sell"] == previous["sell"]
    strong_sell_consistency = latest["strongSell"] == previous["strongSell"]

    # 2. **Momentum Analysis** (Change in buy/sell recommendation over time)
    buy_momentum = latest["buy"] - previous["buy"]
    strongBuy_momentum = latest["strongBuy"] - previous["strongBuy"]
    sell_momentum = latest["sell"] - previous["sell"]
    strongSell_momentum = latest["strongSell"] - previous["strongSell"]

    # 3. **Buy/Sell Difference** (Assess balance between buy and sell)
    buy_sell_balance = latest["buy"] + latest["strongBuy"] - latest["sell"] - latest["strongSell"]

    # 4. **Overall Sentiment** (Market sentiment based on buy/sell ratios)
    overall_sentiment = "Bullish" if buy_sell_balance > 0 else "Neutral" if buy_sell_balance == 0 else "Bearish"

    # 5. **Hold Recommendations Consistency** (Are hold recommendations stable?)
    hold_stability = "Stable" if hold_consistency else "Fluctuating"

    trend_analysis = "Increasing" if latest["strongBuy"] + latest["buy"] > previous["strongBuy"] + previous[
        "buy"] else "Decreasing" if latest["strongBuy"] + latest["buy"] < previous["strongBuy"] + previous[
        "buy"] else "Stable"
    short_term = "Good" if int(buy_sell_balance) > 0 and trend_analysis == "Increasing" else "Neutral"

    # Mid-term: Based on the 1m and 2m periods
    mid_term = "Good" if mid["strongBuy"] + mid["buy"] > mid["sell"] + mid["strongSell"] and long_term["strongBuy"] + \
                         long_term["buy"] > long_term["sell"] + long_term["strongSell"] else "Neutral"

    # Long-term: Based on the 3m period and consistent trend
    long_term_investment = "Good" if long_term["strongBuy"] + long_term["buy"] > long_term["sell"] + long_term[
        "strongSell"] else "Neutral"

    # Decision-making Dictionary
    decision_data = {
        "datetime": str(today.strftime('%Y-%m-%dT%H:%M')),
        "buy_strength": int(latest["strongBuy"] + latest["buy"]),
        "sell_strength": int(latest["sell"] + latest["strongSell"]),
        "buy_to_sell_ratio": np.inf if latest["sell"] + latest["strongSell"] == 0 else (latest["strongBuy"] + latest[
            "buy"]) / (latest["sell"] + latest["strongSell"]),
        "trend": trend_analysis,
        "buy_momentum": int(buy_momentum),
        "strongBuy_momentum": int(strongBuy_momentum),
        "sell_momentum": int(sell_momentum),
        "strongSell_momentum": int(strongSell_momentum),
        "buy_sell_balance": int(buy_sell_balance),
        "overall_sentiment": overall_sentiment,
        "hold_stability": hold_stability,
        "strong_buy_consistency": bool(strong_buy_consistency),
        "buy_consistency": bool(buy_consistency),
        "hold_consistency": bool(hold_consistency),
        "sell_consistency": bool(sell_consistency),
        "strong_sell_consistency": bool(strong_sell_consistency),
        "Short Term": short_term,
        "Mid Term": mid_term,
        "Long Term": long_term_investment
    }

    return decision_data


def analyze_option_chains(df):
    today = datetime.now(pytz.timezone('America/New_York'))

    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    df['days_to_expiry'] = (df['expiration_date'] - pd.Timestamp.today()).dt.days

    # Categorize Options by Expiry
    df['category'] = df['days_to_expiry'].apply(
        lambda x: "Short-Term" if x <= 7 else "Long-Term" if x > 30 else "Medium-Term")

    # Compute Key Investment Indicators
    investment_summary = {
        "datetime": str(today.strftime('%Y-%m-%dT%H:%M')),
        "total_options": len(df),
        "average_implied_volatility": round(float(df['impliedVolatility'].mean()), 2),
        "total_volume": round(float(df['volume'].sum()), 2),
        "most_liquid_option": df.loc[df['volume'].idxmax()][
            ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
             'change', 'percentChange', 'volume', 'openInterest',
             'impliedVolatility', 'inTheMoney', 'contractSize',
             'expiration_date', 'optionType', 'days_to_expiry', 'category']].to_dict(),
        "highest_iv_option": df.loc[df['impliedVolatility'].idxmax()][
            ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
             'change', 'percentChange', 'volume', 'openInterest',
             'impliedVolatility', 'inTheMoney', 'contractSize',
             'expiration_date', 'optionType', 'days_to_expiry', 'category']].to_dict(),
        "best_short_term_option": df[df['category'] == "Short-Term"].nlargest(1,
                                                                              ['volume', 'impliedVolatility'])[
            ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
             'change', 'percentChange', 'volume', 'openInterest',
             'impliedVolatility', 'inTheMoney', 'contractSize',
             'expiration_date', 'optionType', 'days_to_expiry', 'category']].to_dict(
            orient='records'),
        "best_long_term_option": df[df['category'] == "Long-Term"].nlargest(1, ['volume', 'impliedVolatility'])[
            ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
             'change', 'percentChange', 'volume', 'openInterest',
             'impliedVolatility', 'inTheMoney', 'contractSize',
             'expiration_date', 'optionType', 'days_to_expiry', 'category']].to_dict(
            orient='records')
    }

    return investment_summary


def analyze_dividends(df):
    # Ensure the "Date" column is timezone-aware in 'America/New_York', if not already
    df['Date'] = pd.to_datetime(df['Date'])

    # If the "Date" column is timezone-naive, localize it to 'America/New_York'
    if df['Date'].dt.tz is None:
        df['Date'] = df['Date'].dt.tz_localize('America/New_York')
    else:  # If the "Date" column is already timezone-aware, convert to 'America/New_York'
        df['Date'] = df['Date'].dt.tz_convert('America/New_York')

    # Ensure 'today' is timezone-aware in 'America/New_York'

    today = datetime.now(pytz.timezone('America/New_York'))
    # Convert 'today' to timezone-aware

    # Filter short-term, mid-term, and long-term dividends based on today
    short_term_dividends = df[df['Date'] > today - pd.DateOffset(months=6)]
    mid_term_dividends = df[
        (df['Date'] <= today - pd.DateOffset(months=6)) & (df['Date'] > today - pd.DateOffset(years=1))]
    long_term_dividends = df[df['Date'] <= today - pd.DateOffset(years=1)]

    # Calculate the sum of dividends in each period
    short_term_sum = short_term_dividends['Dividends'].sum()
    mid_term_sum = mid_term_dividends['Dividends'].sum()
    long_term_sum = long_term_dividends['Dividends'].sum()

    # Investment suggestion based on dividend history
    short_term_investment = 'Good' if short_term_sum > 0.5 else 'Bad'
    mid_term_investment = 'Good' if mid_term_sum > 1 else 'Bad'
    long_term_investment = 'Good' if long_term_sum > 5 else 'Bad'

    # Construct the final dictionary
    investment_decision = {
        "datetime": str(today.strftime('%Y-%m-%dT%H:%M')),
        "latest_dividend": float(df.iloc[-1]['Dividends']),
        "short_term_dividends": float(short_term_sum),
        "mid_term_dividends": float(mid_term_sum),
        "long_term_dividends": float(long_term_sum),
        "short_term_investment": short_term_investment,
        "mid_term_investment": mid_term_investment,
        "long_term_investment": long_term_investment
    }

    return investment_decision


def analyze_earnings(df):
    today = datetime.now(pytz.timezone('America/New_York'))
    # Ensure the 'Earnings Date' is a datetime column and convert to timezone-aware
    df['Earnings Date'] = pd.to_datetime(df['Earnings Date'])
    if df['Earnings Date'].dt.tz is None:
        df['Earnings Date'] = df['Earnings Date'].dt.tz_localize('America/New_York')
    else:  # If the "Date" column is already timezone-aware, convert to 'America/New_York'
        df['Earnings Date'] = df['Earnings Date'].dt.tz_convert('America/New_York')

    # Filter rows with available EPS data
    df_valid = df.dropna(subset=['EPS Estimate', 'Reported EPS'])

    # Calculate Surprise Percentage
    df_valid['Surprise%'] = ((df_valid['Reported EPS'] - df_valid['EPS Estimate']) / df_valid['EPS Estimate']) * 100

    # Short-Term Decision: Check if recent earnings (within the last 3 months) have surprised positively
    recent_earnings = df_valid[
        df_valid['Earnings Date'] > (datetime.now(pytz.timezone('America/New_York')) - pd.DateOffset(months=3))]
    short_term_positive_surprise = recent_earnings[recent_earnings['Surprise%'] > 0]

    # Long-Term Decision: Consistent positive surprises over the last few quarters
    long_term_positive_surprise = df_valid[df_valid['Surprise%'] > 0]

    # Calculate the average growth in EPS over the last 4 quarters
    df_valid['EPS Growth'] = df_valid['Reported EPS'].pct_change() * 100
    avg_eps_growth = df_valid['EPS Growth'].mean()

    # Additional Insights:
    # 1. Earnings Seasonality (Quarterly Trends)
    df_valid['Quarter'] = df_valid['Earnings Date'].dt.quarter
    seasonality = df_valid.groupby('Quarter')['Surprise%'].mean()

    # 2. Positive Surprise Magnitude
    large_surprise = df_valid[df_valid['Surprise%'] > 10]
    small_surprise = df_valid[df_valid['Surprise%'] < 5]

    # 3. Negative Surprises
    negative_surprise = df_valid[df_valid['Surprise%'] < 0]

    # 4. EPS Consistency (Year-over-Year Growth)
    df_valid['Year'] = df_valid['Earnings Date'].dt.year
    yearly_growth = df_valid.groupby('Year')['Reported EPS'].mean().pct_change() * 100

    # 5. Linear Regression on EPS Trend
    df_valid['DateOrdinal'] = df_valid['Earnings Date'].apply(lambda x: x.toordinal())
    model = LinearRegression()
    model.fit(df_valid[['DateOrdinal']], df_valid['Reported EPS'])
    predicted_eps = model.predict(df_valid[['DateOrdinal']])

    # Constructing the final dictionary with investment insights
    investment_insights = {
        "datetime": str(today.strftime('%Y-%m-%dT%H:%M')),
        'latest_surprise': float(df_valid.iloc[-1]['Surprise%']),  # Latest surprise
        'positive_surprises_short_term': len(short_term_positive_surprise),
        'positive_surprises_long_term': len(long_term_positive_surprise),
        'average_eps_growth': round(float(avg_eps_growth)),
        'next_earnings_date': str(df['Earnings Date'].iloc[0].strftime('%Y-%m-%dT%H:%M')),  # Most recent earnings date
        'short_term_investment': 'Good' if len(short_term_positive_surprise) >= 2 else 'Bad',
        'long_term_investment': 'Good' if avg_eps_growth > 5 else 'Bad',
        'seasonality': seasonality.to_dict(),
        'large_surprise': len(large_surprise),
        'small_surprise': len(small_surprise),
        'negative_surprise': len(negative_surprise),
        'yearly_eps_growth': yearly_growth.to_dict(),
        'predicted_eps_growth': round(float(predicted_eps[-1]), 3)
    }

    return investment_insights
