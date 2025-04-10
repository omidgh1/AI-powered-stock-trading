import numpy as np
import pandas as pd
import datetime
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
import nltk
from rake_nltk import Rake

nltk.download('punkt')
nltk.download('sentiwordnet')


def keyword_extractor(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    return ",".join(keywords)


def sentiment_check(value):
    if -0.1 < value < 0.1:
        return 'neutral'
    elif 0.1 <= value <= 0.5:
        return 'positive'
    elif 0.5 < value:
        return 'storngly positive'
    elif -0.5 <= value <= -0.1:
        return 'negative'
    elif value < 0.5:
        return 'strongly negative'


def history_df_simplifier(df):
    """
    Simplifies a DataFrame containing historical data by rounding values and
    formatting dates to a consistent 'YYYY-MM-DD' format.
    """
    data_dict = df.to_dict(orient='index')
    simplified_dict = {}
    for date, data in data_dict.items():
        simple_date = date.date().strftime('%Y-%m-%d')
        rounded_data = {key: round(value, 3) for key, value in data.items()}
        simplified_dict[simple_date] = rounded_data
    return simplified_dict


def recommendation_simplifier(df):
    """
    Simplifies a DataFrame containing recommendation data by formatting times
    and including only relevant columns.
    """
    data_dict = df[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].to_dict(orient='index')
    simplified_dict = {}
    current_time = datetime.datetime.utcnow()
    for i, data in data_dict.items():
        one_minute_ago = current_time - datetime.timedelta(minutes=i)
        formatted_time = one_minute_ago.strftime('%Y-%m-%d %H:%M')
        simplified_dict[formatted_time] = data
    return simplified_dict


def opt_chain_simplifier(df):
    """
    Simplifies an options chain DataFrame by grouping data by the last trade date
    and calculating the mean for specific columns.
    """
    df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"]).dt.strftime('%Y-%m-%d %H')
    df_grouped = df.groupby('lastTradeDate')[["strike", "lastPrice", "volume", "openInterest"]].mean().round(2)
    return df_grouped.to_dict(orient="index")


def institutional_holders_simplifier(df):
    """
    Simplifies a DataFrame containing institutional holder data by setting the holder as the index
    and rounding numerical columns to 3 decimal places.
    """
    df.set_index('Holder', inplace=True)
    return df[['pctHeld', 'Shares', 'Value', 'pctChange']].round(3).to_dict(orient='index')


def overview_simplifier(df):
    """
    Simplifies an overview DataFrame by rounding the values and formatting the column headers
    as year-based strings.
    """
    df = df.round(3)
    df.columns = pd.to_datetime(df.columns).strftime('%Y')
    return df.to_dict(orient="index")


def scale_dataframe(df, factor, decimal_places):
    df_scaled = df.copy()

    # Apply scaling while keeping NaNs unchanged
    df_scaled = df_scaled.applymap(lambda x: round(x * factor, decimal_places) if not np.isnan(x) else np.nan)

    return df_scaled