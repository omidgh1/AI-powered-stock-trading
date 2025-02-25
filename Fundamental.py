import numpy as np
import pandas as pd
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

API_KEY = "3835157cc8c1442d9396cf8e1b2baf39"
analyzer = SentimentIntensityAnalyzer()


def general_news(API_KEY, ticker):
    NEWS_URL = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "publishedAt"
    }
    response = requests.get(NEWS_URL, params=params)
    articles = response.json().get("articles", [])
    return articles





"""

sentiments = []
for article in articles[:10]:  # Analyze last 10 news articles
        title = article["title"]
        score = analyzer.polarity_scores(title)["compound"]
        sentiments.append(score)

sum(sentiments) / len(sentiments) if sentiments else 0
"""
