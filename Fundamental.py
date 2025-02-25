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


class TwitterExtractor:
    USER_ID_URL = "https://twitter-v23.p.rapidapi.com/UserByScreenName/"
    TWEETS_URL = "https://twitter241.p.rapidapi.com/user-tweets"
    API_KEY = "fd40997d5amshc122325f00c0f6ap12bd3fjsn6bd70be9ec2c"

    HEADERS = {
        "x-rapidapi-key": API_KEY,
    }

    @classmethod
    def get_user_id(cls, username: str):
        """Fetches user ID from Twitter API given a username."""
        response = requests.get(cls.USER_ID_URL,
                                headers={**cls.HEADERS, "x-rapidapi-host": "twitter-v23.p.rapidapi.com"},
                                params={"username": username})

        if response.status_code != 200:
            logging.error(f"Failed to fetch user ID. Status code: {response.status_code}")
            return None

        try:
            user_data = response.json()
            return user_data['data']['user']['result']['rest_id']
        except KeyError:
            logging.error("Unexpected response structure while fetching user ID.")
            return None

    @classmethod
    def get_tweets(cls, user_id: str, count: int = 10):
        """Fetches the latest tweets for a given user ID."""
        response = requests.get(cls.TWEETS_URL, headers={**cls.HEADERS, "x-rapidapi-host": "twitter241.p.rapidapi.com"},
                                params={"user": user_id, "count": str(count)})

        if response.status_code != 200:
            logging.error(f"Failed to fetch tweets. Status code: {response.status_code}")
            return None

        try:
            tweets_data = response.json()
            tweet_entries = tweets_data['result']['timeline']['instructions'][-1]['entries']
            return cls.normalize_tweets(tweet_entries)
        except (KeyError, IndexError):
            logging.error("Unexpected response structure while fetching tweets.")
            return None

    @staticmethod
    def normalize_tweets(tweets):
        """Extracts and normalizes tweet text and timestamp."""
        normalized_tweets = []
        for tweet in tweets:
            try:
                tweet_result = tweet['content']["itemContent"]["tweet_results"]["result"]["legacy"]
                full_text = tweet_result["full_text"].strip()
                created_at = tweet_result["created_at"]
                correct_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").replace(tzinfo=None)
                normalized_tweets.append({'text': full_text, 'date': correct_date})
            except KeyError:
                logging.warning("Skipping malformed tweet entry.")

        return normalized_tweets

    @classmethod
    def extract_tweets(cls, username: str):
        """Main function to extract tweets from a username."""
        user_id = cls.get_user_id(username)
        if not user_id:
            return "Failed to retrieve user ID from Twitter API."

        tweets = cls.get_tweets(user_id)
        return tweets if tweets else "Failed to retrieve tweets from Twitter API."


#tweets = TwitterExtractor.extract_tweets("ElonMuskAOC")


"""

sentiments = []
for article in articles[:10]:  # Analyze last 10 news articles
        title = article["title"]
        score = analyzer.polarity_scores(title)["compound"]
        sentiments.append(score)

sum(sentiments) / len(sentiments) if sentiments else 0
"""
