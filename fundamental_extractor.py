import os
import requests
import datetime
import logging
from technical_extractor import StockInfo


class NewsFetcher:
    def __init__(self, newsapi_key: str = None, alphavantage_key: str = None, ticker: str = None, days_before: int = 1):
        newsapi_key = newsapi_key or os.getenv("newsapi_API_KEY")
        if not newsapi_key:
            logging.warning('The API for NewsApi not found.')
        alphavantage_key = alphavantage_key or os.getenv("alphavantage_API_KEY")
        if not newsapi_key:
            logging.warning('The API for alphavantage not found.')
        self.newsapi_key = newsapi_key
        self.alphavantage_key = alphavantage_key
        self.ticker = ticker
        self.days_before = days_before

    def get_general_news(self):
        """Fetches general news articles for a given stock ticker using NewsAPI."""
        date_from = (datetime.datetime.utcnow() - datetime.timedelta(days=self.days_before)).isoformat() + "Z"
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": self.ticker,
            "apiKey": self.newsapi_key,
            "language": "en",
            "sortBy": "publishedAt",
            "from": date_from
        }

        response = requests.get(url, params=params)
        return response.json().get("articles", [])

    def get_vantage_news(self, limit: int = 20):
        """Fetches news sentiment data for a given stock ticker using Alpha Vantage API."""
        time_from = (datetime.datetime.utcnow() - datetime.timedelta(days=self.days_before)).strftime("%Y%m%dT%H%M")
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker.upper(),
            "apikey": self.alphavantage_key,
            "time_from": time_from,
            "limit": limit,
            "sort": "LATEST"
        }

        response = requests.get(url, params=params)
        return response.json()

    def get_yahoo_news(self):
        stock_news = StockInfo(ticker=self.ticker)
        get_news = stock_news.get_news()
        return get_news


class TwitterExtractor:
    def __init__(self, RapidAPI_twitter_API_KEY: str = None):
        self.api_key = RapidAPI_twitter_API_KEY or os.getenv("RapidAPI_twitter_API_KEY")
        self.USER_ID_URL = "https://twitter-v23.p.rapidapi.com/UserByScreenName/"
        self.TWEETS_URL = "https://twitter241.p.rapidapi.com/user-tweets"
        self.HEADERS = {
            "x-rapidapi-key": self.api_key,
        }

    def get_user_id(self, username: str):
        """Fetches user ID from Twitter API given a username."""
        response = requests.get(self.USER_ID_URL,
                                headers={**self.HEADERS, "x-rapidapi-host": "twitter-v23.p.rapidapi.com"},
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

    def get_tweets(self, user_id: str, count: int = 10):
        """Fetches the latest tweets for a given user ID."""
        response = requests.get(self.TWEETS_URL,
                                headers={**self.HEADERS, "x-rapidapi-host": "twitter241.p.rapidapi.com"},
                                params={"user": user_id, "count": str(count)})

        if response.status_code != 200:
            logging.error(f"Failed to fetch tweets. Status code: {response.status_code}")
            return None

        try:
            tweets_data = response.json()
            tweet_entries = tweets_data['result']['timeline']['instructions'][-1]['entries']
            normalized_tweets = []
            for tweet in tweet_entries:
                try:
                    tweet_result = tweet['content']["itemContent"]["tweet_results"]["result"]["legacy"]
                    full_text = tweet_result.get('full_text', {})
                    created_at = tweet_result.get('created_at', {})
                    retweet_count = tweet_result.get('retweet_count', {})
                    hashtags = tweet_result.get('legacy', {}).get('entities', {}).get('hashtags', [])
                    correct_date = str(datetime.datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").replace(
                        tzinfo=None))
                    normalized_tweets.append({'text': full_text, 'retweet_count': retweet_count,
                                              'hashtags': ','.join(hashtags), 'date': correct_date})
                except KeyError:
                    logging.warning("Skipping malformed tweet entry.")
            return normalized_tweets
        except (KeyError, IndexError):
            logging.error("Unexpected response structure while fetching tweets.")
            return None

    def extract_tweets(self, username: str):
        """Main function to extract tweets from a username."""
        user_id = self.get_user_id(username)
        if not user_id:
            return "Failed to retrieve user ID from Twitter API."

        tweets = self.get_tweets(user_id)
        return tweets if tweets else "Failed to retrieve tweets from Twitter API."
