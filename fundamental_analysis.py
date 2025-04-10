from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob

from AI import OpenAIClient
from utils import keyword_extractor, sentiment_check
import datetime
import logging

# Initialize sentiment analysis models
vader = SentimentIntensityAnalyzer()
finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone")
distilroberta = pipeline("text-classification",
                         model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


def analyze_sentiment(text):
    """
    Perform sentiment analysis using multiple models: VADER, TextBlob, FinBERT, and DistilRoBERTa.
    """
    sentiment_scores = {}

    # VADER Sentiment Score
    sentiment_scores['vader'] = vader.polarity_scores(text)['compound']

    # TextBlob Sentiment Score
    sentiment_scores['textblob'] = TextBlob(text).sentiment.polarity

    # FinBERT Sentiment Score
    finbert_result = finbert(text)[0]
    sentiment_scores['finbert'] = finbert_result['score'] if finbert_result['label'] == 'Positive' else \
        -finbert_result['score'] if finbert_result['label'] == 'Negative' else 0

    # DistilRoBERTa Sentiment Score
    bert_result = distilroberta(text)[0]
    sentiment_scores['distilroberta'] = bert_result['score'] if bert_result['label'] == 'positive' else \
        -bert_result['score'] if bert_result['label'] == 'negative' else 0

    return sentiment_scores


def process_news_articles(news_list, source):
    """
    Processes a list of news articles, extracting relevant information and performing sentiment analysis.
    """
    processed_news = {}

    if not news_list:
        logging.warning("The news list is empty. No data to process.")
        return processed_news

    logging.info(f"Processing {len(news_list)} news articles from source: {source}")

    for news in news_list:
        try:
            # Extract news details based on source
            if source == 'yahoo':
                content = news.get('content', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                datetime_str = content.get('pubDate', '')
                datetime_format = '%Y-%m-%dT%H:%M:%SZ'
            elif source == 'general':
                title = news.get('title', '')
                summary = news.get('description', '')
                datetime_str = news.get('publishedAt', '')
                datetime_format = '%Y-%m-%dT%H:%M:%SZ'
            elif source == 'vantage':
                title = news.get('title', '')
                summary = news.get('summary', '')
                datetime_str = news.get('time_published', '')
                datetime_format = '%Y%m%dT%H%M%S'
            else:
                logging.warning(f"Unknown news source: {source}. Skipping article.")
                return None

            # Convert datetime string to a standard format
            try:
                published_datetime = datetime.datetime.strptime(datetime_str, datetime_format)
                datetime_var = str(published_datetime)
            except (ValueError, TypeError) as e:
                logging.error(f"Error parsing date '{datetime_str}' for source '{source}': {e}")
                datetime_var = "Unknown"

            # Extract keywords
            keywords = keyword_extractor(f"{title} {summary}")

            # Perform sentiment analysis
            title_sentiment = analyze_sentiment(title)
            summary_sentiment = analyze_sentiment(summary)
            combined_sentiment = analyze_sentiment(f"{title} {summary}")

            # Compute average sentiment scores
            avg_title_sentiment = round(sum(title_sentiment.values()) / len(title_sentiment), 2)
            avg_summary_sentiment = round(sum(summary_sentiment.values()) / len(summary_sentiment), 2)
            avg_combined_sentiment = round(sum(combined_sentiment.values()) / len(combined_sentiment), 2)

            # Store processed news data
            processed_news[datetime_var] = {
                'title': sentiment_check(avg_title_sentiment),
                'summary': sentiment_check(avg_summary_sentiment),
                'combined': sentiment_check(avg_combined_sentiment),
                'keywords': keywords
            }

            logging.info(f"Processed news article: {title[:50]}...")

        except Exception as e:
            logging.exception(f"Unexpected error while processing an article: {e}")

    logging.info("Finished processing news articles.")
    return processed_news


def process_tweets(ticker, tweets_list):
    ai_class = OpenAIClient()
    user_input = {'ticker': ticker, 'tweets_list': tweets_list}
    processed_tweets = ai_class.generate_response(system_prompt_file='Tweets_sentiments',
                                                  user_prompt=str(user_input))
    for key, value in processed_tweets.items():
        value['keywords'] = ', '.join(value['keywords'])

    return processed_tweets

