#from fundamental_extractor import NewsFetcher, TwitterExtractor
from technical_extractor import StockInfo
from technical_analysis import (history_technical_indicators, analyze_recommendations, option_chains_analysis,
                                analyze_dividends, analyze_earnings, financial_overview_info,
                                institutional_holders_info, major_holders_info)
#from fundamental_analysis import process_news_articles, process_tweets
#from AI import OpenAIClient
#from dotenv import load_dotenv

#load_dotenv(dotenv_path=".env")

ticker = "MSFT"
#news_fetcher = NewsFetcher(ticker='MSFT', days_before=7)
#yahoo_news = news_fetcher.get_yahoo_news()
#general_news = news_fetcher.get_general_news()
#vantage_news = news_fetcher.get_vantage_news()
#yahoo_processed_news = process_news_articles(yahoo_news, 'yahoo')
#general_processed_news = process_news_articles(general_news, 'general')
#vantage_processed_news = process_news_articles(vantage_news['feed'], 'vantage')

#twitter_class = TwitterExtractor()
#tweets = twitter_class.extract_tweets(username='elonmusk')
#processed_tweets = process_tweets('TSLA',tweets)


stock_info = StockInfo(ticker=ticker)

#dynamic
#historical_data = stock_info.get_stock_history()
#recommendation_data = stock_info.get_recommendations()
#stock_opt_chain_calls_puts = stock_info.get_option_chains()
#dividends_df = stock_info.get_dividends()
#earning_df = stock_info.get_earnings_dates()

#technical_indicators = history_technical_indicators(historical_data)
#recommend_analyze = analyze_recommendations(recommendation_data)
#opt_chain_analysis = option_chains_analysis(stock_opt_chain_calls_puts)
#dividends_analyze_result = analyze_dividends(dividends_df)
#earning_analyze = analyze_earnings(earning_df)

#static
stock_overview_df = stock_info.get_financial_data()
stock_major_holders_df = stock_info.get_major_holders()
stock_institutional_holders_df = stock_info.get_institutional_holders()

stock_overview = financial_overview_info(stock_overview_df)
stock_major_holders = major_holders_info(stock_major_holders_df)
stock_institutional_holder = institutional_holders_info(stock_institutional_holders_df)
print('omid')
