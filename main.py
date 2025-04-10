from fundamental_extractor import NewsFetcher, TwitterExtractor
from technical_extractor import StockInfo
from technical_analysis import (history_technical_indicators, analyze_recommendations, analyze_option_chains,
                                analyze_dividends, analyze_earnings, financial_overview_info,
                                institutional_holders_info, major_holders_info)
from fundamental_analysis import process_news_articles, process_tweets
from AI import OpenAIClient

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

ticker = "FGR.PA"
news_fetcher = NewsFetcher(ticker=ticker, days_before=7)
yahoo_news = news_fetcher.get_yahoo_news()
#general_news = news_fetcher.get_general_news()
#vantage_news = news_fetcher.get_vantage_news()
yahoo_processed_news = process_news_articles(yahoo_news, 'yahoo')
#general_processed_news = process_news_articles(general_news, 'general')
#vantage_processed_news = process_news_articles(vantage_news['feed'], 'vantage')

#twitter_username = "elonmusk"
#twitter_class = TwitterExtractor()
#tweets = twitter_class.extract_tweets(username=twitter_username)
#processed_tweets = process_tweets(ticker, tweets)


stock_info = StockInfo(ticker=ticker)

basic_info = stock_info.get_basic_info()

historical_df = stock_info.get_stock_history()
recommendation_df = stock_info.get_recommendations()
opt_chain_calls_puts_df = stock_info.get_option_chains()
dividends_df = stock_info.get_dividends()
earning_df = stock_info.get_earnings_dates()
overview_df = stock_info.get_financial_data()
major_holders_df = stock_info.get_major_holders()
institutional_holders_df = stock_info.get_institutional_holders()

technical_indicators = history_technical_indicators(historical_df)
recommend_analysis = analyze_recommendations(recommendation_df)
opt_chain_analysis = analyze_option_chains(opt_chain_calls_puts_df)
dividends_analysis = analyze_dividends(dividends_df)
earning_analysis = analyze_earnings(earning_df)

stock_overview = financial_overview_info(overview_df)
stock_major_holders = major_holders_info(major_holders_df)
stock_institutional_holder = institutional_holders_info(institutional_holders_df)


final_result = {
    "Basic_Info": basic_info,
    "Technical_Info": {
        "Overview": stock_overview,
        "Major_Holders": stock_major_holders,
        "Institutional_Holder": stock_institutional_holder,
        #"Indicators": technical_indicators,
        "Recommendation": recommend_analysis,
        "Options_Chain": opt_chain_analysis,
        "Dividends": dividends_analysis,
        "Earning": earning_analysis,
    },
    "Fundamental_Info": {
        "Yahoo": yahoo_processed_news,
        #"NewsAPI": general_processed_news,
        #"Vantage": vantage_processed_news
    }

}

ai_class = OpenAIClient()
respond = ai_class.generate_response(system_prompt_file='test', user_prompt=str(final_result))
print('omid')