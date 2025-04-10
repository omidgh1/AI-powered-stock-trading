from typing import Dict, Any

import pandas as pd
import yfinance as yf
from pandas import Series, DataFrame
from utils import (history_df_simplifier, recommendation_simplifier, opt_chain_simplifier,
                   institutional_holders_simplifier, overview_simplifier)


class StockInfo:
    """
    A class to fetch detailed information and data for a given stock ticker, including
    fundamental information, financials, balance sheet, cash flow, options chain,
    dividend history, earnings dates, recommendations, institutional holdings, and news.
    """

    def __init__(self, ticker: str):
        """
        Initializes the StockInfo class with a stock ticker.
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info

    def get_basic_info(self) -> dict:
        basic_info = {'address': self.info.get('address1'),
                      'city': self.info.get('city'),
                      'state': self.info.get('state'),
                      'zip': self.info.get('zip'),
                      'country': self.info.get('country'),
                      'phone': self.info.get('phone'),
                      'website': self.info.get('website'),
                      'industry': self.info.get('industry'),
                      'sector': self.info.get('sector'),
                      'longName': self.info.get('longName'),
                      'exchange': self.info.get('exchange'),
                      'exchangeTimezoneName': self.info.get('exchangeTimezoneName'),
                      'averageAnalystRating': self.info.get('averageAnalystRating'),
                      'displayName': self.info.get('displayName'),
                      'symbol': self.info.get('symbol'),
                      'ticker': self.ticker}
        return basic_info

    def get_basic_technical(self) -> dict:
        """
        Extracts and returns basic numerical stock information along with company name,
        sector, and industry.
        """
        stock_info = {key: value for key, value in self.info.items() if isinstance(value, (int, float))}
        return stock_info

    def get_financial_data(self) -> pd.DataFrame:
        """
        Combines financial data, balance sheet, and cash flow into one DataFrame.
        """
        stock_finance_df = self.stock.financials
        stock_balance_df = self.stock.balance_sheet
        stock_cashflow_df = self.stock.cashflow
        return pd.concat([stock_finance_df, stock_balance_df, stock_cashflow_df], axis=0, join='outer')

    def get_stock_history(self, period: str = "1y") -> pd.DataFrame:
        """
        Fetches the stock's historical data for a given period.
        """
        return self.stock.history(period=period).reset_index()

    def get_recommendations(self) -> pd.DataFrame:
        """
        Fetches the stock recommendations.
        """
        return self.stock.recommendations

    def get_major_holders(self) -> DataFrame:
        """
        Fetches the major stock holders.
        """

        return self.stock.major_holders

    def get_institutional_holders(self) -> DataFrame:
        """
        Fetches institutional stockholders.
        """
        return self.stock.institutional_holders

    def get_option_chains(self) -> DataFrame:
        """
        Fetches the stock's call and put options chains.
        """
        calls_list, puts_list = [], []
        for exp_date in self.stock.options:
            stock_opt_chain_df = self.stock.option_chain(exp_date)
            stock_opt_chain_call_df = stock_opt_chain_df.calls
            stock_opt_chain_puts_df = stock_opt_chain_df.puts
            stock_opt_chain_call_df['expiration_date'] = exp_date
            stock_opt_chain_puts_df['expiration_date'] = exp_date
            calls_list.append(stock_opt_chain_call_df)
            puts_list.append(stock_opt_chain_puts_df)

        calls_df = pd.concat(calls_list, ignore_index=True)
        calls_df['optionType'] = "Call"
        puts_df = pd.concat(puts_list, ignore_index=True)
        puts_df['optionType'] = "Put"

        final_puts_calls_df = pd.concat([calls_df, puts_df], ignore_index=True)

        return final_puts_calls_df

    def get_dividends(self) -> pd.DataFrame:
        """
        Fetches the stock's dividend history.
        """

        return self.stock.dividends.reset_index()

    def get_earnings_dates(self) -> pd.DataFrame:
        """
        Fetches the stock's earnings dates.
        """
        return self.stock.earnings_dates.reset_index()

    def get_news(self) -> list:
        """
        Fetches the latest news for the stock. 24 hours ago
        """
        news = self.stock.news
        return news

    def get_static_technical_data(self):
        stock_overview_df = self.get_financial_data()
        stock_major_holders_df = self.get_major_holders()
        stock_institutional_holders_df = self.get_institutional_holders()

        stock_overview_json = overview_simplifier(stock_overview_df)
        stock_major_holders_json = stock_major_holders_df.round(3).to_dict(orient="index")
        stock_institutional_holders_json = institutional_holders_simplifier(stock_institutional_holders_df)

        return {"overview": stock_overview_json,
                "major_holders": stock_major_holders_json,
                "institutional_holders": stock_institutional_holders_json}
