import yfinance as yf
import pandas as pd

class StockInfo:
    """
    A class to fetch detailed information and data for a given stock ticker, including
    fundamental information, financials, balance sheet, cash flow, options chain,
    dividend history, earnings dates, recommendations, institutional holdings, and news.
    """

    def __init__(self, ticker: str):
        """
        Initializes the StockInfo class with a stock ticker.

        Args:
        ticker (str): The stock ticker symbol.
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info

    def get_basic_info(self) -> dict:
        """
        Extracts and returns basic numerical stock information along with company name,
        sector, and industry.

        Returns:
        dict: Basic stock information.
        """
        stock_info = {key: value for key, value in self.info.items() if isinstance(value, (int, float))}
        stock_info.update({
            "Company Name": self.info.get("longName"),
            "Sector": self.info.get("sector"),
            "Industry": self.info.get("industry")
        })
        return stock_info

    def get_financial_data(self) -> pd.DataFrame:
        """
        Combines financial data, balance sheet, and cash flow into one DataFrame.

        Returns:
        pd.DataFrame: Combined financial data.
        """
        stock_finance_df = self.stock.financials
        stock_balance_df = self.stock.balance_sheet
        stock_cashflow_df = self.stock.cashflow
        return pd.concat([stock_finance_df, stock_balance_df, stock_cashflow_df], axis=0, join='outer')

    def get_stock_history(self, period: str = "1y") -> pd.DataFrame:
        """
        Fetches the stock's historical data for a given period.

        Args:
        period (str): The time period for historical data (default is "1y").

        Returns:
        pd.DataFrame: Historical stock data.
        """
        return self.stock.history(period=period)

    def get_recommendations(self) -> pd.DataFrame:
        """
        Fetches the stock recommendations.

        Returns:
        pd.DataFrame: Stock recommendations.
        """
        return self.stock.recommendations

    def get_major_holders(self) -> pd.DataFrame:
        """
        Fetches the major stock holders.

        Returns:
        pd.DataFrame: Major stock holders.
        """
        return self.stock.major_holders

    def get_institutional_holders(self) -> pd.DataFrame:
        """
        Fetches institutional stock holders.

        Returns:
        pd.DataFrame: Institutional stock holders.
        """
        return self.stock.institutional_holders

    def get_option_chains(self) -> tuple:
        """
        Fetches the stock's call and put options chains.

        Returns:
        tuple: A tuple containing two DataFrames for call options and put options.
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

        return pd.concat(calls_list, ignore_index=True), pd.concat(puts_list, ignore_index=True)

    def get_dividends(self) -> pd.DataFrame:
        """
        Fetches the stock's dividend history.

        Returns:
        pd.DataFrame: Dividend history.
        """
        return self.stock.dividends

    def get_earnings_dates(self) -> pd.DataFrame:
        """
        Fetches the stock's earnings dates.

        Returns:
        pd.DataFrame: Earnings dates.
        """
        return self.stock.earnings_dates

    def get_news(self) -> list:
        """
        Fetches the latest news for the stock.

        Returns:
        list: Stock news.
        """
        return self.stock.news

    def get_all_data(self) -> tuple:
        """
        Fetches all available data related to the stock.

        Returns:
        tuple: A tuple containing all data including news, earnings dates, dividends,
               options chain, institutional holders, etc.
        """
        stock_info = self.get_basic_info()
        stock_overview_df = self.get_financial_data()
        stock_history_df = self.get_stock_history()
        stock_recommendations_df = self.get_recommendations()
        stock_major_holders_df = self.get_major_holders()
        stock_institutional_holders_df = self.get_institutional_holders()
        stock_opt_chain_calls_df, stock_opt_chain_puts_df = self.get_option_chains()
        stock_dividends_df = self.get_dividends()
        stock_earning_dates_df = self.get_earnings_dates()
        stock_news = self.get_news()

        return (stock_news, stock_earning_dates_df, stock_dividends_df, stock_opt_chain_puts_df,
                stock_opt_chain_calls_df, stock_institutional_holders_df, stock_major_holders_df,
                stock_recommendations_df, stock_history_df, stock_overview_df, stock_info)

