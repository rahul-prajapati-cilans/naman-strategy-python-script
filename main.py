import numpy as np
import pandas as pd
import schedule
import time
import yfinance as yf
from datetime import datetime, timedelta


stocks = [
    "AARTIIND",
    "ABB",
    "ABBOTINDIA",
    "ABCAPITAL",
    "ABFRL",
    "ACC",
    "ADANIENT",
    "ADANIPORTS",
    "ALKEM",
    "AMBUJACEM",
    "APOLLOHOSP",
    "APOLLOTYRE",
    "ASHOKLEY",
    "ASIANPAINT",
    "ASTRAL",
    "ATUL",
    "AUBANK",
    "AUROPHARMA",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BALKRISIND",
    "BALRAMCHIN",
    "BANDHANBNK",
    "BANKBARODA",
    "BATAINDIA",
    "BEL",
    "BERGEPAINT",
    "BHARATFORG",
    "BHARTIARTL",
    "BHEL",
    "BIOCON",
    "BOSCHLTD",
    "BPCL",
    "BRITANNIA",
    "BSOFT",
    "CANBK",
    "CANFINHOME",
    "CHAMBLFERT",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "COROMANDEL",
    "CROMPTON",
    "CUB",
    "CUMMINSIND",
    "DABUR",
    "DALBHARAT",
    "DEEPAKNTR",
    "DIVISLAB",
    "DIXON",
    "DLF",
    "DRREDDY",
    "EICHERMOT",
    "ESCORTS",
    "EXIDEIND",
    "FEDERALBNK",
    "GAIL",
    "GLENMARK",
    "GMRINFRA",
    "GNFC",
    "GODREJCP",
    "GODREJPROP",
    "GRANULES",
    "GRASIM",
    "GUJGASLTD",
    "HAL",
    "HAVELLS",
    "HCLTECH",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDCOPPER",
    "HINDPETRO",
    "HINDUNILVR",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "IDEA",
    # "IDFC",
    "IDFCFIRSTB",
    "IEX",
    "IGL",
    "INDHOTEL",
    "INDIAMART",
    "INDIGO",
    "INDUSINDBK",
    "INDUSTOWER",
    "INFY",
    "IOC",
    "IPCALAB",
    "IRCTC",
    "ITC",
    "JINDALSTEL",
    "JKCEMENT",
    "JSWSTEEL",
    "JUBLFOOD",
    "KOTAKBANK",
    "LALPATHLAB",
    "LAURUSLABS",
    "LICHSGFIN",
    "LT",
    "LTF",
    "LTIM",
    "LTTS",
    "LUPIN",
    "M&M",
    "M&MFIN",
    "MANAPPURAM",
    "MARICO",
    "MARUTI",
    "MCX",
    "METROPOLIS",
    "MFSL",
    "MGL",
    "MOTHERSON",
    "MPHASIS",
    "MRF",
    "MUTHOOTFIN",
    "NATIONALUM",
    "NAUKRI",
    "NAVINFLUOR",
    "NESTLEIND",
    "NMDC",
    "NTPC",
    "OBEROIRLTY",
    "OFSS",
    "ONGC",
    "PAGEIND",
    "PEL",
    "PERSISTENT",
    "PETRONET",
    "PFC",
    "PIDILITIND",
    "PIIND",
    "PNB",
    "POLYCAB",
    "POWERGRID",
    "PVRINOX",
    "RAMCOCEM",
    "RBLBANK",
    "RECLTD",
    "RELIANCE",
    "SAIL",
    "SBICARD",
    "SBILIFE",
    "SBIN",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SRF",
    "SUNPHARMA",
    "SUNTV",
    "SYNGENE",
    "TATACHEM",
    "TATACOMM",
    "TATACONSUM",
    "TATAMOTORS",
    "TATAPOWER",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TORNTPHARM",
    "TRENT",
    "TVSMOTOR",
    "UBL",
    "ULTRACEMCO",
    "UNITDSPR",
    "UPL",
    "VEDL",
    "VOLTAS",
    "WIPRO",
    "ZYDUSLIFE",
    "^NSEI",
    "^NSEBANK",
]


def get_previous_day_ohlc(ticker):
    """
    A function which collects previous day's OHLC for particular stock
    """

    # If the ticker is not a Nifty index, add ".NS" to the ticker
    if ticker not in ["^NSEI", "^NSEBANK"]:
        ticker = ticker + ".NS"

    current_day = pd.Timestamp.now().date()
    previous_day = current_day - pd.Timedelta(days=1)

    # Adjust if the previous day falls on a weekend (Saturday or Sunday)
    if previous_day.weekday() == 5:  # If Saturday, move to Friday
        previous_day -= pd.Timedelta(days=1)
    elif previous_day.weekday() == 6:  # If Sunday, move to Friday
        previous_day -= pd.Timedelta(days=2)

    # Download the previous day's data
    previous_day_data = yf.download(
        ticker, start=previous_day, end=previous_day + pd.Timedelta(days=1)
    )

    # Extract the high and low prices
    previous_day_open = previous_day_data["Open"].values[0]
    previous_day_high = previous_day_data["High"].values[0]
    previous_day_low = previous_day_data["Low"].values[0]
    previous_day_close = previous_day_data["Close"].values[0]

    ohlc_data = {
        "open": previous_day_open,
        "high": previous_day_high,
        "low": previous_day_low,
        "close": previous_day_close,
    }

    return ohlc_data


def get_current_day_stock_data(stock):
    """
    A function which collects current day's stock data
    """
    if stock not in ["^NSEI", "^NSEBANK"]:
        stock = stock + ".NS"
    df = yf.download(stock, period="1d")
    return df


def check_opening_price_logic(
    stock, ohlc_data, df, bullish_daily_stocks, bearish_daily_stocks
):
    """
    Check if the opening price of the current day is within 2.5% of the previous day's closing price
    """
    # Extract the current day's open price
    current_day_open = df["Open"].values[0]

    # Calculate the percentage change in opening price
    percentage_change = (
        (current_day_open - float(ohlc_data["close"])) / float(ohlc_data["close"])
    ) * 100

    # Check if the percentage change is within the range
    if percentage_change > -2.5 and percentage_change <= 0:
        bullish_daily_stocks.append(stock)
    elif percentage_change > 0 and percentage_change < 2.5:
        bearish_daily_stocks.append(stock)
    else:
        print(f"The Stock {stock} is not following the condition")
    return None


def download_data_for_initial_breakout(ticker):
    """
    Download the data for the stock for initial breakout condition
    """
    print(f"Initial Breakout - Downloading data for {ticker}")

    if ticker not in ["^NSEI", "^NSEBANK"]:
        ticker = f"{ticker}.NS"

    # Get the data of the stock from the start date to the current date
    current_df = yf.download(ticker, period="1d", interval="15m")

    # Condition that is the value of "Close" is higher or lower than "Open"
    current_df["Color"] = current_df.apply(
        lambda x: 1 if x["Close"] > x["Open"] else 0, axis=1
    )

    if ticker == "ZYDUSLIFE.NS":
        print(current_df)

    print(f"Initial Breakout - Data downloaded for {ticker}")
    return current_df


def initial_breakout_bull(stock, bullish_initial_breakout, ohlc_data):
    """
    Any candle during the day's 15-minute candle should break the previous day's high, including its body with a green candle.
    The candle should close above the previous day's high with a green candle.
    """
    if stock in bullish_initial_breakout:
        return

    # Get the data of the stock
    current_df = download_data_for_initial_breakout(stock)

    # Get the latest closing price
    latest_close = current_df["Close"].values[-1]
    candle_color = current_df["Color"].values[-1]

    # Check if the latest closing price is higher than the previous day's high
    if latest_close > ohlc_data["high"] and candle_color == 1:
        bullish_initial_breakout.append(stock)
    else:
        print(f"No initial breakout for {stock} in bullish side")
    return None


def initial_breakout_bear(stock, bearish_initial_breakout, ohlc_data):
    """
    Any Candle during the day's 15-minute candle should break the previous day's low, including its body with a red color candle.
    The candle should close below the previous day's low with a red color candle.
    """
    if stock in bearish_initial_breakout:
        return

    # Get the data of the stock
    current_df = download_data_for_initial_breakout(stock)

    # Get the latest closing price
    latest_close = current_df["Close"].values[-1]
    candle_color = current_df["Color"].values[-1]

    # Check if the latest closing price is higher than the previous day's high
    if latest_close < ohlc_data["low"] and candle_color == 0:
        bearish_initial_breakout.append(stock)
    else:
        print(f"No initial breakout for {stock} in bearish side")
    return None


def get_daily_first_candle_data(stock_list, first_candle_stock_data):
    """
    Function to fetch the first candle data for a list of stocks
    """
    if first_candle_stock_data:
        return

    for ticker in stock_list:
        ticker_ = ticker
        if ticker not in ["^NSEI", "^NSEBANK"]:
            ticker = f"{ticker}.NS"
        data = yf.download(ticker, period="1d", interval="15m")
        if not data.empty:
            first_candle = data.iloc[0]
            first_candle_stock_data[ticker_] = {
                "Datetime": first_candle.name.strftime("%Y-%m-%d %H:%M:%S"),
                "High": first_candle["High"],
                "Low": first_candle["Low"],
            }
        else:
            print(f"No data available for {ticker}")
    return None


def download_data_for_confirmation(ticker):
    """
    Download the data for the stock for confirmation condition
    """

    if ticker not in ["^NSEI", "^NSEBANK"]:
        ticker = f"{ticker}.NS"

    confirmation_df = yf.download(ticker, period="1d", interval="1m")
    return confirmation_df


def confirmation_bull(stock, confirmation_bullish_stocks, first_candle_high):
    """
    The second 15-minute candle at 9:30 AM or any other candle later on should break the high of the first 9:15 AM candle of the current day.
    """
    if stock in confirmation_bullish_stocks or not first_candle_high:
        return
    df = download_data_for_confirmation(stock)
    if df["High"].values[-1] > first_candle_high:
        confirmation_bullish_stocks.append(stock)
    return None


def confirmation_bear(stock, confirmation_bearish_stocks, first_candle_low):
    """
    The second 15-minute candle at 9:30 AM or any other candle later on should break the low of the first 9:15 AM candle of the current day.
    """
    if stock in confirmation_bearish_stocks or not first_candle_low:
        return
    df = download_data_for_confirmation(stock)
    if df["Low"].values[-1] < first_candle_low:
        confirmation_bearish_stocks.append(stock)
    return None


def stop_scheduler():
    """
    Stop the scheduler if it's not a weekday or the current time is not between 9:15 AM and 3:30 PM.
    """
    now = datetime.now()
    # Weekdays are Monday (0) to Friday (4)
    if now.weekday() >= 5:  # If it's Saturday or Sunday
        print("Stopping scheduler: Not a weekday")
        return True

    # Define the time range
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # If current time is outside the range
    if not (start_time <= now <= end_time):
        print("Stopping scheduler: Outside the time range of 9:15 AM to 3:30 PM")
        return True
    return False


def calculate_psar(dataframe, start=0.02, increment=0.02, maximum=0.2):
    """
    Calculate the Parabolic SAR for the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing 'High' and 'Low' columns.
    start (float): The initial acceleration factor.
    increment (float): The increment applied to the acceleration factor.
    maximum (float): The maximum value for the acceleration factor.

    Returns:
    pd.DataFrame: The DataFrame with PSAR, Direction, BuySignal, and SellSignal columns.
    """

    df = dataframe.copy()
    length = len(df)
    psar = np.zeros(length)
    af = start
    ep = df["Low"].iloc[0]
    psar[0] = df["High"].iloc[0]
    long = True

    for i in range(1, length):
        if long:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            if df["Low"].iloc[i] < psar[i]:
                long = False
                psar[i] = ep
                af = start
                ep = df["Low"].iloc[i]
            else:
                if df["High"].iloc[i] > ep:
                    ep = df["High"].iloc[i]
                    af = min(af + increment, maximum)
        else:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            if df["High"].iloc[i] > psar[i]:
                long = True
                psar[i] = ep
                af = start
                ep = df["High"].iloc[i]
            else:
                if df["Low"].iloc[i] < ep:
                    ep = df["Low"].iloc[i]
                    af = min(af + increment, maximum)

    df["PSAR"] = psar

    # Determine the direction
    df["PSAR_Direction"] = np.where(df["PSAR"] < df["Close"], 1, -1)

    # Buy/Sell signals
    df["PSAR_BuySignal"] = (df["PSAR_Direction"] == 1) & (
        df["PSAR_Direction"].shift(1) == -1
    )
    df["PSAR_SellSignal"] = (df["PSAR_Direction"] == -1) & (
        df["PSAR_Direction"].shift(1) == 1
    )

    return df


def calculate_sma(dataframe, sma=20):
    """
    Calculate the Simple Moving Average (SMA) for the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing 'Close' column.
    sma (int): The value of the SMA to calculate.

    Returns:
    pd.DataFrame: The DataFrame with SMA column.
    """

    df = dataframe.copy()
    df["SMA"] = df["Close"].rolling(sma).mean()
    df["SMA_DIRECTION"] = np.where(df["SMA"] < df["Close"], 1, -1)
    return df


def calculate_macd(dataframe, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing 'Close' column.
    short_period (int): The short period for calculating the MACD line.
    long_period (int): The long period for calculating the MACD line.
    signal_period (int): The signal period for calculating the signal line.

    Returns:
    pd.DataFrame: The DataFrame with MACD and signal columns.
    go.Figure: The Plotly figure showing the MACD.
    """

    df = dataframe.copy()

    # Calculate MACD line
    df["MACD"] = (
        df["Close"].ewm(span=short_period, adjust=False).mean()
        - df["Close"].ewm(span=long_period, adjust=False).mean()
    )

    # Calculate signal line
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD histogram
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    df["MACD Histogram_Signal"] = np.where((df["MACD"] - df["MACD_Signal"]) > 0, 1, -1)
    return df


def calculate_rsi(dataframe, period=14):
    """
    Calculate the Relative Strength Index (RSI) for the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing 'Close' column.
    period (int): The period for calculating the RSI.

    Returns:
    pd.DataFrame: The DataFrame with the RSI and RSI_Signal columns.
    go.Figure: The Plotly figure showing the RSI.
    """

    df = dataframe.copy()

    # Calculate the difference in price from the previous step
    df["Change"] = df["Close"] - df["Close"].shift(1)

    # Separate gains and losses
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, -df["Change"], 0)

    # Calculate the average gain and loss
    df["Avg_Gain"] = df["Gain"].rolling(window=period).mean()
    df["Avg_Loss"] = df["Loss"].rolling(window=period).mean()

    # Calculate the RSI
    df["RS"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    # Replace infinite values with NaN and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["RSI"], inplace=True)

    # Create a column "RSI_Signal" based on RSI thresholds
    df["RSI_Signal"] = 0
    df.loc[df["RSI"] > 60, "RSI_Signal"] = 1
    df.loc[df["RSI"] < 40, "RSI_Signal"] = -1

    return df


def main():
    """
    Main function to run the script
    """
    previous_day_ohlc = {}
    current_day_stock_df = {}
    bullish_daily_stocks = []
    bearish_daily_stocks = []
    bullish_initial_breakout = []
    bearish_initial_breakout = []
    first_candle_stock_data = {}
    confirmation_bullish_stocks = []
    confirmation_bearish_stocks = []

    for stock in stocks:
        previous_day_ohlc[stock] = get_previous_day_ohlc(stock)
        current_day_stock_df[stock] = get_current_day_stock_data(stock)
        check_opening_price_logic(
            stock,
            previous_day_ohlc[stock],
            current_day_stock_df[stock],
            bullish_daily_stocks,
            bearish_daily_stocks,
        )

    print("BULLISH DAILY STOCKS :", bullish_daily_stocks)
    print("BEARISH DAILY STOCKS :", bearish_daily_stocks)

    def initial_breakout_condition():
        if datetime.now().time() > datetime.strptime("09:30", "%H:%M").time():
            for stock in bullish_daily_stocks:
                initial_breakout_bull(
                    stock, bullish_initial_breakout, previous_day_ohlc[stock]
                )
            for stock in bearish_daily_stocks:
                initial_breakout_bear(
                    stock, bearish_initial_breakout, previous_day_ohlc[stock]
                )
        print("BEARISH INITIAL BREAKOUT :", bearish_initial_breakout)
        print("BULLISH INITIAL BREAKOUT :", bullish_initial_breakout)

    def confirmation_condition():
        if datetime.now().time() > datetime.strptime("09:31", "%H:%M").time():
            print("Confirmation Condition")
            for stock in bullish_daily_stocks:
                confirmation_bull(
                    stock,
                    confirmation_bullish_stocks,
                    first_candle_stock_data[stock]["High"],
                )
            for stock in bearish_daily_stocks:
                confirmation_bear(
                    stock,
                    confirmation_bearish_stocks,
                    first_candle_stock_data[stock]["Low"],
                )
        print("CONFIRMATION BULLISH STOCKS :", confirmation_bullish_stocks)
        print("CONFIRMATION BEARISH STOCKS :", confirmation_bearish_stocks)
        common_bear_stocks = list(
            set(bearish_initial_breakout) & set(confirmation_bearish_stocks)
        )
        common_bull_stocks = list(
            set(bullish_initial_breakout) & set(confirmation_bullish_stocks)
        )
        print("COMMON BEAR STOCKS :", common_bear_stocks)
        print("COMMON BULL STOCKS :", common_bull_stocks)

    schedule.every(3).minutes.do(initial_breakout_condition)
    schedule.every(1).minute.do(
        get_daily_first_candle_data,
        (bearish_daily_stocks + bullish_daily_stocks),
        first_candle_stock_data,
    )
    schedule.every(1).minute.do(confirmation_condition)

    print("Scheduler started")
    while True:
        # if stop_scheduler():
        #     break
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
