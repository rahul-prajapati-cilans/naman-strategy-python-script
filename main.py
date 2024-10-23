import logging
import numpy as np
import os
import pandas as pd
import smtplib
import threading
import time
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()
lock = threading.Lock()

# =================================================================================
# Logging Configuration
# =================================================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the default log level for your application

# Create handlers
file_handler = logging.FileHandler("app.log")  # Log to a file
console_handler = logging.StreamHandler()  # Log to the console (terminal)

# Set log levels for the handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Apply the format to both handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure specific loggers for external libraries
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("apscheduler").setLevel(logging.CRITICAL)

# =================================================================================


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
    # Not available in Yahoo Finance
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
    # Nifty Indexes
    "^NSEI",
    "^NSEBANK",
]

TESTING_EMAILS = [
    "kaushal.cilans@gmail.com",
    "rahul.cilans@gmail.com",
]
CLIENT_EMAILS = [
    "namanshah1912@gmail.com",
    "vandanshahca@gmail.com",
    "kabraravi15@gmail.com",
]
TESTING = os.getenv("TESTING", False)
RECEIVER_EMAILS = TESTING_EMAILS if TESTING else CLIENT_EMAILS + TESTING_EMAILS
COMMENT_LINE = "--------------------------------------------"
COMMENT_EQUAL = "============================================"

INDEX_LIST = ["^NSEI", "^NSEBANK"]


# Global variables
is_initial_breakout_running = False
initial_breakout_first_run = True
ultimate_bullish_stocks = []
ultimate_bearish_stocks = []
bullish_initial_breakout = []
bearish_initial_breakout = []
confirmation_bullish_stocks = []
confirmation_bearish_stocks = []
mprs_dataframes = {}


def get_previous_day_ohlc(ticker):
    """
    A function which collects previous day's OHLC for particular stock
    """

    # If the ticker is not a Nifty index, add ".NS" to the ticker
    if ticker not in INDEX_LIST:
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
        ticker,
        start=previous_day,
        end=current_day,
        interval="1d",
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
    if stock not in INDEX_LIST:
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
        logging.info(f"The Stock {stock} is not following the condition of 2.5%.")
    return None


def ultimate_condition(daily_stocks, first_candle_stock_data):
    """
    The ultimate condition for the stock to be bullish or bearish
    """
    global ultimate_bullish_stocks, ultimate_bearish_stocks
    start_ultimate = time.time()
    logging.info("Checking ultimate condition for the stocks")
    if ultimate_bullish_stocks and ultimate_bearish_stocks:
        logging.info("Ultimate condition already met. Skipping...")
        logging.info(COMMENT_EQUAL)
        logging.info(f"ULTIMATE BULLISH STOCKS : {ultimate_bullish_stocks}")
        logging.info(COMMENT_LINE)
        logging.info(f"ULTIMATE BEARISH STOCKS : {ultimate_bearish_stocks}")
        logging.info(COMMENT_EQUAL)
        return

    for stock in daily_stocks:
        data = first_candle_stock_data[stock]
        if data["Open"] == data["High"]:
            logging.info(f"The stock {stock} is bearish")
            ultimate_bearish_stocks.append(stock)
        elif data["Open"] == data["Low"]:
            logging.info(f"The stock {stock} is bullish")
            ultimate_bullish_stocks.append(stock)
    logging.info(COMMENT_EQUAL)
    logging.info(f"ULTIMATE BULLISH STOCKS : {ultimate_bullish_stocks}")
    logging.info(COMMENT_LINE)
    logging.info(f"ULTIMATE BEARISH STOCKS : {ultimate_bearish_stocks}")
    logging.info(COMMENT_EQUAL)
    end_ultimate = time.time()
    logging.info(
        f"Time taken for ultimate condition: {end_ultimate - start_ultimate} seconds."
    )
    return None


def download_data_for_initial_breakout(ticker):
    """
    Download the data for the stock for initial breakout condition
    """
    logging.info(f"Initial Breakout - Downloading data for {ticker}")

    if ticker not in INDEX_LIST:
        ticker = f"{ticker}.NS"

    # Get the data of the stock from the start date to the current date
    current_df = yf.download(ticker, period="1d", interval="15m")

    # Condition that is the value of "Close" is higher or lower than "Open"
    current_df["Color"] = current_df.apply(
        lambda x: 1 if x["Close"] > x["Open"] else 0, axis=1
    )

    if ticker == "ZYDUSLIFE.NS":
        logging.info(current_df)

    logging.info(f"Initial Breakout - Data downloaded for {ticker}")
    return current_df


def initial_breakout_bull(stock, ohlc_data):
    """
    Any candle during the day's 15-minute candle should break the previous day's high, including its body with a green candle.
    The candle should close above the previous day's high with a green candle.
    """
    global bullish_initial_breakout
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
        logging.info(f"No initial breakout for {stock} in bullish side")
    return None


def initial_breakout_bear(stock, ohlc_data):
    """
    Any Candle during the day's 15-minute candle should break the previous day's low, including its body with a red color candle.
    The candle should close below the previous day's low with a red color candle.
    """
    global bearish_initial_breakout
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
        logging.info(f"No initial breakout for {stock} in bearish side")
    return None


def get_daily_first_candle_data(stock_list, first_candle_stock_data):
    """
    Function to fetch the first candle data for a list of stocks
    """
    start_first_candle = time.time()
    logging.info("Fetching first candle data for the day for the stocks")
    if first_candle_stock_data:
        return

    for ticker in stock_list:
        ticker_ = ticker
        if ticker not in INDEX_LIST:
            ticker = f"{ticker}.NS"
        data = yf.download(ticker, period="1d", interval="15m")
        if not data.empty:
            first_candle = data.iloc[0]
            first_candle_stock_data[ticker_] = {
                "Datetime": first_candle.name.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": first_candle["Open"],
                "High": first_candle["High"],
                "Low": first_candle["Low"],
            }
        else:
            logging.info(f"No data available for {ticker}")
    end_first_candle = time.time()
    logging.info(
        f"Time taken for fetching first candle data: {end_first_candle - start_first_candle} seconds."
    )
    return None


def download_data_for_confirmation(ticker):
    """
    Download the data for the stock for confirmation condition
    """

    if ticker not in INDEX_LIST:
        ticker = f"{ticker}.NS"

    confirmation_df = yf.download(ticker, period="1d", interval="1m")
    return confirmation_df


def confirmation_bull(stock, first_candle_high):
    """
    The second 15-minute candle at 9:30 AM or any other candle later on should break the high of the first 9:15 AM candle of the current day.
    """
    global confirmation_bullish_stocks
    if stock in confirmation_bullish_stocks or not first_candle_high:
        return
    df = download_data_for_confirmation(stock)
    if df["High"].values[-1] > first_candle_high:
        confirmation_bullish_stocks.append(stock)
    return None


def confirmation_bear(stock, first_candle_low):
    """
    The second 15-minute candle at 9:30 AM or any other candle later on should break the low of the first 9:15 AM candle of the current day.
    """
    global confirmation_bearish_stocks
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
        logging.info("Stopping scheduler: Not a weekday")
        return True

    # Define the time range
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # If current time is outside the range
    if not (start_time <= now <= end_time):
        logging.info("Stopping scheduler: Outside the time range of 9:15 AM to 3:30 PM")
        return True
    return False


def shutdown_scheduler(scheduler):
    """
    Function to shut down the scheduler
    """
    logging.info("Shutting down scheduler...")
    scheduler.shutdown()


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


def download_data_for_mprs(ticker):
    """
    Download the data for the stock for MPRS condition
    """
    global mprs_dataframes
    ticker_ = ticker

    if ticker not in INDEX_LIST:
        ticker = f"{ticker}.NS"

    current_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    mprs_df = yf.download(ticker, start=current_date, interval="15m")
    mprs_dataframes[ticker_] = mprs_df
    return mprs_df


def execute_bull(common_bull_stocks):
    final_bullish = []

    for ticker in common_bull_stocks:
        mprs_df = download_data_for_mprs(ticker)
        logging.info(f"Processing Final Bullish for {ticker}")

        # Calculate SMA and check its direction
        sma = calculate_sma(dataframe=mprs_df)
        sma_direction = sma["SMA_DIRECTION"].tail(1).values[0]
        if sma_direction != 1:
            continue  # If SMA condition is not met, skip to next stock
        logging.info(f"{ticker} - SMA condition met with direction: {sma_direction}")

        # Calculate PSAR and check its direction
        psar = calculate_psar(dataframe=mprs_df)
        psar_direction = psar["PSAR_Direction"].tail(1).values[0]
        if psar_direction != 1:
            continue  # If PSAR condition is not met, skip to next stock
        logging.info(f"{ticker} - PSAR condition met with direction: {psar_direction}")

        # Calculate RSI and check if it's greater than 60
        rsi = calculate_rsi(dataframe=mprs_df)
        rsi_value = rsi["RSI"].tail(1).values[0]
        logging.info(f"{ticker} - RSI value: {rsi_value}")
        if rsi_value <= 60:
            continue  # If RSI condition is not met, skip to next stock
        logging.info(f"{ticker} - RSI condition met with value: {rsi_value}")

        # Calculate MACD and check its signal
        macd = calculate_macd(dataframe=mprs_df)
        macd_signal = macd["MACD_Signal"].tail(1).values[0]
        logging.info(f"{ticker} - MACD signal: {macd_signal}")
        if macd_signal <= 0:
            continue  # If MACD condition is not met, skip to next stock
        logging.info(f"{ticker} - MACD condition met with signal: {macd_signal}")

        # If all conditions are met, the stock is bullish
        final_bullish.append(ticker)
        logging.info(f"The stock {ticker} is bullish")
    return final_bullish


def execute_bear(common_bear_stocks):
    final_bearish = []

    for ticker in common_bear_stocks:
        mprs_df = download_data_for_mprs(ticker)
        logging.info(f"Processing Final Bearish for {ticker}")

        # Calculate SMA and check its direction
        sma = calculate_sma(dataframe=mprs_df)
        sma_direction = sma["SMA_DIRECTION"].tail(1).values[0]
        if sma_direction != -1:
            continue  # If SMA condition is not met, skip to next stock
        logging.info(f"{ticker} - SMA condition met with direction: {sma_direction}")

        # Calculate PSAR and check its direction
        psar = calculate_psar(dataframe=mprs_df)
        psar_direction = psar["PSAR_Direction"].tail(1).values[0]
        if psar_direction != -1:
            continue  # If PSAR condition is not met, skip to next stock
        logging.info(f"{ticker} - PSAR condition met with direction: {psar_direction}")

        # Calculate RSI and check if it's less than 40
        rsi = calculate_rsi(dataframe=mprs_df)
        rsi_value = rsi["RSI"].tail(1).values[0]
        logging.info(f"{ticker} - RSI value: {rsi_value}")
        if rsi_value >= 40:
            continue  # If RSI condition is not met, skip to next stock
        logging.info(f"{ticker} - RSI condition met with value: {rsi_value}")

        # Calculate MACD and check its signal
        macd = calculate_macd(dataframe=mprs_df)
        macd_signal = macd["MACD_Signal"].tail(1).values[0]
        logging.info(f"{ticker} - MACD signal: {macd_signal}")
        if macd_signal >= 0:
            continue  # If MACD condition is not met, skip to next stock
        logging.info(f"{ticker} - MACD condition met with signal: {macd_signal}")

        # If all conditions are met, the stock is bearish
        final_bearish.append(ticker)
        logging.info(f"The stock {ticker} is bearish")
    return final_bearish


def handle_initial_breakout(previous_day_ohlc):
    """
    Process the initial breakout condition for both bullish and bearish stocks.
    """
    start_initial = time.time()
    global is_initial_breakout_running, initial_breakout_first_run, ultimate_bullish_stocks, ultimate_bearish_stocks, bullish_initial_breakout, bearish_initial_breakout
    if initial_breakout_first_run:
        logging.info("First run, delaying for 2 minutes...")
        time.sleep(120)
        initial_breakout_first_run = False
    with lock:
        is_initial_breakout_running = True
    try:
        logging.info(
            f"Checking initial breakout for the stocks - {datetime.now().time()}"
        )
        if datetime.now().time() > datetime.strptime("09:30", "%H:%M").time():
            with lock:
                for stock in ultimate_bullish_stocks:
                    if previous_day_ohlc.get(stock, None) is None:
                        continue
                    initial_breakout_bull(stock, previous_day_ohlc[stock])
                for stock in ultimate_bearish_stocks:
                    if previous_day_ohlc.get(stock, None) is None:
                        continue
                    initial_breakout_bear(stock, previous_day_ohlc[stock])
                logging.info(COMMENT_EQUAL)
                logging.info(f"BEARISH INITIAL BREAKOUT : {bearish_initial_breakout}")
                logging.info(COMMENT_LINE)
                logging.info(f"BULLISH INITIAL BREAKOUT : {bullish_initial_breakout}")
                logging.info(COMMENT_EQUAL)
        return None
    finally:
        with lock:
            is_initial_breakout_running = False
        end_initial = time.time()
        logging.info(
            f"Time taken for initial breakout: {end_initial - start_initial} seconds."
        )


def handle_confirmation(first_candle_stock_data):
    """
    Process the confirmation condition for both bullish and bearish stocks.
    """
    global is_initial_breakout_running, ultimate_bullish_stocks, ultimate_bearish_stocks, bullish_initial_breakout, bearish_initial_breakout, confirmation_bearish_stocks, confirmation_bullish_stocks
    start_confirmation = time.time()
    if is_initial_breakout_running:
        logging.info("Skipping handle confirmation, initial breakout is running.")
        return

    with lock:
        logging.info(f"Checking confirmation for the stocks - {datetime.now().time()}")
        if datetime.now().time() > datetime.strptime("09:33", "%H:%M").time():
            for stock in ultimate_bullish_stocks:
                confirmation_bull(stock, first_candle_stock_data[stock]["High"])
            for stock in ultimate_bearish_stocks:
                confirmation_bear(stock, first_candle_stock_data[stock]["Low"])
            logging.info(COMMENT_EQUAL)
            logging.info(f"CONFIRMATION BULLISH STOCKS : {confirmation_bullish_stocks}")
            logging.info(COMMENT_LINE)
            logging.info(f"CONFIRMATION BEARISH STOCKS : {confirmation_bearish_stocks}")
            logging.info(COMMENT_EQUAL)

            common_bear_stocks = list(
                set(bearish_initial_breakout) & set(confirmation_bearish_stocks)
            )
            common_bull_stocks = list(
                set(bullish_initial_breakout) & set(confirmation_bullish_stocks)
            )
            logging.info(COMMENT_EQUAL)
            logging.info(f"COMMON BEAR STOCKS : {common_bear_stocks}")
            logging.info(COMMENT_LINE)
            logging.info(f"COMMON BULL STOCKS : {common_bull_stocks}")
            logging.info(COMMENT_EQUAL)
            final_bullish, final_bearish = handle_mprs(
                common_bull_stocks, common_bear_stocks
            )
            prepare_and_send_alert(final_bullish, final_bearish)
    end_confirmation = time.time()
    logging.info(
        f"Time taken for confirmation: {end_confirmation - start_confirmation} seconds."
    )
    return None


def prepare_and_send_alert(final_bullish, final_bearish):
    """
    Prepare the stock alert message and send it via email.
    Only sends email if at least one of the final lists contains stocks.
    """
    global ultimate_bullish_stocks, ultimate_bearish_stocks, bullish_initial_breakout, bearish_initial_breakout, confirmation_bullish_stocks, confirmation_bearish_stocks
    if not final_bullish and not final_bearish:
        logging.info("No stocks to alert. Email not sent.")
        return

    message_text = "Stock Alert:\n\n"

    if final_bullish:
        message_text += "Bullish Stocks:\n"
        for stock in final_bullish:
            message_text += format_message(stock, "bullish")
        message_text += "\n"
    else:
        message_text += "No Bullish Stocks at the moment.\n\n"

    if final_bearish:
        message_text += "Bearish Stocks:\n"
        for stock in final_bearish:
            message_text += format_message(stock, "bearish")
        message_text += "\n"
    else:
        message_text += "No Bearish Stocks at the moment.\n\n"

    send_email(message_text)

    ultimate_bullish_stocks = list(set(ultimate_bullish_stocks) - set(final_bullish))
    ultimate_bearish_stocks = list(set(ultimate_bearish_stocks) - set(final_bearish))
    confirmation_bullish_stocks = list(
        set(confirmation_bullish_stocks) - set(final_bullish)
    )
    confirmation_bearish_stocks = list(
        set(confirmation_bearish_stocks) - set(final_bearish)
    )
    bullish_initial_breakout = list(set(bullish_initial_breakout) - set(final_bullish))
    bearish_initial_breakout = list(set(bearish_initial_breakout) - set(final_bearish))
    return None


def handle_mprs(common_bull_stocks, common_bear_stocks):
    """
    Process the MPRS condition for both bullish and bearish stocks.
    """
    final_bullish = execute_bull(common_bull_stocks)
    final_bearish = execute_bear(common_bear_stocks)
    logging.info(COMMENT_EQUAL)
    logging.info(f"FINAL BULLISH STOCKS : {final_bullish}")
    logging.info(COMMENT_LINE)
    logging.info(f"FINAL BEARISH STOCKS : {final_bearish}")
    logging.info(COMMENT_EQUAL)
    return final_bullish, final_bearish


def format_message(stock, condition):
    try:
        stock_df = mprs_dataframes.get(stock, None)
        if stock_df is None:
            logging.warning(f"No MPRS data available for stock: {stock}")
            message = (
                f"Stock : {stock}\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n"
            )
            return message
        stock_df = stock_df.sort_index()

        # Get the last two rows (candles) from the DataFrame
        last_candle = stock_df.iloc[-1]
        prev_candle = stock_df.iloc[-2]
        trigger_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S")

        # Set SL and target based on condition
        if condition == "bullish":
            sl = prev_candle["Low"]
            target = last_candle["Close"] + (last_candle["Close"] - prev_candle["Low"])
        elif condition == "bearish":
            sl = prev_candle["High"]
            target = last_candle["Close"] - (prev_candle["High"] - last_candle["Close"])
        else:
            sl = target = "Unknown"

        message = (
            f"Stock Name: {stock}\n"
            f"Time of Trigger: {trigger_time}\n"
            f"Entry Price: Rs {last_candle['Close']:.2f}\n"
            f"SL: Rs {sl:.2f} (Low of previous candle)\n"
            f"Target: Rs {target:.2f} (same as SL)\n"
            f"+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n"
        )
        return message
    except Exception as e:
        logging.error(f"Error formatting message for {stock}: {str(e)}")
        return f"Stock : {stock}\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n"


def send_email(message_text):
    """
    Parameters:
    - message_text: Text message to be sent via email.

    The function sets up email parameters, creates a MIME multipart message,
    and sends the email using the smtplib library to the specified recipients.
    """
    logging.info("Sending Email...")

    sender_email = os.getenv("EMAIL_HOST_USER")  # Email address of the sender
    password = os.getenv("EMAIL_HOST_PASSWORD")  # Password of the sender's email
    subject = "Naman Alert"  # Subject of the email
    body = message_text  # Body of the email, contains the message text

    # Join all recipient emails into a single string separated by commas
    recipient_emails = ", ".join(RECEIVER_EMAILS)

    # Establish a connection with the SMTP server
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_emails
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))
            server.sendmail(sender_email, RECEIVER_EMAILS, message.as_string())
            logging.info(f"Email Sent Successfully to {recipient_emails}!")

    except Exception as e:
        logging.error(f"Error Sending Email: {str(e)}")

    return None


def main():
    """
    Main function to run the script
    """
    if stop_scheduler():
        logging.info(
            "The script is not running on a weekday or outside the time range."
        )
        return
    previous_day_ohlc = {}
    current_day_stock_df = {}
    bullish_daily_stocks = []
    bearish_daily_stocks = []
    first_candle_stock_data = {}

    start_ = time.time()
    for stock in stocks:
        try:
            previous_day_ohlc[stock] = get_previous_day_ohlc(stock)
            current_day_stock_df[stock] = get_current_day_stock_data(stock)
            check_opening_price_logic(
                stock,
                previous_day_ohlc[stock],
                current_day_stock_df[stock],
                bullish_daily_stocks,
                bearish_daily_stocks,
            )
        except Exception as e:
            logging.error(
                f"Error during processing (OHLC, DF & 2.5 percent) for {stock}: {str(e)}"
            )
            continue
    end_ = time.time()
    logging.info(
        f"Time taken to process previous day OHLC, current day DF and checking 2.5 percent condition: {end_ - start_} seconds."
    )
    logging.info(COMMENT_EQUAL)
    logging.info(f"BULLISH DAILY STOCKS: {bullish_daily_stocks}")
    logging.info(COMMENT_LINE)
    logging.info(f"BEARISH DAILY STOCKS: {bearish_daily_stocks}")
    logging.info(COMMENT_EQUAL)

    # Initialize APScheduler
    scheduler = BlockingScheduler()

    # Schedule tasks for breakout and confirmation checks
    scheduler.add_job(
        get_daily_first_candle_data,
        trigger="cron",
        hour=9,
        minute=31,
        args=[bearish_daily_stocks + bullish_daily_stocks, first_candle_stock_data],
    )

    scheduler.add_job(
        ultimate_condition,
        trigger="cron",
        hour=9,
        minute=32,
        args=[
            bullish_daily_stocks + bearish_daily_stocks,
            first_candle_stock_data,
        ],
    )

    scheduler.add_job(
        handle_initial_breakout,
        trigger="cron",
        minute="1-59/15",  # This runs at 1, 16, 31, and 46 minutes past the hour
        args=[previous_day_ohlc],
    )

    scheduler.add_job(
        handle_confirmation,
        trigger="cron",
        second=0,  # This will run at the 00th second of every minute
        args=[first_candle_stock_data],
    )

    scheduler.add_job(
        shutdown_scheduler, CronTrigger(hour=15, minute=31), args=[scheduler]
    )

    logging.info("Scheduler started")
    try:
        if stop_scheduler():
            return
        scheduler.start()  # This will block and run the scheduler
    except (KeyboardInterrupt, SystemExit) as e:
        logging.info("Scheduler stopped")
        scheduler.shutdown()
        raise e
    finally:
        scheduler.shutdown()  # Ensure that the scheduler shuts down properly


if __name__ == "__main__":
    main()
