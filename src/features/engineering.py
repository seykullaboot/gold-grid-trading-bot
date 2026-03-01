"""
src/features/engineering.py — Feature Engineering สำหรับ Gold Trading

สร้าง Technical Indicators ครบชุด:
- Trend Indicators: EMA, SMA, MACD, Bollinger Bands
- Momentum Indicators: RSI, Stochastic, Williams %R
- Volatility Indicators: ATR, Historical Volatility
- Volume Indicators: OBV, Volume SMA
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """คลาสสำหรับสร้าง Technical Indicator Features"""

    def __init__(self):
        """กำหนดค่าเริ่มต้น FeatureEngineer"""
        self.feature_columns = []

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม Trend Indicators

        ครอบคลุม: EMA (10,20,50,200), SMA (20,50,200),
                  MACD, Bollinger Bands, ADX

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV

        Returns:
            DataFrame ที่มี trend indicators เพิ่มเติม
        """
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- Exponential Moving Averages ---
        for period in [10, 20, 50, 200]:
            col = f"ema_{period}"
            df[col] = ta.trend.ema_indicator(close, window=period)
            # Distance from EMA (%)
            df[f"dist_ema_{period}"] = (close - df[col]) / df[col]

        # --- Simple Moving Averages ---
        for period in [20, 50, 200]:
            col = f"sma_{period}"
            df[col] = ta.trend.sma_indicator(close, window=period)
            df[f"dist_sma_{period}"] = (close - df[col]) / df[col]

        # --- Golden/Death Cross Signals ---
        df["golden_cross"] = (
            (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        ).astype(int)
        df["death_cross"] = (
            (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        ).astype(int)

        # --- MACD ---
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["macd_crossover"] = (
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)

        # --- Bollinger Bands ---
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_percent"] = bb.bollinger_pband()  # %B indicator

        # --- Average Directional Index (ADX) ---
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()  # +DI
        df["adx_neg"] = adx.adx_neg()  # -DI

        # --- Parabolic SAR ---
        df["psar"] = ta.trend.psar_up(high, low, close)

        logger.info("เพิ่ม Trend Indicators สำเร็จ")
        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม Momentum Indicators

        ครอบคลุม: RSI, Stochastic Oscillator, Williams %R, CCI, ROC

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV

        Returns:
            DataFrame ที่มี momentum indicators เพิ่มเติม
        """
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- RSI (Relative Strength Index) ---
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = ta.momentum.RSIIndicator(close, window=period).rsi()

        # RSI divergence signal
        df["rsi_14"] = df.get("rsi_14", ta.momentum.RSIIndicator(close, window=14).rsi())
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

        # --- Stochastic Oscillator ---
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["stoch_cross"] = (
            (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        ).astype(int)

        # --- Williams %R ---
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high, low, close, lbp=14
        ).williams_r()

        # --- CCI (Commodity Channel Index) ---
        df["cci"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()

        # --- Rate of Change (ROC) ---
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = ta.momentum.ROCIndicator(close, window=period).roc()

        # --- Momentum ---
        df["momentum_10"] = close - close.shift(10)

        logger.info("เพิ่ม Momentum Indicators สำเร็จ")
        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม Volatility Indicators

        ครอบคลุม: ATR, Historical Volatility, Keltner Channel

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV

        Returns:
            DataFrame ที่มี volatility indicators เพิ่มเติม
        """
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- Average True Range (ATR) ---
        for period in [7, 14, 21]:
            atr = ta.volatility.AverageTrueRange(high, low, close, window=period)
            df[f"atr_{period}"] = atr.average_true_range()
            df[f"atr_{period}_pct"] = df[f"atr_{period}"] / close  # ATR as % of price

        # --- Historical Volatility ---
        returns = close.pct_change()
        for period in [10, 20, 60]:
            df[f"hist_vol_{period}d"] = returns.rolling(window=period).std() * np.sqrt(252)

        # --- Volatility Ratio ---
        df["vol_ratio"] = df["hist_vol_10d"] / df["hist_vol_60d"]

        # --- True Range ---
        df["true_range"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

        # --- Keltner Channel ---
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_middle"] = kc.keltner_channel_mband()

        # Squeeze Indicator (BB inside KC = ความผันผวนต่ำ)
        if all(c in df.columns for c in ["bb_upper", "bb_lower"]):
            df["squeeze"] = (
                (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
            ).astype(int)

        logger.info("เพิ่ม Volatility Indicators สำเร็จ")
        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม Volume Indicators

        ครอบคลุม: OBV, Volume SMA, Volume Ratio, MFI

        Args:
            df: DataFrame ที่มีคอลัมน์ OHLCV

        Returns:
            DataFrame ที่มี volume indicators เพิ่มเติม
        """
        if "Volume" not in df.columns:
            logger.warning("ไม่มีคอลัมน์ Volume — ข้าม volume indicators")
            return df

        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # --- On-Balance Volume (OBV) ---
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df["obv_ema"] = df["obv"].ewm(span=20).mean()

        # --- Volume SMA ---
        for period in [10, 20, 50]:
            df[f"volume_sma_{period}"] = volume.rolling(window=period).mean()

        # Volume ratio (ปัจจุบัน vs SMA20)
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # --- Money Flow Index (MFI) ---
        df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # --- Chaikin Money Flow ---
        df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()

        # --- Volume-weighted price change ---
        df["vwap_proxy"] = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()

        logger.info("เพิ่ม Volume Indicators สำเร็จ")
        return df

    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม Price Pattern Features

        Args:
            df: DataFrame OHLCV

        Returns:
            DataFrame ที่มี pattern features เพิ่มเติม
        """
        df = df.copy()
        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]

        # Candle body และ shadow
        df["body"] = (close - open_).abs()
        df["upper_shadow"] = high - df[["Close", "Open"]].max(axis=1)
        df["lower_shadow"] = df[["Close", "Open"]].min(axis=1) - low
        df["body_pct"] = df["body"] / (high - low + 1e-10)

        # ทิศทาง candle
        df["is_bullish"] = (close > open_).astype(int)

        # Higher highs / Lower lows (5 วัน)
        df["higher_high"] = (high > high.rolling(window=5).max().shift(1)).astype(int)
        df["lower_low"] = (low < low.rolling(window=5).min().shift(1)).astype(int)

        # Price distance from 52-week high/low
        df["dist_52w_high"] = (close - high.rolling(252).max()) / high.rolling(252).max()
        df["dist_52w_low"] = (close - low.rolling(252).min()) / low.rolling(252).min()

        logger.info("เพิ่ม Price Pattern Features สำเร็จ")
        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้าง features ทั้งหมดในครั้งเดียว

        Args:
            df: DataFrame ข้อมูล OHLCV ดิบ

        Returns:
            DataFrame ที่มี features ครบชุด
        """
        logger.info("กำลังสร้าง features ทั้งหมด...")

        # เพิ่ม returns พื้นฐาน
        df = df.copy()
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # เพิ่ม indicators แต่ละประเภท
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_price_patterns(df)

        # ลบแถวที่มี NaN (เกิดจาก rolling windows)
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)

        # บันทึก feature columns
        self.feature_columns = [
            c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]
        ]

        logger.info(
            f"สร้าง features สำเร็จ: {len(self.feature_columns)} features, "
            f"{len(df)} แถว (ลบ {removed} แถวเพราะ NaN)"
        )
        return df

    def get_feature_list(self) -> list:
        """
        ดึงรายชื่อ features ที่สร้างแล้ว

        Returns:
            รายการชื่อ features
        """
        return self.feature_columns.copy()
