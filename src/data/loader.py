"""
src/data/loader.py — โมดูลสำหรับดาวน์โหลดและโหลดข้อมูลทองคำ

รองรับ:
- ดาวน์โหลดจาก Yahoo Finance (yfinance)
- โหลดจากไฟล์ CSV local
- ตรวจสอบความถูกต้องของข้อมูล
- แปลง timeframe
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """คลาสสำหรับดาวน์โหลดและโหลดข้อมูลราคาทองคำ"""

    # คอลัมน์ที่จำเป็นต้องมีในข้อมูล
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, data_path: str = "data/raw/"):
        """
        กำหนดค่าเริ่มต้น DataLoader

        Args:
            data_path: เส้นทางสำหรับบันทึกข้อมูล
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataLoader เริ่มต้นด้วย data_path: {self.data_path}")

    def download_gold_data(
        self,
        ticker: str = "GLD",
        start_date: str = "2004-01-01",
        end_date: str = "2024-12-31",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        ดาวน์โหลดข้อมูลทองคำจาก Yahoo Finance

        Args:
            ticker: สัญลักษณ์หุ้น (เช่น GLD, GC=F)
            start_date: วันเริ่มต้น (YYYY-MM-DD)
            end_date: วันสิ้นสุด (YYYY-MM-DD)
            save_csv: บันทึกเป็น CSV หรือไม่

        Returns:
            DataFrame ที่มีข้อมูล OHLCV
        """
        logger.info(f"กำลังดาวน์โหลดข้อมูล {ticker} ตั้งแต่ {start_date} ถึง {end_date}")

        try:
            # ดาวน์โหลดด้วย yfinance
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                raise ValueError(f"ไม่พบข้อมูลสำหรับ ticker: {ticker}")

            # ทำความสะอาด column names
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            # เก็บเฉพาะคอลัมน์ที่จำเป็น
            available_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
            df = df[available_cols]

            logger.info(
                f"ดาวน์โหลดสำเร็จ: {len(df)} แถว ตั้งแต่ {df.index[0].date()} ถึง {df.index[-1].date()}"
            )

            # บันทึกเป็น CSV ถ้าต้องการ
            if save_csv:
                filepath = self.data_path / f"{ticker}_{start_date}_{end_date}.csv"
                df.to_csv(filepath)
                logger.info(f"บันทึกข้อมูลที่ {filepath}")

            return df

        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดาวน์โหลดข้อมูล: {e}")
            raise

    def load_local_data(self, filepath: str) -> pd.DataFrame:
        """
        โหลดข้อมูลจากไฟล์ CSV หรือ Parquet ใน local

        Args:
            filepath: เส้นทางไปยังไฟล์ข้อมูล

        Returns:
            DataFrame ที่โหลดจากไฟล์
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {filepath}")

        logger.info(f"กำลังโหลดข้อมูลจาก {filepath}")

        # โหลดตามประเภทไฟล์
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        elif filepath.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError(f"ไม่รองรับประเภทไฟล์: {filepath.suffix}")

        logger.info(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        ตรวจสอบความถูกต้องของข้อมูล

        Args:
            df: DataFrame ที่ต้องการตรวจสอบ

        Returns:
            True ถ้าข้อมูลถูกต้อง, False ถ้าไม่ถูกต้อง
        """
        issues = []

        # ตรวจสอบว่า DataFrame ไม่ว่างเปล่า
        if df.empty:
            issues.append("DataFrame ว่างเปล่า")

        # ตรวจสอบคอลัมน์ที่จำเป็น
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            issues.append(f"ขาดคอลัมน์: {missing_cols}")

        # ตรวจสอบ index เป็น DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index ต้องเป็น DatetimeIndex")

        # ตรวจสอบค่า negative
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns and (df[col] <= 0).any():
                issues.append(f"คอลัมน์ {col} มีค่า <= 0")

        # ตรวจสอบ OHLC consistency (High >= Low, High >= Open, High >= Close)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            invalid_hl = (df["High"] < df["Low"]).sum()
            if invalid_hl > 0:
                issues.append(f"High < Low จำนวน {invalid_hl} แถว")

        # ตรวจสอบ missing values เกิน 5%
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.05]
        if not high_missing.empty:
            issues.append(f"คอลัมน์ที่มี missing > 5%: {high_missing.to_dict()}")

        # แสดงผลการตรวจสอบ
        if issues:
            for issue in issues:
                logger.warning(f"ปัญหาข้อมูล: {issue}")
            return False

        logger.info(f"ข้อมูลผ่านการตรวจสอบ: {len(df)} แถว, ช่วง {df.index[0]} ถึง {df.index[-1]}")
        return True

    def resample_data(self, df: pd.DataFrame, timeframe: str = "W") -> pd.DataFrame:
        """
        แปลง timeframe ของข้อมูล

        Args:
            df: DataFrame ข้อมูล OHLCV daily
            timeframe: timeframe ที่ต้องการ ('W'=weekly, 'M'=monthly, 'Q'=quarterly)

        Returns:
            DataFrame ที่แปลง timeframe แล้ว
        """
        logger.info(f"กำลังแปลงข้อมูลเป็น timeframe: {timeframe}")

        # กำหนด aggregation rules
        ohlcv_rules = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }

        # กรองเฉพาะคอลัมน์ที่มีอยู่
        rules = {k: v for k, v in ohlcv_rules.items() if k in df.columns}

        # Resample
        df_resampled = df.resample(timeframe).agg(rules)

        # ลบแถวที่ไม่มีข้อมูล (วันหยุด)
        df_resampled = df_resampled.dropna(subset=["Close"])

        logger.info(
            f"แปลง timeframe สำเร็จ: {len(df)} -> {len(df_resampled)} แถว"
        )
        return df_resampled

    def download_multiple_tickers(
        self,
        tickers: list,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        ดาวน์โหลดข้อมูลหลาย tickers พร้อมกัน

        Args:
            tickers: รายการ ticker symbols
            start_date: วันเริ่มต้น
            end_date: วันสิ้นสุด

        Returns:
            dict ของ {ticker: DataFrame}
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.download_gold_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    save_csv=True,
                )
            except Exception as e:
                logger.error(f"ไม่สามารถดาวน์โหลด {ticker}: {e}")
                result[ticker] = None

        successful = sum(1 for v in result.values() if v is not None)
        logger.info(f"ดาวน์โหลดสำเร็จ {successful}/{len(tickers)} tickers")
        return result
