"""
src/data/processor.py — โมดูลสำหรับ preprocessing และทำความสะอาดข้อมูล

รวมถึง:
- ทำความสะอาดข้อมูล
- จัดการ missing values
- Normalization / Standardization
- แบ่งข้อมูล train/val/test
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """คลาสสำหรับ preprocessing และแปลงข้อมูล"""

    def __init__(self):
        """กำหนดค่าเริ่มต้น DataProcessor"""
        self.scaler = None
        self.scaler_type = None
        self._fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ทำความสะอาดข้อมูลโดยรวม

        Args:
            df: DataFrame ข้อมูลดิบ

        Returns:
            DataFrame ที่ผ่านการทำความสะอาดแล้ว
        """
        logger.info(f"กำลังทำความสะอาดข้อมูล: {len(df)} แถว")
        df = df.copy()

        # ลบแถวที่ซ้ำกัน
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df[~df.duplicated()]
            logger.warning(f"ลบแถวซ้ำ {duplicates} แถว")

        # ตรวจสอบและแก้ไข OHLC inconsistency
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            # High ต้องมากกว่าหรือเท่ากับ Open, Close, Low
            df["High"] = df[["High", "Open", "Close", "Low"]].max(axis=1)
            df["Low"] = df[["Low", "Open", "Close", "High"]].min(axis=1)

        # ลบแถวที่ Close เป็น 0 หรือ NaN
        before = len(df)
        df = df[df["Close"] > 0] if "Close" in df.columns else df
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"ลบแถวที่ Close <= 0 จำนวน {removed} แถว")

        # เรียงลำดับ index ตามวันที่
        df = df.sort_index()

        logger.info(f"ทำความสะอาดข้อมูลสำเร็จ: เหลือ {len(df)} แถว")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        max_gap: int = 5,
    ) -> pd.DataFrame:
        """
        จัดการค่าที่หายไป (Missing Values)

        Args:
            df: DataFrame ที่มี missing values
            method: วิธีการเติมค่า ('ffill', 'bfill', 'interpolate', 'drop')
            max_gap: จำนวนแถวสูงสุดที่จะเติมค่าต่อเนื่อง

        Returns:
            DataFrame ที่จัดการ missing values แล้ว
        """
        missing_before = df.isnull().sum().sum()
        if missing_before == 0:
            logger.info("ไม่พบ missing values")
            return df

        logger.info(f"พบ missing values ทั้งหมด {missing_before} ค่า")
        df = df.copy()

        if method == "ffill":
            # Forward fill (เติมค่าจากวันก่อนหน้า) — เหมาะสำหรับข้อมูลราคา
            df = df.fillna(method="ffill", limit=max_gap)
            # Backward fill สำหรับ missing ต้นข้อมูล
            df = df.fillna(method="bfill", limit=max_gap)

        elif method == "bfill":
            df = df.fillna(method="bfill", limit=max_gap)

        elif method == "interpolate":
            # Linear interpolation
            df = df.interpolate(method="linear", limit=max_gap)

        elif method == "drop":
            # ลบแถวที่มี missing values
            df = df.dropna()

        # ตรวจสอบ missing ที่เหลืออยู่
        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            logger.warning(f"ยังมี missing values เหลืออยู่ {missing_after} ค่า — ลบแถวเหล่านี้")
            df = df.dropna()

        logger.info(f"จัดการ missing values สำเร็จ: {missing_before} -> 0 missing values")
        return df

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: list = None,
        method: str = "minmax",
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize หรือ Standardize ข้อมูล

        Args:
            df: DataFrame ที่ต้องการ normalize
            columns: รายการคอลัมน์ที่ต้องการ normalize (None = ทุกคอลัมน์ตัวเลข)
            method: 'minmax' (0-1) หรือ 'standard' (Z-score)
            fit: True = fit และ transform, False = transform อย่างเดียว

        Returns:
            DataFrame ที่ normalize แล้ว
        """
        df = df.copy()

        # เลือกคอลัมน์ตัวเลข
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"กำลัง normalize {len(columns)} คอลัมน์ ด้วยวิธี {method}")

        # สร้าง scaler ถ้าจำเป็น
        if fit or not self._fitted:
            if method == "minmax":
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif method == "standard":
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"ไม่รู้จัก method: {method}")
            self.scaler_type = method

            df[columns] = self.scaler.fit_transform(df[columns])
            self._fitted = True
        else:
            # ใช้ scaler ที่ fit แล้ว (สำหรับ test data)
            df[columns] = self.scaler.transform(df[columns])

        return df

    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: list = None,
    ) -> pd.DataFrame:
        """
        แปลงข้อมูลกลับจาก normalized scale

        Args:
            df: DataFrame ที่ normalize แล้ว
            columns: รายการคอลัมน์ (None = ทุกคอลัมน์ตัวเลข)

        Returns:
            DataFrame ที่แปลงกลับแล้ว
        """
        if not self._fitted or self.scaler is None:
            raise RuntimeError("ยังไม่ได้ fit scaler — ต้อง normalize ก่อน")

        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df[columns] = self.scaler.inverse_transform(df[columns])
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        แบ่งข้อมูลเป็น train, validation, test sets

        Args:
            df: DataFrame ข้อมูลทั้งหมด
            train_ratio: สัดส่วน training data (default 70%)
            val_ratio: สัดส่วน validation data (default 15%)
            shuffle: สุ่มลำดับข้อมูลก่อนแบ่ง (ไม่แนะนำสำหรับ time series)

        Returns:
            tuple (train_df, val_df, test_df)
        """
        n = len(df)
        test_ratio = 1 - train_ratio - val_ratio

        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio ต้องน้อยกว่า 1.0")

        # คำนวณจุดแบ่ง
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if shuffle:
            # สุ่มลำดับ (ใช้กับ non-time-series เท่านั้น)
            df = df.sample(frac=1, random_state=42)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"แบ่งข้อมูล: train={len(train_df)} ({train_ratio:.0%}), "
            f"val={len(val_df)} ({val_ratio:.0%}), "
            f"test={len(test_df)} ({test_ratio:.0%})"
        )

        return train_df, val_df, test_df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่มคอลัมน์ผลตอบแทน (Returns)

        Args:
            df: DataFrame ที่มีคอลัมน์ Close

        Returns:
            DataFrame ที่มีคอลัมน์ returns เพิ่มเติม
        """
        df = df.copy()

        # Daily returns (%)
        df["returns"] = df["Close"].pct_change()

        # Log returns
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Rolling volatility (20 วัน)
        df["volatility_20d"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

        # Cumulative returns
        df["cum_returns"] = (1 + df["returns"]).cumprod() - 1

        logger.info("เพิ่มคอลัมน์ returns สำเร็จ")
        return df

    def create_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        return_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        สร้าง labels สำหรับ supervised learning

        Args:
            df: DataFrame ที่มีคอลัมน์ Close
            forward_days: จำนวนวันล่วงหน้าสำหรับคำนวณผลตอบแทน
            return_threshold: threshold สำหรับแบ่ง buy/sell/hold

        Returns:
            DataFrame ที่มีคอลัมน์ label เพิ่มเติม
        """
        df = df.copy()

        # คำนวณ forward returns
        df["forward_return"] = df["Close"].pct_change(periods=forward_days).shift(-forward_days)

        # แบ่ง labels: 1=Buy, -1=Sell, 0=Hold
        df["label"] = 0
        df.loc[df["forward_return"] > return_threshold, "label"] = 1
        df.loc[df["forward_return"] < -return_threshold, "label"] = -1

        label_counts = df["label"].value_counts()
        logger.info(f"สร้าง labels: Buy={label_counts.get(1, 0)}, Sell={label_counts.get(-1, 0)}, Hold={label_counts.get(0, 0)}")
        return df
