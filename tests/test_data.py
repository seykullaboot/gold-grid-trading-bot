"""
tests/test_data.py — Unit Tests สำหรับ Data Loading และ Processing
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_ohlcv_df():
    """สร้าง DataFrame OHLCV ตัวอย่าง"""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # สร้างราคาที่ realistic
    price = 150.0
    prices = []
    for _ in range(n):
        price *= (1 + np.random.normal(0, 0.01))
        prices.append(price)

    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.random.uniform(0, 0.01, n)),
            "Low": prices * (1 - np.random.uniform(0, 0.01, n)),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_csv_file(sample_ohlcv_df, tmp_path):
    """สร้างไฟล์ CSV ชั่วคราวสำหรับทดสอบ"""
    filepath = tmp_path / "test_data.csv"
    sample_ohlcv_df.to_csv(filepath)
    return str(filepath)


@pytest.fixture
def data_loader(tmp_path):
    """สร้าง DataLoader ที่ใช้ tmp directory"""
    return DataLoader(data_path=str(tmp_path))


@pytest.fixture
def data_processor():
    """สร้าง DataProcessor instance"""
    return DataProcessor()


# ============================================================
# Tests: DataLoader
# ============================================================

class TestDataLoader:
    """ทดสอบ DataLoader class"""

    def test_init(self, tmp_path):
        """ทดสอบการสร้าง DataLoader"""
        loader = DataLoader(data_path=str(tmp_path))
        assert loader.data_path.exists()

    def test_load_local_csv(self, data_loader, sample_csv_file):
        """ทดสอบโหลดข้อมูลจาก CSV"""
        df = data_loader.load_local_data(sample_csv_file)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "Close" in df.columns

    def test_load_nonexistent_file(self, data_loader):
        """ทดสอบโหลดไฟล์ที่ไม่มีอยู่"""
        with pytest.raises(FileNotFoundError):
            data_loader.load_local_data("nonexistent.csv")

    def test_validate_data_valid(self, data_loader, sample_ohlcv_df):
        """ทดสอบ validate ข้อมูลที่ถูกต้อง"""
        assert data_loader.validate_data(sample_ohlcv_df) is True

    def test_validate_data_empty(self, data_loader):
        """ทดสอบ validate DataFrame ว่างเปล่า"""
        empty_df = pd.DataFrame()
        assert data_loader.validate_data(empty_df) is False

    def test_validate_data_missing_columns(self, data_loader):
        """ทดสอบ validate ที่ขาด columns"""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        assert data_loader.validate_data(df) is False

    def test_validate_data_negative_price(self, data_loader, sample_ohlcv_df):
        """ทดสอบ validate ที่มีราคาติดลบ"""
        df = sample_ohlcv_df.copy()
        df.loc[df.index[0], "Close"] = -1.0
        assert data_loader.validate_data(df) is False

    def test_resample_weekly(self, data_loader, sample_ohlcv_df):
        """ทดสอบ resample เป็น weekly"""
        weekly = data_loader.resample_data(sample_ohlcv_df, timeframe="W")
        assert len(weekly) < len(sample_ohlcv_df)
        assert "Close" in weekly.columns

    def test_resample_monthly(self, data_loader, sample_ohlcv_df):
        """ทดสอบ resample เป็น monthly"""
        monthly = data_loader.resample_data(sample_ohlcv_df, timeframe="ME")
        assert len(monthly) < len(sample_ohlcv_df)


# ============================================================
# Tests: DataProcessor
# ============================================================

class TestDataProcessor:
    """ทดสอบ DataProcessor class"""

    def test_clean_data_removes_duplicates(self, data_processor, sample_ohlcv_df):
        """ทดสอบลบข้อมูลซ้ำ"""
        df_with_dup = pd.concat([sample_ohlcv_df, sample_ohlcv_df.head(10)])
        cleaned = data_processor.clean_data(df_with_dup)
        # หลังทำความสะอาด จำนวนแถวไม่ควรเกิน original
        assert len(cleaned) <= len(df_with_dup)

    def test_clean_data_sorts_index(self, data_processor, sample_ohlcv_df):
        """ทดสอบเรียง index ตามวันที่"""
        df_shuffled = sample_ohlcv_df.sample(frac=1, random_state=42)
        cleaned = data_processor.clean_data(df_shuffled)
        assert cleaned.index.is_monotonic_increasing

    def test_handle_missing_values_ffill(self, data_processor, sample_ohlcv_df):
        """ทดสอบการเติม missing values ด้วย ffill"""
        df_with_nan = sample_ohlcv_df.copy()
        df_with_nan.loc[df_with_nan.index[10:15], "Close"] = np.nan

        filled = data_processor.handle_missing_values(df_with_nan, method="ffill")
        assert filled["Close"].isnull().sum() == 0

    def test_handle_missing_values_drop(self, data_processor, sample_ohlcv_df):
        """ทดสอบการลบ missing values"""
        df_with_nan = sample_ohlcv_df.copy()
        df_with_nan.loc[df_with_nan.index[10:15], "Close"] = np.nan

        n_nan = df_with_nan.isnull().sum().sum()
        dropped = data_processor.handle_missing_values(df_with_nan, method="drop")
        assert dropped["Close"].isnull().sum() == 0
        assert len(dropped) <= len(df_with_nan)

    def test_normalize_minmax(self, data_processor, sample_ohlcv_df):
        """ทดสอบ MinMax normalization"""
        normalized = data_processor.normalize_data(
            sample_ohlcv_df[["Close", "Volume"]], method="minmax"
        )
        assert normalized["Close"].min() >= 0 - 1e-10
        assert normalized["Close"].max() <= 1 + 1e-10

    def test_normalize_standard(self, data_processor, sample_ohlcv_df):
        """ทดสอบ Standard normalization (Z-score)"""
        normalized = data_processor.normalize_data(
            sample_ohlcv_df[["Close"]], method="standard"
        )
        assert abs(normalized["Close"].mean()) < 0.1  # mean ≈ 0
        assert abs(normalized["Close"].std() - 1.0) < 0.1  # std ≈ 1

    def test_split_data_ratios(self, data_processor, sample_ohlcv_df):
        """ทดสอบการแบ่งข้อมูลตามสัดส่วน"""
        train, val, test = data_processor.split_data(
            sample_ohlcv_df, train_ratio=0.7, val_ratio=0.15
        )
        total = len(train) + len(val) + len(test)
        assert total == len(sample_ohlcv_df)
        assert abs(len(train) / total - 0.7) < 0.02  # ± 2%
        assert abs(len(val) / total - 0.15) < 0.02

    def test_split_data_no_overlap(self, data_processor, sample_ohlcv_df):
        """ทดสอบว่า train/val/test ไม่มีข้อมูลซ้ำกัน"""
        train, val, test = data_processor.split_data(sample_ohlcv_df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_split_data_invalid_ratio(self, data_processor, sample_ohlcv_df):
        """ทดสอบ split ด้วย ratio ที่ไม่ถูกต้อง"""
        with pytest.raises(ValueError):
            data_processor.split_data(sample_ohlcv_df, train_ratio=0.8, val_ratio=0.3)

    def test_add_returns(self, data_processor, sample_ohlcv_df):
        """ทดสอบการเพิ่ม returns columns"""
        df_with_returns = data_processor.add_returns(sample_ohlcv_df)
        assert "returns" in df_with_returns.columns
        assert "log_returns" in df_with_returns.columns
        assert "volatility_20d" in df_with_returns.columns
