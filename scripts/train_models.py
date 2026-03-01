#!/usr/bin/env python3
"""
scripts/train_models.py — Script สำหรับ Train AI Models

ขั้นตอน:
1. โหลดข้อมูล processed
2. Feature Engineering
3. Train Market Regime Detector (Random Forest)
4. Train Volatility Predictor (LSTM)
5. ประเมิน models และบันทึกผล
6. บันทึก models ลงไฟล์
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# เพิ่ม src ใน path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.regime_detector import MarketRegimeDetector
from src.models.volatility_predictor import VolatilityPredictor
from src.utils.config import ConfigLoader
from src.utils.logger import setup_project_logging

logger = logging.getLogger(__name__)


def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """
    โหลดและเตรียมข้อมูลสำหรับ training

    Args:
        config: configuration dict

    Returns:
        DataFrame ที่มี features ครบชุด
    """
    data_cfg = config.get("data", {})
    ticker = data_cfg.get("ticker", "GLD")
    start = data_cfg.get("start_date", "2004-01-01")
    end = data_cfg.get("end_date", "2024-12-31")
    raw_path = data_cfg.get("raw_data_path", "data/raw/")

    logger.info("📂 โหลดข้อมูล...")

    # ลองโหลดจาก local ก่อน
    loader = DataLoader(data_path=raw_path)
    csv_files = list(Path(raw_path).glob(f"{ticker}*.csv"))

    if csv_files:
        logger.info(f"พบไฟล์ local: {csv_files[0].name}")
        df = loader.load_local_data(str(csv_files[0]))
    else:
        logger.info(f"ดาวน์โหลดข้อมูล {ticker} จาก yfinance...")
        df = loader.download_gold_data(ticker=ticker, start_date=start, end_date=end)

    logger.info(f"โหลดข้อมูล: {len(df)} แถว")

    # ทำความสะอาดข้อมูล
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.handle_missing_values(df)

    # Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)

    logger.info(f"สร้าง features: {len(engineer.feature_columns)} features, {len(df)} แถว")
    return df


def train_regime_detector(
    df: pd.DataFrame,
    config: dict,
    save_path: str,
) -> dict:
    """
    Train Market Regime Detector

    Args:
        df: DataFrame ที่มี features
        config: configuration
        save_path: เส้นทางบันทึก model

    Returns:
        dict ของ evaluation metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("🤖 Train Market Regime Detector (Random Forest)")
    logger.info("=" * 60)

    model_cfg = config.get("models", {}).get("regime_detector", {})

    # สร้าง model
    detector = MarketRegimeDetector(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 10),
        random_state=model_cfg.get("random_state", 42),
    )

    # เตรียม data
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # เตรียม features และ labels
    X_train, y_train = detector.prepare_features(train_df)
    X_val, y_val = detector.prepare_features(val_df)
    X_test, y_test = detector.prepare_features(test_df)

    # Train
    detector.train(X_train, y_train)

    # Evaluate
    logger.info("\n📊 Validation Results:")
    val_results = detector.evaluate(X_val, y_val)

    logger.info("\n📊 Test Results:")
    test_results = detector.evaluate(X_test, y_test)

    # Feature Importance
    if detector.feature_importances_:
        sorted_features = sorted(
            detector.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("\n🔑 Top 10 Feature Importances:")
        for feat, imp in sorted_features[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

    # บันทึก model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    detector.save_model(save_path)
    logger.info(f"\n💾 บันทึก model ที่ {save_path}")

    return {
        "val_accuracy": val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
    }


def train_volatility_predictor(
    df: pd.DataFrame,
    config: dict,
    save_path: str,
) -> dict:
    """
    Train Volatility Predictor (LSTM)

    Args:
        df: DataFrame ที่มี features
        config: configuration
        save_path: เส้นทางบันทึก model

    Returns:
        dict ของ evaluation metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("🧠 Train Volatility Predictor (LSTM)")
    logger.info("=" * 60)

    model_cfg = config.get("models", {}).get("volatility_predictor", {})

    # สร้าง model
    predictor = VolatilityPredictor(
        sequence_length=model_cfg.get("sequence_length", 30),
        lstm_units=model_cfg.get("lstm_units", [64, 32]),
        dropout_rate=model_cfg.get("dropout_rate", 0.2),
        learning_rate=model_cfg.get("learning_rate", 0.001),
    )

    # เตรียม sequences
    X, y = predictor.prepare_sequences(df, target_col="hist_vol_20d")

    # แบ่ง data
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Train
    history = predictor.train(
        X_train, y_train,
        epochs=model_cfg.get("epochs", 100),
        batch_size=model_cfg.get("batch_size", 32),
        validation_split=0.0,  # ใช้ explicit val set
        patience=model_cfg.get("patience", 15),
    )

    # Evaluate
    logger.info("\n📊 Test Results:")
    test_metrics = predictor.evaluate(X_test, y_test)

    # บันทึก model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(save_path)
    logger.info(f"\n💾 บันทึก model ที่ {save_path}")

    return test_metrics


def main():
    """Main function สำหรับ training"""
    parser = argparse.ArgumentParser(description="Train AI Models สำหรับ Gold Grid Bot")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-regime", action="store_true", help="ข้าม Regime Detector")
    parser.add_argument("--skip-volatility", action="store_true", help="ข้าม Volatility Predictor")
    args = parser.parse_args()

    setup_project_logging(level="INFO")
    logger.info("🥇 Gold Grid Trading Bot — Model Training")

    # โหลด config
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.load_config()

    # โหลดและเตรียมข้อมูล
    df = load_and_prepare_data(config)

    all_results = {}

    # Train Regime Detector
    if not args.skip_regime:
        regime_path = config.get("models", {}).get("regime_detector", {}).get(
            "model_save_path", "data/models/regime_detector.pkl"
        )
        regime_results = train_regime_detector(df, config, regime_path)
        all_results["regime_detector"] = regime_results

    # Train Volatility Predictor
    if not args.skip_volatility:
        vol_path = config.get("models", {}).get("volatility_predictor", {}).get(
            "model_save_path", "data/models/volatility_predictor.h5"
        )
        vol_results = train_volatility_predictor(df, config, vol_path)
        all_results["volatility_predictor"] = vol_results

    # แสดงสรุปผล
    logger.info("\n" + "=" * 60)
    logger.info("🎯 สรุปผล Training")
    logger.info("=" * 60)

    if "regime_detector" in all_results:
        r = all_results["regime_detector"]
        logger.info(f"Regime Detector — Val Acc: {r['val_accuracy']:.4f}, Test Acc: {r['test_accuracy']:.4f}")

    if "volatility_predictor" in all_results:
        v = all_results["volatility_predictor"]
        logger.info(f"Volatility Predictor — RMSE: {v.get('rmse', 'N/A'):.6f}, MAPE: {v.get('mape', 'N/A'):.2f}%")

    logger.info("\n✅ Training เสร็จสิ้น!")


if __name__ == "__main__":
    main()
