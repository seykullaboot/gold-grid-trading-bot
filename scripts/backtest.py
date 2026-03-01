#!/usr/bin/env python3
"""
scripts/backtest.py — Script สำหรับรัน Backtesting

รัน backtest เปรียบเทียบ 3 strategies:
1. Buy & Hold (Benchmark)
2. Classic Grid Strategy
3. AI Adaptive Grid Strategy

แสดงผล metrics ครบชุดและบันทึก reports
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import PerformanceMetrics
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.regime_detector import MarketRegimeDetector
from src.models.volatility_predictor import VolatilityPredictor
from src.strategies.adaptive_grid import AdaptiveGridStrategy
from src.strategies.base_grid import BaseGridStrategy
from src.utils.config import ConfigLoader
from src.utils.logger import setup_project_logging

logger = logging.getLogger(__name__)


def load_data_and_features(config: dict) -> pd.DataFrame:
    """
    โหลดข้อมูลและสร้าง features

    Args:
        config: configuration dict

    Returns:
        DataFrame พร้อม features
    """
    data_cfg = config.get("data", {})
    raw_path = data_cfg.get("raw_data_path", "data/raw/")
    ticker = data_cfg.get("ticker", "GLD")

    loader = DataLoader(raw_path)
    csv_files = list(Path(raw_path).glob(f"{ticker}*.csv"))

    if csv_files:
        df = loader.load_local_data(str(csv_files[0]))
    else:
        df = loader.download_gold_data(
            ticker=ticker,
            start_date=data_cfg.get("start_date", "2004-01-01"),
            end_date=data_cfg.get("end_date", "2024-12-31"),
        )

    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.handle_missing_values(df)

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)

    return df


def load_ai_predictions(
    df: pd.DataFrame,
    models_path: str,
    config: dict,
) -> pd.DataFrame:
    """
    โหลด AI models และสร้าง predictions

    Args:
        df: DataFrame ที่มี features
        models_path: เส้นทาง models
        config: configuration

    Returns:
        DataFrame ที่มี regime และ predicted_volatility
    """
    predictions = pd.DataFrame(index=df.index)
    predictions["regime"] = "ranging"
    predictions["predicted_volatility"] = 0.20

    # โหลด Regime Detector
    regime_path = Path(models_path) / "regime_detector.pkl"
    if regime_path.exists():
        try:
            detector = MarketRegimeDetector()
            detector.load_model(str(regime_path))
            regime_series = detector.predict_regime_series(df)
            predictions["regime"] = regime_series
            logger.info(f"✅ โหลด Regime Detector: {regime_series.value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถโหลด Regime Detector: {e}")
    else:
        logger.warning(f"ไม่พบ Regime Detector ที่ {regime_path} — ใช้ default 'ranging'")

    # โหลด Volatility Predictor
    vol_path = Path(models_path) / "volatility_predictor.h5"
    if vol_path.exists():
        try:
            predictor = VolatilityPredictor()
            predictor.load_model(str(vol_path))

            X, _ = predictor.prepare_sequences(df)
            vol_preds = predictor.predict(X)

            # align index
            vol_index = df.index[predictor.sequence_length:]
            vol_series = pd.Series(vol_preds, index=vol_index[:len(vol_preds)])
            predictions.loc[vol_series.index, "predicted_volatility"] = vol_series
            predictions["predicted_volatility"] = predictions["predicted_volatility"].fillna(
                df["hist_vol_20d"] if "hist_vol_20d" in df.columns else 0.20
            )
            logger.info(f"✅ โหลด Volatility Predictor: mean vol={vol_preds.mean():.4f}")
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถโหลด Volatility Predictor: {e}")
    else:
        logger.warning(f"ไม่พบ Volatility Predictor ที่ {vol_path} — ใช้ historical volatility")
        if "hist_vol_20d" in df.columns:
            predictions["predicted_volatility"] = df["hist_vol_20d"]

    return predictions


def run_backtest(
    df: pd.DataFrame,
    strategy,
    config: dict,
    ai_predictions: pd.DataFrame = None,
    strategy_name: str = "Strategy",
) -> tuple:
    """
    รัน backtest สำหรับ strategy

    Args:
        df: DataFrame ข้อมูล
        strategy: Strategy object
        config: configuration
        ai_predictions: AI predictions (optional)
        strategy_name: ชื่อ strategy

    Returns:
        tuple (results_df, metrics_dict)
    """
    bt_cfg = config.get("backtesting", {})

    backtest_config = BacktestConfig(
        initial_capital=bt_cfg.get("initial_capital", 100_000),
        commission_pct=bt_cfg.get("commission_pct", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
        risk_free_rate=bt_cfg.get("risk_free_rate", 0.05),
    )

    engine = BacktestEngine(df, strategy, config=backtest_config)

    # ใช้เฉพาะ test set สำหรับ backtest
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_start_idx = int(len(df) * (train_ratio + val_ratio))
    test_start_date = str(df.index[test_start_idx].date())

    results = engine.run(start_date=test_start_date, ai_predictions=ai_predictions)

    # คำนวณ metrics
    metrics_calc = PerformanceMetrics(
        risk_free_rate=bt_cfg.get("risk_free_rate", 0.05)
    )

    equity_curve = results["equity"]
    benchmark_curve = results.get("benchmark_equity", None)

    report = metrics_calc.generate_report(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        strategy_name=strategy_name,
    )

    return results, report


def print_comparison_table(reports: dict) -> None:
    """
    แสดงตารางเปรียบเทียบ strategies

    Args:
        reports: dict ของ {strategy_name: metrics_dict}
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 ตารางเปรียบเทียบ Performance")
    logger.info("=" * 80)

    headers = ["Metric", "Classic Grid", "AI Adaptive Grid", "Buy & Hold"]
    col_width = 22

    header_line = "|".join(f" {h:<{col_width-2}} " for h in headers)
    logger.info(f"|{header_line}|")
    logger.info(f"|{'-' * (col_width * len(headers) + len(headers) - 1)}|")

    metrics_to_show = [
        ("Total Return", "returns.total_return", "{:.2%}"),
        ("CAGR", "returns.cagr", "{:.2%}"),
        ("Sharpe Ratio", "risk_adjusted.sharpe_ratio", "{:.3f}"),
        ("Sortino Ratio", "risk_adjusted.sortino_ratio", "{:.3f}"),
        ("Max Drawdown", "drawdown.max_drawdown", "{:.2%}"),
        ("Calmar Ratio", "risk_adjusted.calmar_ratio", "{:.3f}"),
    ]

    def get_nested(d, path):
        keys = path.split(".")
        v = d
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, "N/A")
            else:
                return "N/A"
        return v

    strategy_names = ["Classic Grid", "AI Adaptive Grid"]

    for metric_name, path, fmt in metrics_to_show:
        row = [f" {metric_name:<{col_width-2}} "]
        for sname in strategy_names:
            if sname in reports:
                val = get_nested(reports[sname], path)
                try:
                    row.append(f" {fmt.format(val):<{col_width-2}} ")
                except Exception:
                    row.append(f" {'N/A':<{col_width-2}} ")
            else:
                row.append(f" {'N/A':<{col_width-2}} ")

        # Buy & Hold จาก benchmark ใน Classic Grid
        if "Classic Grid" in reports:
            bm = reports["Classic Grid"].get("benchmark", {})
            val = get_nested(bm, path.split(".")[-1]) if bm else "N/A"
            try:
                row.append(f" {fmt.format(val):<{col_width-2}} ")
            except Exception:
                row.append(f" {'N/A':<{col_width-2}} ")
        else:
            row.append(f" {'N/A':<{col_width-2}} ")

        logger.info(f"|{'|'.join(row)}|")


def save_reports(reports: dict, output_dir: str = "reports/") -> None:
    """
    บันทึก reports เป็น JSON

    Args:
        reports: dict ของ reports
        output_dir: โฟลเดอร์สำหรับบันทึก
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, report in reports.items():
        filename = name.lower().replace(" ", "_") + "_report.json"
        filepath = Path(output_dir) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 บันทึก report ที่ {filepath}")


def main():
    """Main function สำหรับ backtesting"""
    parser = argparse.ArgumentParser(description="รัน Backtest สำหรับ Gold Grid Bot")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models-path", default="data/models/")
    parser.add_argument("--output-dir", default="reports/")
    parser.add_argument("--skip-adaptive", action="store_true")
    args = parser.parse_args()

    setup_project_logging(level="INFO")
    logger.info("🥇 Gold Grid Trading Bot — Backtesting")

    # โหลด config
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.load_config()

    # โหลดข้อมูล
    logger.info("\n📂 โหลดข้อมูล...")
    df = load_data_and_features(config)
    logger.info(f"ข้อมูล: {len(df)} วัน ({df.index[0].date()} - {df.index[-1].date()})")

    # โหลด AI predictions
    ai_predictions = load_ai_predictions(df, args.models_path, config)

    reports = {}

    # --- Classic Grid Backtest ---
    logger.info("\n🔲 รัน Classic Grid Backtest...")
    classic_params = config.get("strategy", {}).get("classic_grid", {})
    classic_strategy = BaseGridStrategy(classic_params)
    classic_results, classic_report = run_backtest(
        df, classic_strategy, config, strategy_name="Classic Grid"
    )
    reports["Classic Grid"] = classic_report

    # --- Adaptive Grid Backtest ---
    if not args.skip_adaptive:
        logger.info("\n🧠 รัน AI Adaptive Grid Backtest...")
        adaptive_params = config.get("strategy", {}).get("adaptive_grid", {})
        adaptive_strategy = AdaptiveGridStrategy(adaptive_params)
        adaptive_results, adaptive_report = run_backtest(
            df, adaptive_strategy, config,
            ai_predictions=ai_predictions,
            strategy_name="AI Adaptive Grid",
        )
        reports["AI Adaptive Grid"] = adaptive_report

    # แสดงตารางเปรียบเทียบ
    print_comparison_table(reports)

    # บันทึก reports
    save_reports(reports, args.output_dir)

    logger.info("\n✅ Backtesting เสร็จสิ้น!")


if __name__ == "__main__":
    main()
