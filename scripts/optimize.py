#!/usr/bin/env python3
"""
scripts/optimize.py — Script สำหรับ Parameter Optimization

ค้นหา optimal parameters ด้วย:
- Grid Search
- Random Search
- Bayesian Optimization (ถ้ามี optuna)

บันทึก optimal parameters ลง config
"""

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.metrics import PerformanceMetrics
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.strategies.base_grid import BaseGridStrategy
from src.utils.config import ConfigLoader
from src.utils.logger import setup_project_logging

logger = logging.getLogger(__name__)


def quick_backtest(
    df: pd.DataFrame,
    params: dict,
    initial_capital: float = 100_000,
) -> float:
    """
    รัน backtest แบบ quick สำหรับ optimization

    Args:
        df: DataFrame ข้อมูล
        params: strategy parameters
        initial_capital: เงินทุนเริ่มต้น

    Returns:
        Sharpe Ratio (ค่าที่ต้องการ maximize)
    """
    try:
        params["initial_capital"] = initial_capital
        strategy = BaseGridStrategy(params)
        results = strategy.run(df)

        if results.empty or "equity" not in results.columns:
            return -999.0

        equity = results["equity"]
        if len(equity) < 30:
            return -999.0

        # คำนวณ Sharpe Ratio
        metrics = PerformanceMetrics()
        returns = equity.pct_change().dropna()
        sharpe = metrics.calculate_sharpe_ratio(returns)

        # Penalty สำหรับ drawdown สูง
        max_dd, _, _ = metrics.calculate_max_drawdown(equity)
        if max_dd < -0.20:  # ถ้า max DD > 20% → penalty
            sharpe -= (abs(max_dd) - 0.20) * 10

        return float(sharpe) if not np.isnan(sharpe) else -999.0

    except Exception as e:
        logger.debug(f"Error ใน quick_backtest: {e}")
        return -999.0


def grid_search(
    df: pd.DataFrame,
    param_grid: dict,
    n_best: int = 10,
) -> List[dict]:
    """
    Grid Search ค้นหา parameters ที่ดีที่สุด

    Args:
        df: DataFrame ข้อมูล (training set)
        param_grid: dict ของ {param_name: [values]}
        n_best: จำนวน best results ที่ต้องการ

    Returns:
        รายการ best parameter sets เรียงตาม score
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_combinations = list(product(*param_values))
    total = len(all_combinations)
    logger.info(f"Grid Search: {total} combinations")

    results = []

    for i, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        score = quick_backtest(df, params.copy())

        results.append({"params": params, "score": score})

        if (i + 1) % 50 == 0 or i + 1 == total:
            logger.info(f"  Progress: {i+1}/{total} ({(i+1)/total:.1%}), best={max(r['score'] for r in results):.4f}")

    # เรียงตาม score
    results.sort(key=lambda x: x["score"], reverse=True)
    best_results = results[:n_best]

    logger.info(f"\n🏆 Top {n_best} Results:")
    for j, r in enumerate(best_results[:5], 1):
        logger.info(f"  {j}. Score={r['score']:.4f}, Params={r['params']}")

    return best_results


def random_search(
    df: pd.DataFrame,
    param_ranges: dict,
    n_trials: int = 100,
    n_best: int = 10,
) -> List[dict]:
    """
    Random Search ค้นหา parameters

    Args:
        df: DataFrame ข้อมูล
        param_ranges: dict ของ {param_name: {'min': x, 'max': y, 'step': z}}
        n_trials: จำนวน trials
        n_best: จำนวน best results

    Returns:
        รายการ best parameter sets
    """
    logger.info(f"Random Search: {n_trials} trials")
    rng = np.random.default_rng(42)
    results = []

    for i in range(n_trials):
        params = {}

        for name, range_def in param_ranges.items():
            if "options" in range_def:
                params[name] = rng.choice(range_def["options"])
            else:
                min_v = range_def["min"]
                max_v = range_def["max"]
                step = range_def.get("step", None)

                if step:
                    choices = np.arange(min_v, max_v + step, step)
                    params[name] = float(rng.choice(choices))
                else:
                    params[name] = float(rng.uniform(min_v, max_v))

        score = quick_backtest(df, params.copy())
        results.append({"params": params, "score": score})

        if (i + 1) % 20 == 0:
            best = max(r["score"] for r in results)
            logger.info(f"  Trial {i+1}/{n_trials}, best score={best:.4f}")

    results.sort(key=lambda x: x["score"], reverse=True)
    best_results = results[:n_best]

    logger.info(f"\n🏆 Top {n_best} Results (Random Search):")
    for j, r in enumerate(best_results[:5], 1):
        logger.info(f"  {j}. Score={r['score']:.4f}, Params={r['params']}")

    return best_results


def walk_forward_validation(
    df: pd.DataFrame,
    params: dict,
    train_months: int = 18,
    test_months: int = 3,
) -> dict:
    """
    Walk-Forward Validation สำหรับ parameters

    Args:
        df: DataFrame ข้อมูล
        params: parameters ที่ต้องการทดสอบ
        train_months: จำนวนเดือนสำหรับ training
        test_months: จำนวนเดือนสำหรับ testing

    Returns:
        dict ของ out-of-sample metrics
    """
    logger.info(f"Walk-Forward Validation: train={train_months}mo, test={test_months}mo")

    train_days = train_months * 21  # ~21 trading days per month
    test_days = test_months * 21

    scores = []
    n = len(df)
    step = test_days

    start = train_days
    while start + test_days <= n:
        train_df = df.iloc[start - train_days : start]
        test_df = df.iloc[start : start + test_days]

        score = quick_backtest(test_df, params.copy())
        scores.append(score)
        start += step

    if scores:
        wf_results = {
            "n_folds": len(scores),
            "mean_sharpe": float(np.mean(scores)),
            "std_sharpe": float(np.std(scores)),
            "min_sharpe": float(min(scores)),
            "max_sharpe": float(max(scores)),
            "positive_folds": sum(1 for s in scores if s > 0),
        }
        logger.info(f"  Walk-Forward: mean={wf_results['mean_sharpe']:.4f} ± {wf_results['std_sharpe']:.4f}")
        return wf_results

    return {}


def save_optimal_params(optimal_params: dict, output_path: str) -> None:
    """
    บันทึก optimal parameters

    Args:
        optimal_params: dict ของ best parameters
        output_path: เส้นทางบันทึก
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimal_params, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"💾 บันทึก optimal parameters ที่ {output_path}")


def main():
    """Main function สำหรับ optimization"""
    parser = argparse.ArgumentParser(description="Optimize Parameters สำหรับ Gold Grid Bot")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--parameters", default="config/parameters.yaml")
    parser.add_argument("--method", default="random_search", choices=["grid_search", "random_search"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--output", default="reports/optimal_params.json")
    args = parser.parse_args()

    setup_project_logging(level="INFO")
    logger.info("🥇 Gold Grid Trading Bot — Parameter Optimization")
    logger.info(f"   Method: {args.method}, Trials: {args.n_trials}")

    # โหลด config
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.load_config()

    param_loader = ConfigLoader(args.parameters)
    parameters = param_loader.load_config(args.parameters)

    # โหลดข้อมูล
    data_cfg = config.get("data", {})
    raw_path = data_cfg.get("raw_data_path", "data/raw/")
    ticker = data_cfg.get("ticker", "GLD")

    loader = DataLoader(raw_path)
    csv_files = list(Path(raw_path).glob(f"{ticker}*.csv"))

    if csv_files:
        df_raw = loader.load_local_data(str(csv_files[0]))
    else:
        df_raw = loader.download_gold_data(ticker=ticker)

    processor = DataProcessor()
    df = processor.clean_data(df_raw)
    df = processor.handle_missing_values(df)

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)

    # ใช้เฉพาะ training set
    train_ratio = data_cfg.get("train_ratio", 0.70)
    train_end = int(len(df) * train_ratio)
    train_df = df.iloc[:train_end]

    logger.info(f"Training set: {len(train_df)} วัน")

    # รัน optimization
    if args.method == "grid_search":
        # ตัวอย่าง grid search ขนาดเล็ก
        param_grid = {
            "num_grids": [8, 10, 12, 15],
            "grid_spacing_pct": [0.01, 0.02, 0.03],
            "position_size_pct": [0.04, 0.05, 0.06],
        }
        best_results = grid_search(train_df, param_grid, n_best=10)

    else:  # random_search
        grid_params = parameters.get("grid_parameters", {})
        param_ranges = {
            "num_grids": {"min": 5, "max": 20, "step": 1},
            "grid_spacing_pct": {
                "min": grid_params.get("grid_spacing_pct", {}).get("min", 0.005),
                "max": grid_params.get("grid_spacing_pct", {}).get("max", 0.060),
                "step": 0.005,
            },
            "position_size_pct": {
                "min": 0.03,
                "max": 0.10,
                "step": 0.01,
            },
        }
        best_results = random_search(train_df, param_ranges, n_trials=args.n_trials)

    # Walk-Forward Validation บน best params
    if best_results:
        best_params = best_results[0]["params"]
        logger.info(f"\n🔄 Walk-Forward Validation บน Best Parameters: {best_params}")
        wf_results = walk_forward_validation(df, best_params)

        optimal = {
            "best_params": best_params,
            "optimization_score": best_results[0]["score"],
            "method": args.method,
            "walk_forward": wf_results,
            "top_10": best_results[:10],
        }

        save_optimal_params(optimal, args.output)

    logger.info("\n✅ Optimization เสร็จสิ้น!")


if __name__ == "__main__":
    main()
