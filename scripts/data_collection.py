#!/usr/bin/env python3
"""
scripts/data_collection.py — Script ดาวน์โหลดข้อมูลทองคำ

ดาวน์โหลดข้อมูล:
1. GLD (SPDR Gold Shares ETF) ย้อนหลัง 20 ปี
2. Gold Futures (GC=F)
3. Macroeconomic indicators (DXY, TLT, SPY)
4. บันทึกเป็น CSV ใน data/raw/
"""

import argparse
import logging
import sys
from pathlib import Path

# เพิ่ม src ใน path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.utils.logger import setup_project_logging

logger = logging.getLogger(__name__)


def download_gold_data(loader: DataLoader, start: str, end: str) -> None:
    """
    ดาวน์โหลดข้อมูลทองคำหลัก

    Args:
        loader: DataLoader instance
        start: วันเริ่มต้น
        end: วันสิ้นสุด
    """
    logger.info("=" * 60)
    logger.info("📥 ดาวน์โหลดข้อมูลทองคำ")
    logger.info("=" * 60)

    # รายการ tickers ที่ต้องการ
    gold_tickers = {
        "GLD": "SPDR Gold Shares ETF",
        "GC=F": "Gold Futures",
        "IAU": "iShares Gold Trust",
    }

    for ticker, name in gold_tickers.items():
        try:
            logger.info(f"กำลังดาวน์โหลด {name} ({ticker})...")
            df = loader.download_gold_data(
                ticker=ticker,
                start_date=start,
                end_date=end,
                save_csv=True,
            )
            logger.info(f"✅ {ticker}: {len(df):,} แถว ({df.index[0].date()} - {df.index[-1].date()})")
        except Exception as e:
            logger.error(f"❌ ไม่สามารถดาวน์โหลด {ticker}: {e}")


def download_macro_indicators(loader: DataLoader, start: str, end: str) -> None:
    """
    ดาวน์โหลดข้อมูล Macroeconomic indicators

    ใช้เป็น features เพิ่มเติมสำหรับ AI models

    Args:
        loader: DataLoader instance
        start: วันเริ่มต้น
        end: วันสิ้นสุด
    """
    logger.info("=" * 60)
    logger.info("📊 ดาวน์โหลด Macroeconomic Indicators")
    logger.info("=" * 60)

    macro_tickers = {
        "SPY": "S&P 500 ETF (market sentiment)",
        "TLT": "20+ Year Treasury Bond ETF (safe haven)",
        "DX-Y.NYB": "US Dollar Index",
        "^VIX": "CBOE Volatility Index",
        "USO": "United States Oil Fund",
        "SLV": "iShares Silver Trust",
    }

    for ticker, name in macro_tickers.items():
        try:
            logger.info(f"กำลังดาวน์โหลด {name} ({ticker})...")
            df = loader.download_gold_data(
                ticker=ticker,
                start_date=start,
                end_date=end,
                save_csv=True,
            )
            logger.info(f"✅ {ticker}: {len(df):,} แถว")
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถดาวน์โหลด {ticker}: {e}")


def validate_downloaded_data(data_path: str) -> None:
    """
    ตรวจสอบข้อมูลที่ดาวน์โหลดมา

    Args:
        data_path: เส้นทาง data/raw/
    """
    logger.info("=" * 60)
    logger.info("🔍 ตรวจสอบข้อมูลที่ดาวน์โหลด")
    logger.info("=" * 60)

    loader = DataLoader(data_path)
    csv_files = list(Path(data_path).glob("*.csv"))

    if not csv_files:
        logger.warning("ไม่พบไฟล์ CSV ใน data/raw/")
        return

    for csv_file in csv_files:
        try:
            df = loader.load_local_data(str(csv_file))
            is_valid = loader.validate_data(df)
            status = "✅" if is_valid else "⚠️"
            logger.info(
                f"{status} {csv_file.name}: {len(df):,} แถว, "
                f"Missing: {df.isnull().sum().sum()}"
            )
        except Exception as e:
            logger.error(f"❌ ไม่สามารถตรวจสอบ {csv_file.name}: {e}")


def generate_data_summary(data_path: str) -> None:
    """
    สร้างสรุปข้อมูลทั้งหมด

    Args:
        data_path: เส้นทาง data/raw/
    """
    import pandas as pd

    logger.info("\n" + "=" * 60)
    logger.info("📋 สรุปข้อมูลทั้งหมด")
    logger.info("=" * 60)

    csv_files = list(Path(data_path).glob("*.csv"))
    summary = []

    for f in csv_files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            summary.append({
                "File": f.name,
                "Rows": len(df),
                "Start": str(df.index.min().date()),
                "End": str(df.index.max().date()),
                "Missing": df.isnull().sum().sum(),
                "Size (KB)": round(f.stat().st_size / 1024, 1),
            })
        except Exception:
            pass

    if summary:
        import pandas as pd
        summary_df = pd.DataFrame(summary)
        logger.info(f"\n{summary_df.to_string(index=False)}")


def main():
    """Main function สำหรับ data collection"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="ดาวน์โหลดข้อมูลสำหรับ Gold Grid Trading Bot"
    )
    parser.add_argument("--start", default="2004-01-01", help="วันเริ่มต้น (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="วันสิ้นสุด (YYYY-MM-DD)")
    parser.add_argument("--data-path", default="data/raw/", help="เส้นทางบันทึกข้อมูล")
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="ข้ามการดาวน์โหลด macro indicators",
    )
    args = parser.parse_args()

    # ตั้งค่า logging
    setup_project_logging(level="INFO")

    logger.info("🥇 Gold Grid Trading Bot — Data Collection")
    logger.info(f"   Period: {args.start} ถึง {args.end}")
    logger.info(f"   Data path: {args.data_path}")

    # สร้าง DataLoader
    loader = DataLoader(data_path=args.data_path)

    # ดาวน์โหลดข้อมูลทองคำ
    download_gold_data(loader, args.start, args.end)

    # ดาวน์โหลด macro indicators
    if not args.skip_macro:
        download_macro_indicators(loader, args.start, args.end)

    # ตรวจสอบข้อมูล
    validate_downloaded_data(args.data_path)

    # สรุป
    generate_data_summary(args.data_path)

    logger.info("\n✅ Data Collection เสร็จสิ้น!")


if __name__ == "__main__":
    main()
