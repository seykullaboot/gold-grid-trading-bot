"""
src/utils/logger.py — Logging Setup Utilities

ตั้งค่า logger สำหรับโปรเจกต์ Gold Grid Trading Bot
รองรับ console output และ file logging
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    ตั้งค่า logger สำหรับ module

    Args:
        name: ชื่อ logger
        log_file: เส้นทางไฟล์ log (None = ไม่บันทึกไฟล์)
        level: logging level (logging.DEBUG, INFO, WARNING, ERROR)
        format_str: format string (ใช้ค่า default ถ้าไม่ระบุ)
        max_bytes: ขนาดไฟล์ log สูงสุด (bytes)
        backup_count: จำนวนไฟล์ backup สูงสุด
        console: แสดงบน console หรือไม่

    Returns:
        Logger object ที่ตั้งค่าแล้ว
    """
    logger = logging.getLogger(name)

    # ป้องกัน duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # กำหนด format
    if format_str is None:
        format_str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler (rotating)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    ดึง logger instance ที่มีอยู่แล้ว (หรือสร้างใหม่)

    Args:
        name: ชื่อ logger (ปกติใช้ __name__ ของ module)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # ถ้ายังไม่มี handler → ตั้งค่า default
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logger(name, console=True)

    return logger


def setup_project_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    log_file: str = "trading_bot.log",
) -> None:
    """
    ตั้งค่า logging สำหรับทั้งโปรเจกต์

    เรียกครั้งเดียวตอนเริ่มต้น script หลัก

    Args:
        log_dir: โฟลเดอร์สำหรับเก็บไฟล์ log
        level: ระดับ logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: ชื่อไฟล์ log
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # สร้าง log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # ตั้งค่า root logger
    setup_logger(
        name="gold_grid",
        log_file=log_path,
        level=log_level,
        console=True,
    )

    logging.getLogger("gold_grid").info(
        f"Logging initialized: level={level}, file={log_path}"
    )
