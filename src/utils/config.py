"""
src/utils/config.py — Configuration Loader

โหลดและจัดการ configuration จาก YAML files
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    โหลดและจัดการ configuration จากไฟล์ YAML

    รองรับ:
    - โหลด config.yaml (ค่าตั้งต้น)
    - โหลด parameters.yaml (parameter ranges)
    - ดึง sub-configs แต่ละส่วน
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        กำหนดค่าเริ่มต้น

        Args:
            config_path: เส้นทางไปยัง config.yaml
        """
        self.config_path = Path(config_path)
        self._config: Optional[Dict] = None
        self._parameters: Optional[Dict] = None

    def load_config(self, config_path: str = None) -> Dict:
        """
        โหลด configuration จากไฟล์ YAML

        Args:
            config_path: เส้นทาง (ใช้ค่า default ถ้าไม่ระบุ)

        Returns:
            dict ของ configuration
        """
        path = Path(config_path) if config_path else self.config_path

        if not path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ config: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._config = config
        logger.info(f"โหลด config จาก {path}")
        return config

    def load_parameters(self, parameters_path: str = "config/parameters.yaml") -> Dict:
        """
        โหลด parameter ranges จากไฟล์

        Args:
            parameters_path: เส้นทางไปยัง parameters.yaml

        Returns:
            dict ของ parameter ranges
        """
        path = Path(parameters_path)
        if not path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ parameters: {path}")

        with open(path, "r", encoding="utf-8") as f:
            parameters = yaml.safe_load(f)

        self._parameters = parameters
        logger.info(f"โหลด parameters จาก {path}")
        return parameters

    @property
    def config(self) -> Dict:
        """ดึง config (โหลดอัตโนมัติถ้ายังไม่โหลด)"""
        if self._config is None:
            self.load_config()
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        ดึงค่า config ด้วย dot notation

        Args:
            key_path: เส้นทาง key (เช่น 'strategy.classic_grid.num_grids')
            default: ค่า default ถ้าไม่พบ

        Returns:
            ค่า config
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_strategy_params(self, strategy_type: str = "adaptive_grid") -> Dict:
        """
        ดึง strategy parameters

        Args:
            strategy_type: 'classic_grid' หรือ 'adaptive_grid'

        Returns:
            dict ของ strategy parameters
        """
        params = self.get(f"strategy.{strategy_type}", {})
        logger.debug(f"Strategy params ({strategy_type}): {params}")
        return params

    def get_risk_params(self) -> Dict:
        """
        ดึง risk management parameters

        Returns:
            dict ของ risk parameters
        """
        return self.get("risk", {})

    def get_model_params(self, model_type: str = "regime_detector") -> Dict:
        """
        ดึง AI model parameters

        Args:
            model_type: 'regime_detector' หรือ 'volatility_predictor'

        Returns:
            dict ของ model parameters
        """
        return self.get(f"models.{model_type}", {})

    def get_data_params(self) -> Dict:
        """
        ดึง data parameters

        Returns:
            dict ของ data settings
        """
        return self.get("data", {})

    def get_backtest_params(self) -> Dict:
        """
        ดึง backtesting parameters

        Returns:
            dict ของ backtest settings
        """
        return self.get("backtesting", {})
