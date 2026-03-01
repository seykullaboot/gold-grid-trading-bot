"""
src/models/regime_detector.py — Market Regime Detector ด้วย Random Forest

จำแนก Market Regime เป็น 3 ประเภท:
- 0: Trending (ตลาดมีแนวโน้มชัดเจน)
- 1: Ranging (ตลาดไซด์เวย์)
- 2: Volatile (ตลาดผันผวนสูง)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ป้าย Market Regime
REGIME_LABELS = {0: "trending", 1: "ranging", 2: "volatile"}
REGIME_COLORS = {0: "blue", 1: "green", 2: "red"}


class MarketRegimeDetector:
    """
    จำแนก Market Regime ด้วย Random Forest Classifier

    ใช้ Technical Indicators เป็น Features:
    - ADX (ความแรงของ trend)
    - ATR (ความผันผวน)
    - Bollinger Band Width
    - RSI
    - Volume ratio
    """

    # Features ที่ใช้สำหรับ prediction
    FEATURE_COLS = [
        "adx", "adx_pos", "adx_neg",
        "atr_14_pct", "hist_vol_20d",
        "bb_width", "vol_ratio",
        "rsi_14",
        "macd_hist",
        "dist_ema_20", "dist_ema_50",
        "roc_10", "roc_20",
        "stoch_k", "stoch_d",
    ]

    def __init__(self, n_estimators: int = 200, max_depth: int = 10, random_state: int = 42):
        """
        กำหนดค่าเริ่มต้น MarketRegimeDetector

        Args:
            n_estimators: จำนวน decision trees
            max_depth: ความลึกสูงสุดของ tree
            random_state: seed สำหรับ reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self._is_trained = False
        self.feature_importances_ = None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        เตรียม features สำหรับ training หรือ prediction

        Args:
            df: DataFrame ที่มี technical indicators

        Returns:
            tuple (X, y) — X=features array, y=labels (None ถ้าไม่มีคอลัมน์ label)
        """
        # ตรวจสอบ features ที่มีอยู่
        available_features = [f for f in self.FEATURE_COLS if f in df.columns]
        missing = [f for f in self.FEATURE_COLS if f not in df.columns]

        if missing:
            logger.warning(f"Features ที่ไม่พบ: {missing}")

        if not available_features:
            raise ValueError("ไม่พบ features ที่จำเป็น — ต้องรัน FeatureEngineer ก่อน")

        X = df[available_features].values

        # สร้าง labels ถ้ายังไม่มี
        y = None
        if "regime" in df.columns:
            y = df["regime"].values
        elif all(c in df.columns for c in ["adx", "hist_vol_20d"]):
            y = self._create_regime_labels(df)

        return X, y

    def _create_regime_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        สร้าง regime labels จาก technical indicators อัตโนมัติ

        Rules:
        - ADX > 25 AND vol ต่ำ  → Trending (0)
        - ADX < 20               → Ranging (1)
        - vol สูงมาก             → Volatile (2)

        Args:
            df: DataFrame ที่มี ADX และ volatility

        Returns:
            array of regime labels
        """
        labels = np.ones(len(df), dtype=int)  # default: ranging

        if "adx" in df.columns and "hist_vol_20d" in df.columns:
            vol_q75 = df["hist_vol_20d"].quantile(0.75)
            vol_q90 = df["hist_vol_20d"].quantile(0.90)

            # Trending: ADX สูง + volatility ปกติ
            trending_mask = (df["adx"] > 25) & (df["hist_vol_20d"] < vol_q75)
            labels[trending_mask.values] = 0

            # Volatile: volatility สูงมาก
            volatile_mask = df["hist_vol_20d"] > vol_q90
            labels[volatile_mask.values] = 2

        return labels

    def train(self, X: np.ndarray, y: np.ndarray) -> "MarketRegimeDetector":
        """
        Train Random Forest model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)

        Returns:
            self (สำหรับ method chaining)
        """
        logger.info(f"กำลัง Train Regime Detector: {X.shape[0]} samples, {X.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self._is_trained = True

        # บันทึก feature importances
        feature_cols = [f for f in self.FEATURE_COLS if f in self.FEATURE_COLS]
        self.feature_importances_ = dict(
            zip(feature_cols[:X.shape[1]], self.model.feature_importances_)
        )

        # Training accuracy
        train_acc = self.model.score(X_scaled, y)
        logger.info(f"Training Accuracy: {train_acc:.4f}")

        # Distribution of regimes
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  Regime {REGIME_LABELS.get(u, u)}: {c} samples ({c/len(y):.1%})")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ทำนาย Market Regime

        Args:
            X: Feature matrix

        Returns:
            array of predicted regimes (0=trending, 1=ranging, 2=volatile)
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train — ต้องเรียก train() ก่อน")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        ทำนายความน่าจะเป็นของแต่ละ Regime

        Args:
            X: Feature matrix

        Returns:
            array (n_samples, 3) ของ probabilities
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        ประเมินผล model

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            dict ของ metrics
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        # คำนวณ metrics
        accuracy = self.model.score(X_scaled, y)
        report = classification_report(
            y, y_pred,
            target_names=[REGIME_LABELS[i] for i in sorted(REGIME_LABELS.keys())],
            output_dict=True,
        )
        cm = confusion_matrix(y, y_pred)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"\n{classification_report(y, y_pred, target_names=[REGIME_LABELS[i] for i in sorted(REGIME_LABELS.keys())])}")

        return results

    def save_model(self, path: str) -> None:
        """
        บันทึก model ลงไฟล์

        Args:
            path: เส้นทางสำหรับบันทึก
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.FEATURE_COLS,
            "feature_importances": self.feature_importances_,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"บันทึก Regime Detector ที่ {filepath}")

    def load_model(self, path: str) -> "MarketRegimeDetector":
        """
        โหลด model จากไฟล์

        Args:
            path: เส้นทางของไฟล์ model

        Returns:
            self
        """
        filepath = Path(path)

        if not filepath.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ model: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_importances_ = model_data.get("feature_importances")
        self._is_trained = True

        logger.info(f"โหลด Regime Detector จาก {filepath}")
        return self

    def predict_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        ทำนาย regime สำหรับ DataFrame ทั้งหมดและคืนเป็น Series

        Args:
            df: DataFrame ที่มี feature columns

        Returns:
            Series ของ regime labels
        """
        X, _ = self.prepare_features(df)
        predictions = self.predict(X)

        # แปลงเป็น string labels
        regime_series = pd.Series(
            [REGIME_LABELS[p] for p in predictions],
            index=df.index,
            name="regime",
        )
        return regime_series
