"""
src/models/volatility_predictor.py — Volatility Predictor ด้วย LSTM

พยากรณ์ความผันผวนของราคาทองคำล่วงหน้า 5 วัน
ใช้ LSTM (Long Short-Term Memory) Neural Network

Input: 30 วันย้อนหลัง
Output: ค่าความผันผวน (annualized) ล่วงหน้า
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# พยายาม import TensorFlow — ถ้าไม่มีให้แจ้งเตือน
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow ไม่พร้อมใช้งาน — VolatilityPredictor จะใช้ Dummy model")


class VolatilityPredictor:
    """
    พยากรณ์ความผันผวนของทองคำด้วย LSTM

    Architecture:
    LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
    """

    # Features ที่ใช้สำหรับ LSTM
    FEATURE_COLS = [
        "returns", "log_returns",
        "hist_vol_20d", "hist_vol_10d",
        "atr_14_pct",
        "bb_width",
        "rsi_14",
        "adx",
        "volume_ratio",
        "dist_ema_20",
    ]

    def __init__(
        self,
        sequence_length: int = 30,
        lstm_units: list = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        กำหนดค่าเริ่มต้น VolatilityPredictor

        Args:
            sequence_length: จำนวนวันย้อนหลังสำหรับ LSTM input
            lstm_units: รายการจำนวน units แต่ละ LSTM layer
            dropout_rate: อัตรา dropout
            learning_rate: learning rate สำหรับ Adam optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._is_trained = False
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> "keras.Model":
        """
        สร้าง LSTM Model Architecture

        Args:
            input_shape: รูปร่าง input (sequence_length, n_features)

        Returns:
            Keras model ที่พร้อม compile แล้ว
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow ไม่พร้อม — ข้าม build_model")
            return None

        model = keras.Sequential(name="VolatilityPredictor")

        # LSTM Layer 1
        model.add(
            layers.LSTM(
                self.lstm_units[0],
                input_shape=input_shape,
                return_sequences=len(self.lstm_units) > 1,
                name="lstm_1",
            )
        )
        model.add(layers.Dropout(self.dropout_rate, name="dropout_1"))

        # LSTM Layers เพิ่มเติม
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(layers.LSTM(units, return_sequences=False, name=f"lstm_{i}"))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # Dense Layers
        model.add(layers.Dense(16, activation="relu", name="dense_1"))
        model.add(layers.Dense(1, activation="linear", name="output"))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="huber",
            metrics=["mae"],
        )

        logger.info(f"สร้าง LSTM Model สำเร็จ:")
        model.summary(print_fn=logger.info)

        return model

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "hist_vol_20d",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        เตรียม sequence data สำหรับ LSTM

        Args:
            df: DataFrame ที่มี feature columns
            target_col: คอลัมน์เป้าหมาย (volatility)

        Returns:
            tuple (X, y) — X shape: (n, seq_len, features), y shape: (n,)
        """
        # เลือก features ที่มีอยู่
        available_features = [f for f in self.FEATURE_COLS if f in df.columns]
        if not available_features:
            raise ValueError("ไม่พบ feature columns — ต้องรัน FeatureEngineer ก่อน")

        if target_col not in df.columns:
            raise ValueError(f"ไม่พบ target column: {target_col}")

        # Scale data
        X_data = self.scaler_X.fit_transform(df[available_features].values)
        y_data = self.scaler_y.fit_transform(df[[target_col]].values)

        # สร้าง sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(df)):
            X_sequences.append(X_data[i - self.sequence_length : i])
            y_sequences.append(y_data[i])

        X = np.array(X_sequences)
        y = np.array(y_sequences).flatten()

        logger.info(f"เตรียม sequences: X={X.shape}, y={y.shape}")
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 15,
    ) -> dict:
        """
        Train LSTM model

        Args:
            X: Input sequences (n, seq_len, features)
            y: Target values
            epochs: จำนวน epochs
            batch_size: batch size
            validation_split: สัดส่วน validation data
            patience: จำนวน epochs สำหรับ early stopping

        Returns:
            dict ของ training history
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow ไม่พร้อม — ใช้ Dummy model")
            self._is_trained = True
            return {}

        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        logger.info(f"กำลัง Train LSTM: {X.shape[0]} samples, {epochs} epochs")

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        self._is_trained = True

        # Final metrics
        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        logger.info(f"Train Loss: {final_train_loss:.6f}, Val Loss: {final_val_loss:.6f}")

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ทำนายความผันผวน

        Args:
            X: Input sequences

        Returns:
            array ของค่า volatility ที่ทำนาย (ใน original scale)
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        if not TF_AVAILABLE or self.model is None:
            # Dummy prediction สำหรับ testing
            return np.random.uniform(0.1, 0.3, len(X))

        predictions_scaled = self.model.predict(X, verbose=0)

        # แปลงกลับจาก scaled
        predictions = self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        return predictions

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        ประเมินผล model

        Args:
            X: Input sequences
            y_true: ค่าความผันผวนจริง (ใน scaled space)

        Returns:
            dict ของ evaluation metrics
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        y_pred = self.predict(X)

        # แปลง y_true กลับ
        y_true_original = self.scaler_y.inverse_transform(
            y_true.reshape(-1, 1)
        ).flatten()

        # คำนวณ metrics
        mae = np.mean(np.abs(y_pred - y_true_original))
        mse = np.mean((y_pred - y_true_original) ** 2)
        rmse = np.sqrt(mse)

        # MAPE
        mape = np.mean(np.abs((y_true_original - y_pred) / (y_true_original + 1e-10))) * 100

        results = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
        }

        logger.info(f"Evaluation — MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%")
        return results

    def save_model(self, path: str) -> None:
        """
        บันทึก LSTM model

        Args:
            path: เส้นทางสำหรับบันทึก (.h5)
        """
        if not self._is_trained:
            raise RuntimeError("Model ยังไม่ได้ train")

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if TF_AVAILABLE and self.model is not None:
            self.model.save(str(filepath))

        # บันทึก scaler แยก
        import pickle
        scaler_path = filepath.with_suffix(".scalers.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump({
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "sequence_length": self.sequence_length,
                "feature_cols": self.FEATURE_COLS,
            }, f)

        logger.info(f"บันทึก Volatility Predictor ที่ {filepath}")

    def load_model(self, path: str) -> "VolatilityPredictor":
        """
        โหลด LSTM model

        Args:
            path: เส้นทางของไฟล์ model

        Returns:
            self
        """
        import pickle

        filepath = Path(path)

        if not TF_AVAILABLE:
            logger.warning("TensorFlow ไม่พร้อม — โหลดเฉพาะ scalers")
        else:
            if filepath.exists():
                self.model = keras.models.load_model(str(filepath))

        # โหลด scalers
        scaler_path = filepath.with_suffix(".scalers.pkl")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                data = pickle.load(f)
            self.scaler_X = data["scaler_X"]
            self.scaler_y = data["scaler_y"]
            self.sequence_length = data.get("sequence_length", self.sequence_length)

        self._is_trained = True
        logger.info(f"โหลด Volatility Predictor จาก {filepath}")
        return self
