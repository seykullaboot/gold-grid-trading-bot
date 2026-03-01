"""
tests/test_models.py — Unit Tests สำหรับ AI Models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_detector import MarketRegimeDetector, REGIME_LABELS
from src.models.volatility_predictor import VolatilityPredictor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_features_df():
    """สร้าง DataFrame ที่มี feature columns สำหรับทดสอบ"""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    df = pd.DataFrame(
        {
            "Close": 150 + np.cumsum(np.random.randn(n) * 1.5),
            "adx": np.random.uniform(10, 50, n),
            "adx_pos": np.random.uniform(10, 30, n),
            "adx_neg": np.random.uniform(10, 30, n),
            "atr_14_pct": np.random.uniform(0.005, 0.03, n),
            "hist_vol_20d": np.random.uniform(0.08, 0.40, n),
            "bb_width": np.random.uniform(0.01, 0.10, n),
            "vol_ratio": np.random.uniform(0.5, 2.0, n),
            "rsi_14": np.random.uniform(20, 80, n),
            "macd_hist": np.random.randn(n) * 0.5,
            "dist_ema_20": np.random.uniform(-0.05, 0.05, n),
            "dist_ema_50": np.random.uniform(-0.08, 0.08, n),
            "roc_10": np.random.uniform(-5, 5, n),
            "roc_20": np.random.uniform(-8, 8, n),
            "stoch_k": np.random.uniform(0, 100, n),
            "stoch_d": np.random.uniform(0, 100, n),
            "returns": np.random.randn(n) * 0.01,
            "log_returns": np.random.randn(n) * 0.01,
            "hist_vol_10d": np.random.uniform(0.08, 0.40, n),
            "volume_ratio": np.random.uniform(0.5, 3.0, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def trained_regime_detector(sample_features_df):
    """สร้างและ train Regime Detector"""
    detector = MarketRegimeDetector(n_estimators=50, random_state=42)
    X, y = detector.prepare_features(sample_features_df)
    assert y is not None
    detector.train(X, y)
    return detector


# ============================================================
# Tests: MarketRegimeDetector
# ============================================================

class TestMarketRegimeDetector:
    """ทดสอบ MarketRegimeDetector"""

    def test_init(self):
        """ทดสอบ initialization"""
        detector = MarketRegimeDetector(n_estimators=100)
        assert not detector._is_trained
        assert detector.model.n_estimators == 100

    def test_prepare_features_returns_tuple(self, sample_features_df):
        """ทดสอบ prepare_features คืนค่าถูกต้อง"""
        detector = MarketRegimeDetector()
        X, y = detector.prepare_features(sample_features_df)
        assert X is not None
        assert X.ndim == 2
        assert X.shape[0] == len(sample_features_df)

    def test_prepare_features_creates_labels(self, sample_features_df):
        """ทดสอบว่า prepare_features สร้าง labels ได้"""
        detector = MarketRegimeDetector()
        X, y = detector.prepare_features(sample_features_df)
        assert y is not None
        assert len(y) == len(sample_features_df)
        assert set(y).issubset({0, 1, 2})

    def test_train_sets_trained_flag(self, sample_features_df):
        """ทดสอบว่า train() set _is_trained = True"""
        detector = MarketRegimeDetector(n_estimators=50)
        X, y = detector.prepare_features(sample_features_df)
        detector.train(X, y)
        assert detector._is_trained

    def test_predict_returns_valid_regimes(self, trained_regime_detector, sample_features_df):
        """ทดสอบว่า predict คืนค่า regime ที่ถูกต้อง"""
        X, _ = trained_regime_detector.prepare_features(sample_features_df)
        predictions = trained_regime_detector.predict(X)
        assert len(predictions) == len(sample_features_df)
        assert set(predictions).issubset({0, 1, 2})

    def test_predict_before_train_raises(self, sample_features_df):
        """ทดสอบว่า predict ก่อน train ต้อง raise error"""
        detector = MarketRegimeDetector()
        X, _ = detector.prepare_features(sample_features_df)
        with pytest.raises(RuntimeError):
            detector.predict(X)

    def test_predict_proba_shape(self, trained_regime_detector, sample_features_df):
        """ทดสอบ shape ของ predict_proba"""
        X, _ = trained_regime_detector.prepare_features(sample_features_df)
        proba = trained_regime_detector.predict_proba(X)
        assert proba.shape == (len(sample_features_df), 3)
        # ผลรวมแต่ละแถวต้องเป็น 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate_returns_metrics(self, trained_regime_detector, sample_features_df):
        """ทดสอบว่า evaluate คืนค่า metrics"""
        X, y = trained_regime_detector.prepare_features(sample_features_df)
        results = trained_regime_detector.evaluate(X, y)
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_save_and_load_model(self, trained_regime_detector, tmp_path):
        """ทดสอบบันทึกและโหลด model"""
        save_path = str(tmp_path / "test_regime_model.pkl")
        trained_regime_detector.save_model(save_path)
        assert Path(save_path).exists()

        # โหลด model ใหม่
        new_detector = MarketRegimeDetector()
        new_detector.load_model(save_path)
        assert new_detector._is_trained

    def test_predict_regime_series(self, trained_regime_detector, sample_features_df):
        """ทดสอบ predict_regime_series คืนค่า Series"""
        series = trained_regime_detector.predict_regime_series(sample_features_df)
        assert isinstance(series, pd.Series)
        assert len(series) == len(sample_features_df)
        assert series.index.equals(sample_features_df.index)
        # ค่าต้องเป็น string labels
        assert series.isin(["trending", "ranging", "volatile"]).all()

    def test_feature_importances_set_after_training(self, sample_features_df):
        """ทดสอบว่า feature_importances_ ถูก set หลัง train"""
        detector = MarketRegimeDetector(n_estimators=50)
        X, y = detector.prepare_features(sample_features_df)
        detector.train(X, y)
        assert detector.feature_importances_ is not None


# ============================================================
# Tests: VolatilityPredictor
# ============================================================

class TestVolatilityPredictor:
    """ทดสอบ VolatilityPredictor"""

    def test_init(self):
        """ทดสอบ initialization"""
        predictor = VolatilityPredictor(sequence_length=20)
        assert predictor.sequence_length == 20
        assert not predictor._is_trained

    def test_prepare_sequences_shape(self, sample_features_df):
        """ทดสอบ shape ของ sequences"""
        predictor = VolatilityPredictor(sequence_length=10)
        X, y = predictor.prepare_sequences(sample_features_df, target_col="hist_vol_20d")

        assert X.ndim == 3
        assert X.shape[1] == 10  # sequence length
        assert len(X) == len(y)
        assert len(X) == len(sample_features_df) - 10

    def test_prepare_sequences_missing_target(self, sample_features_df):
        """ทดสอบ error เมื่อไม่มี target column"""
        predictor = VolatilityPredictor()
        with pytest.raises(ValueError):
            predictor.prepare_sequences(sample_features_df, target_col="nonexistent_col")

    def test_predict_before_train_raises(self, sample_features_df):
        """ทดสอบว่า predict ก่อน train ต้อง raise error"""
        predictor = VolatilityPredictor()
        X, _ = predictor.prepare_sequences(sample_features_df)
        with pytest.raises(RuntimeError):
            predictor.predict(X)

    def test_build_model_returns_model_if_tf_available(self, sample_features_df):
        """ทดสอบ build_model สร้าง model"""
        from src.models.volatility_predictor import TF_AVAILABLE
        predictor = VolatilityPredictor(sequence_length=10)

        if TF_AVAILABLE:
            model = predictor.build_model(input_shape=(10, 5))
            assert model is not None
        else:
            # ถ้าไม่มี TF ควรคืน None
            model = predictor.build_model(input_shape=(10, 5))
            assert model is None
