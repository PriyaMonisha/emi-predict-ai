# filename: tests/test_feature_store.py
# purpose:  Unit tests for src/features/feature_store.py (Redis mocked)
# version:  1.0

import json
from unittest import mock

import pytest
import redis

import src.features.feature_store as fs


def _mock_redis() -> mock.MagicMock:
    """Return a MagicMock that looks like a connected Redis client."""
    r = mock.MagicMock(spec=redis.Redis)
    r.ping.return_value = True
    return r


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_write_features_returns_true():
    r = _mock_redis()
    r.setex.return_value = True
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.write_features("cust_001", {"score": 750, "salary": 60000})
    assert result is True
    r.setex.assert_called_once()


def test_write_features_returns_false_on_redis_error():
    r = _mock_redis()
    r.setex.side_effect = redis.RedisError("connection refused")
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.write_features("cust_001", {"score": 750})
    assert result is False


def test_read_features_cache_hit():
    payload = {"score": 750, "salary": 60000}
    r = _mock_redis()
    r.get.return_value = json.dumps(payload)
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.read_features("cust_001")
    assert result == payload


def test_read_features_cache_miss():
    r = _mock_redis()
    r.get.return_value = None
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.read_features("cust_001")
    assert result is None


def test_invalidate_returns_true_when_key_deleted():
    r = _mock_redis()
    r.delete.return_value = 1   # Redis returns count of deleted keys
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.invalidate("cust_001")
    assert result is True


def test_batch_write_counts_written():
    r = _mock_redis()
    pipe = mock.MagicMock()
    pipe.execute.return_value = [True, True, True]
    r.pipeline.return_value = pipe
    records = [
        {"customer_id": f"c{i}", "score": 700 + i}
        for i in range(3)
    ]
    with mock.patch.object(fs, "get_connection", return_value=r):
        result = fs.batch_write(records, id_col="customer_id",
                                feature_cols=["score"])
    assert result["written"] == 3
    assert result["errors"] == 0


def test_health_check_returns_true_when_redis_up():
    r = _mock_redis()
    r.ping.return_value = True
    with mock.patch.object(fs, "get_connection", return_value=r):
        assert fs.health_check() is True


def test_health_check_returns_false_on_error():
    r = _mock_redis()
    r.ping.side_effect = redis.RedisError("timeout")
    with mock.patch.object(fs, "get_connection", return_value=r):
        assert fs.health_check() is False
