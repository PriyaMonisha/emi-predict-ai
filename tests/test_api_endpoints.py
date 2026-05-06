# filename: tests/test_api_endpoints.py
# purpose:  HTTP contract tests for FastAPI endpoints (lifespan mocked)
# version:  1.1

# main.py replaces sys.stdout/sys.stderr at import time (UTF-8 workaround).
# Importing it at module level breaks pytest's capture mechanism during
# collection. All src.api.main imports are deferred into fixtures/functions.

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_PRED_DF = pd.DataFrame({
    "clf_proba":     [0.9],
    "clf_label":     [1],
    "conf_zone":     ["auto_approve"],
    "predicted_emi": [8000.0],
})

_BATCH_PRED_DF = pd.DataFrame({
    "clf_proba":     [0.9, 0.2],
    "clf_label":     [1, 0],
    "conf_zone":     ["auto_approve", "auto_reject"],
    "predicted_emi": [8000.0, 3000.0],
})

_AUTH = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def _app():
    """Lazily import app — deferred so main.py's stdout patch runs after
    pytest has set up capture (no .buffer on captured stdout → no-op)."""
    from src.api.main import app
    return app


@pytest.fixture(scope="module")
def client(_app):
    """TestClient with joblib.load mocked so lifespan never hits disk."""
    _fe  = mock.MagicMock()
    _clf = mock.MagicMock(spec=["predict_proba", "predict"])
    _reg = mock.MagicMock(spec=["predict"])

    _clf.predict_proba.side_effect = (
        lambda X: np.column_stack([np.ones(len(X)) * 0.1, np.ones(len(X)) * 0.9])
    )
    _reg.predict.side_effect = lambda X: np.ones(len(X)) * 8000.0
    _fe.transform.side_effect = lambda df: df

    with mock.patch("joblib.load", side_effect=[_fe, _clf, _reg]):
        with TestClient(_app, raise_server_exceptions=True) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    with mock.patch("src.api.main.health_check", return_value=True):
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["redis_ok"], bool)
    assert isinstance(body["models_loaded"], bool)


# ── /metrics ───────────────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_200(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"http_requests_total" in r.content


# ── /predict ───────────────────────────────────────────────────────────────────

def test_predict_ok_returns_200(client, valid_payload):
    with mock.patch("src.api.main._run_inference", return_value=(_PRED_DF, False)):
        r = client.post("/predict", json=valid_payload, headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "clf_proba" in body
    assert "conf_zone" in body
    assert body["clf_label"] in (0, 1)
    assert isinstance(body["cache_hit"], bool)
    assert isinstance(body["latency_ms"], float)


def test_predict_missing_api_key_returns_422(client, valid_payload):
    # FastAPI returns 422 when a required Header field is absent
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422


def test_predict_wrong_api_key_returns_401(client, valid_payload):
    with mock.patch("src.api.main._run_inference", return_value=(_PRED_DF, False)):
        r = client.post("/predict", json=valid_payload,
                        headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


# ── /predict/batch ─────────────────────────────────────────────────────────────

def test_batch_predict_ok_returns_200(client, _app, valid_payload):
    payload = {"customers": [valid_payload, valid_payload]}
    with mock.patch("src.api.main.preprocess_for_inference",
                    return_value=pd.DataFrame({"dummy": [1, 2]})), \
         mock.patch("src.api.main.predict_from_preloaded",
                    return_value=_BATCH_PRED_DF):
        _app.state.models.fe.transform.side_effect = lambda df: df
        r = client.post("/predict/batch", json=payload, headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["total_scored"] == 2
    assert len(body["predictions"]) == 2
    assert "auto_approve" in body
    assert "auto_reject" in body
