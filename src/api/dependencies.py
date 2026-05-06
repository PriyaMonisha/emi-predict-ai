# filename: src/api/dependencies.py
# purpose:  FastAPI dependencies — model state container and API key auth
# version:  1.0

import os
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, Request

from src.features.feature_engineering import FeatureEngineer


@dataclass
class ModelState:
    """Holds all ML objects loaded once at startup via lifespan."""
    clf: Any             # LightGBM champion classifier
    reg: Any             # XGBoost champion regressor
    fe:  FeatureEngineer # Feature engineer paired with champion run


def get_model_state(request: Request) -> ModelState:
    return request.app.state.models


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Validate X-API-Key header. Raises HTTP 401 on missing or wrong key."""
    expected = os.environ.get("API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
