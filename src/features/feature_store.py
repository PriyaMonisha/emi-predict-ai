# filename: src/features/feature_store.py
# purpose:  Section 8 — Redis-backed feature cache (cache-aside pattern)
# version:  1.1

import os
import json
import hashlib
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)

_KEY_PREFIX  = "emi:features"


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars to Python-native types for JSON serialization."""
    def default(self, o):
        if hasattr(o, "item"):          # numpy scalar → Python scalar
            return o.item()
        if hasattr(o, "tolist"):        # numpy array → list
            return o.tolist()
        return super().default(o)
_DEFAULT_TTL = 86400   # 24 hours — infrastructure.md locked
_HASH_SLICE  = 16      # first 16 hex chars of SHA-256 (64-bit space, no collision risk at 17k rows)


def get_connection() -> redis.Redis:
    """
    Build a Redis client from env vars.

    REDIS_HOST  default localhost
    REDIS_PORT  default 6379
    REDIS_DB    default 0

    Returns a lazy client (connects on first command, not here).
    """
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=int(os.environ.get("REDIS_DB", 0)),
        decode_responses=True,   # always str, never bytes — required for json.loads
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def _make_key(customer_id: str) -> str:
    """Hash customer_id to a Redis key. Never stores plaintext PII."""
    hashed = hashlib.sha256(customer_id.encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}:{hashed[:_HASH_SLICE]}"


def write_features(
    customer_id: str,
    features: dict,
    ttl: int = _DEFAULT_TTL,
) -> bool:
    """
    Serialize and cache one customer's post-engineering feature vector.

    Returns True on success, False on any error (cache is non-blocking).
    Uses SETEX to atomically set value and expiry in one command.
    """
    key = _make_key(customer_id)
    try:
        r = get_connection()
        r.setex(name=key, time=ttl, value=json.dumps(features, cls=_NumpyEncoder))
        logger.debug(f"[feature_store] write {key}  ttl={ttl}s")
        return True
    except (redis.RedisError, TypeError, ValueError) as exc:
        logger.warning(f"[feature_store] write_features failed for {key}: {exc}")
        return False


def read_features(customer_id: str) -> Optional[dict]:
    """
    Read cached feature dict for one customer.

    Returns dict on hit, None on miss or any error.
    """
    key = _make_key(customer_id)
    try:
        r = get_connection()
        raw = r.get(key)
        if raw is None:
            logger.debug(f"[feature_store] miss: {key}")
            return None
        logger.debug(f"[feature_store] hit: {key}")
        return json.loads(raw)
    except (redis.RedisError, json.JSONDecodeError) as exc:
        logger.warning(f"[feature_store] read_features failed for {key}: {exc}")
        return None


def invalidate(customer_id: str) -> bool:
    """Delete cached features for one customer (call after raw data update)."""
    key = _make_key(customer_id)
    try:
        r = get_connection()
        deleted = r.delete(key)
        logger.info(f"[feature_store] invalidate {key}  deleted={deleted}")
        return bool(deleted)
    except redis.RedisError as exc:
        logger.warning(f"[feature_store] invalidate failed for {key}: {exc}")
        return False


def batch_write(
    records: list[dict],
    id_col: str,
    feature_cols: list[str],
    ttl: int = _DEFAULT_TTL,
) -> dict:
    """
    Write post-feature-engineering vectors for a batch of customers.

    Uses Redis pipeline (single round-trip) for efficiency.
    transaction=False avoids MULTI/EXEC overhead for independent key writes.

    Args:
        records      : list of row dicts — typically df.to_dict("records")
        id_col       : column holding the raw customer identifier
        feature_cols : columns to cache (the engineered feature names)
        ttl          : TTL per key in seconds

    Returns:
        {"written": N, "errors": M}
    """
    written = 0
    errors  = 0
    r = get_connection()
    pipe = r.pipeline(transaction=False)

    keys_scheduled = 0
    for row in records:
        cid = str(row.get(id_col, "")).strip()
        if not cid:
            errors += 1
            continue
        payload = json.dumps({k: row[k] for k in feature_cols if k in row}, cls=_NumpyEncoder)
        pipe.setex(name=_make_key(cid), time=ttl, value=payload)
        keys_scheduled += 1

    if keys_scheduled:
        try:
            results = pipe.execute()
            for ok in results:
                if ok:
                    written += 1
                else:
                    errors += 1
        except redis.RedisError as exc:
            logger.error(f"[feature_store] batch_write pipeline error: {exc}")
            errors += keys_scheduled

    logger.info(
        f"[feature_store] batch_write: {written} written, {errors} errors "
        f"({len(records)} total records)"
    )
    return {"written": written, "errors": errors}


def health_check() -> bool:
    """Ping Redis. Returns True if reachable, False otherwise."""
    try:
        r = get_connection()
        return r.ping()
    except redis.RedisError as exc:
        logger.warning(f"[feature_store] health_check failed: {exc}")
        return False
