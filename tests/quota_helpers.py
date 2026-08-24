"""Helpers to pre-fill the per-test quota store."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def prefill_quota(bucket: str, count: int) -> None:
    """Write `count` used calls into this test's quota store file."""
    path = Path(os.environ["QUOTA_STORE_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    monthly = data.setdefault("monthly", {})
    entry = monthly.setdefault(bucket, {"period": "", "count": 0})
    entry["period"] = datetime.now().strftime("%Y-%m")
    entry["count"] = count
    path.write_text(json.dumps(data), encoding="utf-8")


def store_path() -> Path:
    return Path(os.environ["QUOTA_STORE_PATH"])
