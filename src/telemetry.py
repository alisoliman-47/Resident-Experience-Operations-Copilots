from __future__ import annotations

import json
import logging
import time
from typing import Any


logger = logging.getLogger("resident_ops")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event_name: str, **fields: Any) -> None:
    payload = {"event": event_name, "ts": int(time.time()), **fields}
    logger.info(json.dumps(payload, default=str))
