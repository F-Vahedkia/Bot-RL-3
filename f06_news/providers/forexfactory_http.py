# -*- coding: utf-8 -*-
# f06_news/providers/forexfactory_http.py
# Status in (Bot-RL-3): Completed

"""
ForexFactory HTTP provider (download CSV) — messages in English; Persian comments.

Notes:
- We do not hardcode FF endpoints (they change). Pass URLs via CLI/config.
- Will download any given URL(s) (CSV-like) with retry/timeout and save under news/raw/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import logging
import pathlib
import time

# Persian: تلاش برای استفاده از requests؛ در صورت عدم نصب، از urllib استفاده می‌کنیم
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request  # type: ignore
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _hash_url(u: str) -> str:
    return hashlib.sha256(u.encode("utf-8")).hexdigest()[:10]

@dataclass
class ForexFactoryHTTPProvider:
    """
    Persian: Provider دانلود CSV از URLهای داده‌شده.
    """
    urls: List[str]
    dest_dir: pathlib.Path
    timeout_sec: float = 10.0
    retries: int = 2
    sleep_backoff_sec: float = 1.5

    saved_paths: List[pathlib.Path] = field(default_factory=list)

    def _download_one(self, url: str) -> Optional[pathlib.Path]:
        _ensure_dir(self.dest_dir)
        suffix = ".csv"  # Persian: فرض CSV؛ اگر لازم شد از هدر Content-Type استنباط کنید
        name = f"ff_{_hash_url(url)}{suffix}"
        out = self.dest_dir / name

        # Persian: اگر قبلاً دانلود شده، همان را برگردانیم
        if out.exists() and out.stat().st_size > 0:
            logger.info("File already exists, skipping download: %s", out)
            return out

        err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                logger.info("Downloading: %s (attempt %d)", url, attempt + 1)
                if _HAS_REQUESTS:
                    r = requests.get(url, timeout=self.timeout_sec)
                    r.raise_for_status()
                    data = r.content
                else:
                    with urllib.request.urlopen(url, timeout=self.timeout_sec) as resp:
                        data = resp.read()

                out.write_bytes(data)
                logger.info("Saved: %s (%d bytes)", out, len(data))
                return out
            except Exception as ex:
                err = ex
                logger.warning("Download failed (%s). Retrying...", ex)
                time.sleep(self.sleep_backoff_sec * (attempt + 1))

        logger.error("Failed to download after retries: %s | last_error=%s", url, err)
        return None

    def fetch(self) -> List[pathlib.Path]:
        """English: Download all URLs and return list of saved local paths."""
        self.saved_paths.clear()
        for u in self.urls:
            p = self._download_one(u)
            if p:
                self.saved_paths.append(p)
        return list(self.saved_paths)
