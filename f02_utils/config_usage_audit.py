# f02_utils/config_usage_audit.py
# Run: python -m f02_utils.config_usage_audit

from __future__ import annotations
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
}

RISK_KEYS = [
    "risk",
    "risk_per_trade",
    "max_daily_loss_pct",
    "max_total_drawdown_pct",
    "position_sizing",
    "method",
    "min_lot",
    "lot_step",
    "max_lot",
    "use_atr_for_sl",
    "sl_atr_mult",
    "tp_atr_mult",
    "breakeven",
    "at_r_multiple",
    "trailing",
    "type",
    "atr_period",
    "atr_mult",
    "position_limits",
    "max_open_positions",
    "max_positions_per_symbol",
    "cooldown_minutes_after_stop",
]


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def scan_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        for key in RISK_KEYS:
            if key in line:
                print(
                    f"{path.relative_to(PROJECT_ROOT)}"
                    f":{line_number}"
                    f" | {key}"
                    f" | {line.strip()}"
                )


def main() -> None:
    for path in PROJECT_ROOT.rglob("*.py"):
        if is_excluded(path):
            continue

        scan_file(path)


if __name__ == "__main__":
    main()