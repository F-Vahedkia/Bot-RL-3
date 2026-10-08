# f15_testcheck/unit/ts03_idx_fibo__1ratios_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_fibo__1ratios_1

from f02_utils.config_loader import ConfigLoader
from f02_utils.config_operations import _deep_get

# ------------------------------------------------------------
# نسبت‌های پیش‌فرض فیبوناچی
# ------------------------------------------------------------
cfg_all = ConfigLoader().get_all()

if cfg_all is not None:
    # _deep_get فقط dot-notation می‌پذیرد
    DEFAULT_RETR_RATIOS = _deep_get(cfg_all, "features.fibonacci.retracement_ratios")
    DEFAULT_EXT_RATIOS  = _deep_get(cfg_all, "features.fibonacci.extension_ratios")

print(DEFAULT_RETR_RATIOS)
print(DEFAULT_EXT_RATIOS)
