# -*- coding: utf-8 -*-
"""تست: حضور اندیکاتورهای فیبوناچی در REGISTRY"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from f04_features.OLD.feature_registry import REGISTRY
exp = ["golden_zone","fib_cluster","fib_ext_targets","fibo_features_full","levels_from_legs","select_legs_from_swings"]
keys = sorted(REGISTRY.keys())
miss = [k for k in exp if k not in keys]
print("FIBO in REGISTRY:", sorted([k for k in keys if k.startswith("fib")]+[k for k in exp if k in keys]))
print("MISSING:", miss)
import sys; sys.exit(1 if miss else 0)
