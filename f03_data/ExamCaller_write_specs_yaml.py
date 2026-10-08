# f03_data/_ExamCaller_write_specs_yaml
# finished at 1404/09/19-21:51

'''
این برنامه یک نمونه است برای نمایش اینکه چگونه
تابع write_specs_yaml را از ماژول f03_data.symbol_specs_snapshot فراخوانی کنیم
تابع write_specs_yaml در واقع تابع اصلی و ماحصل فایل symbol_specs_snapshot.py است

Run:
python -m f03_data.ExamCaller_write_specs_yaml
'''

from f03_data.symbol_specs_snapshot import write_specs_yaml
import argparse

config_path = "f01_config/config.yaml"
symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
# overlay_path  = "f01_config/symbol_specs__NN__.yaml"
overlay_path = None
snapshot_path = "f03_data/specs_snapshot.yaml"

args = argparse.Namespace(
    config=str(config_path),
    symbols=list(symbols) if symbols is not None else None,
    overlay_out=str(overlay_path) if overlay_path is not None else None,
    snapshot_out=str(snapshot_path) if snapshot_path is not None else None
)
write_specs_yaml(args)