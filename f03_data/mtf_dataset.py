# f03_data/mtf_dataset.py (6)
#
# Date Reviewed: 1405/06/25
# =======================================================================================
""" ---> Docstring
کانتینر داده چندتایم‌فریمی MTFDataset برای لایه f03_data.

MTFDataset داده‌های یک symbol را در چند timeframe نگهداری می‌کند و برای هر timeframe یک
DataFrame مستقل در frames دارد. این کلاس عمداً هیچ هم‌ترازسازی زمانی، resampling یا merge
بین timeframeها انجام نمی‌دهد.

قرارداد داده:
    - symbol و base_tf در سطح dataset نگهداری می‌شوند.
    - frames یک Dict[str, pd.DataFrame] است که کلیدهای timeframe در آن uppercase هستند.
    - add(timeframe, df) یک frame را ثبت یا جایگزین می‌کند و نام timeframe را uppercase می‌کند.
    - get(timeframe) و __getitem__(timeframe) frame متناظر را برمی‌گردانند.
    - replace(timeframe, df) فقط frame موجود را جایگزین می‌کند و در نبود آن KeyError می‌دهد.
    - copy() یک dataset مستقل با copy از DataFrameهای frames ایجاد می‌کند.
    - timeframes فهرست کلیدهای فعلی frames را برمی‌گرداند.
    - apply() یک تابع را روی هر DataFrame اعمال می‌کند؛ apply_each() علاوه بر DataFrame نام
        timeframe را نیز به تابع می‌دهد.

قانون معماری مهم:
هر DataFrame متعلق به یک timeframe مستقل است و MTFDataset داده‌ها را به شبکه زمانی
base_tf تبدیل یا روی آن merge نمی‌کند. مصرف‌کننده باید این استقلال را حفظ کند مگر اینکه
صراحتاً مسئولیت دیگری در قرارداد لایه بعدی تعریف شده باشد.
"""
# =======================================================================================

# =======================================================================================
# Imports
# =======================================================================================
# f03_data/mtf_dataset.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from datetime import datetime
from pathlib import Path
import pandas as pd

# =======================================================================================
# MAIN CLASS
# =======================================================================================
@dataclass
class MTFDataset:
    """
    نگهدارنده‌ی دیتافریم‌های مربوط به یک نماد در چند تایم‌فریم مختلف.
    هیچ همترازسازی انجام نمی‌دهد.
    هر تایم‌فریم دیتافریم مستقل خودش را دارد.
    """
    # ===============================================================
    # Class Variables
    # ===============================================================
    symbol: str
    base_tf: str
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ===============================================================
    # add
    # =============================================================== 1
    def add(self, timeframe: str, df: pd.DataFrame) -> None:
        self.frames[timeframe.upper()] = df

    # ===============================================================
    # get
    # =============================================================== 2
    def get(self, timeframe: str) -> pd.DataFrame:
        return self.frames[timeframe.upper()]

    # ===============================================================
    # __getitem__
    # =============================================================== 3
    def __getitem__(self, timeframe: str) -> pd.DataFrame:
        """
        توسط این متد عبارت df=dataset["M5"] همانند عبارت df=dataset.get("M5") عمل میکند
        """
        return self.get(timeframe)

    # ===============================================================
    # replace
    # =============================================================== 4
    def replace(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        جایگزین کردن دیتافریم یک تایم‌فریم.

        اگر تایم‌فریم قبلاً وجود نداشته باشد، خطا می‌دهد تا
        جایگزینی اشتباه با add اشتباه گرفته نشود.
        """
        timeframe = timeframe.upper()

        if timeframe not in self.frames:
            raise KeyError(f"Timeframe '{timeframe}' does not exist.")

        self.frames[timeframe] = df

    # ===============================================================
    # copy
    # =============================================================== 5
    def copy(self) -> "MTFDataset":     # Forward Reference
        out = MTFDataset(
            symbol=self.symbol,
            base_tf=self.base_tf,
        )
        for tf, df in self.frames.items():
            out.frames[tf] = df.copy()
        return out

    # ===============================================================
    # timeframes
    # =============================================================== 6
    @property
    def timeframes(self) -> List[str]:
        return list(self.frames.keys())
    
    # ===============================================================
    # apply
    # =============================================================== 7
    def apply(self,
        func: Callable[..., pd.DataFrame],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        روشهای  استفاده:
        dataset.apply(_add_time_features_to_df, self.cfg)
        dataset.apply(compute_indicators)
        dataset.apply(build_price_action)
        dataset.apply(build_patterns)
        """
        for tf in self.frames:
            self.frames[tf] = func(
                self.frames[tf],
                *args,
                **kwargs
            )

    # ===============================================================
    # apply_each
    # =============================================================== 8
    def apply_each(self,
        func: Callable[..., pd.DataFrame],
        *args: Any,
        **kwargs: Any
    ) -> "MTFDataset":
        """
        اعمال یک تابع روی تمام تایم‌فریم‌ها.
        تابع باید دیتافریم و نام تایم‌فریم را دریافت کرده و یک دیتافریم جدید برگرداند.
        امضا:
            func(df, timeframe, *args, **kwargs) -> pd.DataFrame
        مثال:
            dataset.apply_each(
                _finalize_dataframe,
                self.cfg,
                symbol,
            )
        """
        for tf in self.frames:
            self.frames[tf] = func(
                self.frames[tf],
                tf,
                *args,
                **kwargs
            )
        return self

    # ===============================================================
    # _print
    # =============================================================== 9
    def _print(
        self,
        file_path: str,
        n_rows: int = 5,
    ) -> None:
        """
        ذخیره‌ی محتویات MTFDataset در یک فایل Markdown.

        Parameters
        ----------
        file_path : str
            مسیر فایل Markdown خروجی.

        n_rows : int, default=5
            تعداد سطرهایی که از ابتدا و انتهای هر DataFrame نمایش داده می‌شود.

            n_rows > 0:
                n_rows سطر اول + n_rows سطر آخر

            n_rows == 0:
                تمام سطرهای DataFrame

        Notes
        -----
        برای هر تایم‌فریم یک بخش <details> ایجاد می‌شود تا
        در VSCode قابل Collapse / Uncollapse باشد.

        ستون‌های نمایشی:
            timestamp, open, high, low, close, volume, spread
        """
        if n_rows < 0:
            raise ValueError("n_rows must be >= 0.")
        display_columns = ["open", "high", "low", "close", "volume", "spread"]
        lines: list[str] = []

        # -----------------------------------------------------------
        # Dataset information
        # -----------------------------------------------------------
        lines.append(f"# MTFDataset — {self.symbol}")
        lines.append("")
        lines.append(f"- **Symbol:** `{self.symbol}`")
        lines.append(f"- **Base timeframe:** `{self.base_tf}`")
        lines.append(f"- **Timeframes:** `{', '.join(self.timeframes)}`")
        lines.append("")

        # -----------------------------------------------------------
        # Each timeframe
        # -----------------------------------------------------------
        for tf, df in self.frames.items():

            # -----------------------------------
            # 1- Select display columns that actually exist
            # -----------------------------------
            columns = [
                col
                for col in display_columns
                if col in df.columns
            ]
            missing_columns = [
                col
                for col in display_columns
                if col not in df.columns
            ]

            # -----------------------------------
            # 2- Determine rows to display
            # -----------------------------------
            if n_rows == 0:
                display_df = df.loc[:, columns].copy()
                display_df.insert(0, "timestamp", display_df.index)
                row_description = f"{len(df)} rows"

            else:
                if len(df) <= 2 * n_rows:
                    display_df = df.loc[:, columns].copy()
                else:
                    head = df.loc[:, columns].head(n_rows)
                    tail = df.loc[:, columns].tail(n_rows)
                    display_df = pd.concat([head, tail]).copy()

                display_df.insert(0, "timestamp", display_df.index)
                row_description = f"head={n_rows}, tail={n_rows}"
            # -----------------------------------
            # 3- Timeframe section
            # -----------------------------------
            lines.append(
                f"<details>"
            )
            lines.append(
                f"<summary>"
                f"{tf} — "
                f"{len(df)} rows × {len(df.columns)} columns "
                f"({row_description})"
                f"</summary>"
            )
            lines.append("")

            # -----------------------------------
            # 4- Missing columns information
            # -----------------------------------
            if missing_columns:
                lines.append(
                    "> **Missing columns:** "
                    + ", ".join(
                        f"`{col}`"
                        for col in missing_columns
                    )
                )
                lines.append("")

            # -----------------------------------
            # 5- Empty DataFrame
            # -----------------------------------
            if df.empty:
                lines.append("**DataFrame is empty.**")
            else:
                lines.append(display_df.to_markdown(index=False))
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # -----------------------------------------------------------
        # Write file
        # -----------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = Path(file_path)
        output_path = path.with_name(
            f"{path.stem}_{timestamp}{path.suffix}"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

# ======================================================================================= END






""" آموزشی:
-----------
    @property
    عبارت @property یک Decorator در پایتون است که باعث می‌شود یک متد،
    از بیرون کلاس مثل یک متغیر (Attribute) رفتار کند، نه مثل یک تابع.
    بنابراین میتوانیم بجای 
        .timeframe()
    به سادگی از 
        .timeframe
    استفاده کنیم و دیگر نیازی به پرانتزهای نشان دهنده تابع یا متد نداریم
    
"""