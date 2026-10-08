# f06_env/portfolio_state.py (2)

"""
این فایل نشان میدهد:
    - در زمان اجرای Environment وضعیت واقعی حساب و پوزیشن‌ها چگونه در حافظه نگهداری و به‌روزرسانی می‌شود.
و یا به عبارت دیگر:
    - وضعیت Runtime حساب معاملاتی و پوزیشن‌های باز را نگهداری می‌کند.
و یا به عبارت دیگر:
    - الان حساب در چه وضعیتی است و هر Symbol چه Position ای دارد؟
و یا به عبارت دیگر:
    - نگهداری، اعتبارسنجی، دسترسی و به‌روزرسانی وضعیت مالی حساب و وضعیت پوزیشن‌های هر Symbol، همین الان چگونه است؟

این کلاس دارای دو سطح state است. یکی سطح پوزیشن یا نماد و دیگری سطح پورتفولیو.

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import math

# =============================================================================
# Class-1
# =============================================================================
@dataclass
class PositionState:
    """
    Runtime state of one symbol position.
    این کلاس وضعیت یک پوزیشن مربوط به یک Symbol را نگه می‌دارد.
    بنابراین، این کلاس به این سوال جواب میدهد که:
        - پوزیشن این Symbol در همین لحظه چه وضعیتی دارد؟
    """

    symbol: str
    side: int = 0                           # -1:short, 0:flat, +1:long
    lots: float = 0.0                       # در اینجا، این وضعیت بعلی حجم است و نه درخواست حجم
    entry_price: Optional[float] = None
    current_price: Optional[float] = None

    realized_pnl: float = 0.0               # سود و زیان تحقق یافته
    unrealized_pnl: float = 0.0             # سود و زیان تحقق نیافته

    used_margin: float = 0.0
    notional: float = 0.0                   # ارزش اسمی Position

    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def __post_init__(self) -> None:

        # --- 1)
        if not self.symbol:
            raise ValueError("symbol is required")
        self.symbol = str(self.symbol).upper().strip()

        # --- 2)
        self.side = int(self.side)
        if self.side not in (-1, 0, 1):
            raise ValueError("side must be -1, 0, or 1")

        # --- 3)
        self.lots = float(self.lots)
        if not math.isfinite(self.lots) or self.lots < 0.0:
            raise ValueError("lots must be finite and >= 0")

        # --- 2,3)
        if self.side == 0 and self.lots != 0.0:
            raise ValueError("flat position must have lots=0")


# =============================================================================
# Class-2
# =============================================================================
@dataclass
class PortfolioState:
    """
    Account-level state shared by all symbols.
    """

    initial_balance: float = 10_000.0   # سرمایه مبنا/اولیه
    balance: float = 10_000.0           # موجودی حساب بر اساس PnL تحقق‌یافته.
    equity: float = 10_000.0            # ارزش لحظه ای حساب: equity = balance + unrealized_pnl

    peak_equity: float = 10_000.0       # بیشترین Equity ثبت‌ شده تاکنون. این برای Drawdown لازم است.
    day_start_equity: float = 10_000.0  # Equity ابتدای روز که برای Daily Drawdown استفاده می‌شود.

    used_margin: float = 0.0
    free_margin: float = 10_000.0

    realized_pnl: float = 0.0           # سود و زیان تحقق یافته کل پورتوفولیو
    unrealized_pnl: float = 0.0         # سود و زیان تحقق نیافته کل پورتوفولیو

    step_count: int = 0                 # تعداد Stepهای Environment که طی شده است.
    positions: Dict[str, PositionState] = field(default_factory=dict)   # str: symbol
                                        # یک دیکشنری شبیه به این:
                                        #    {
                                        #        "EURUSD": PositionState(...),
                                        #        "GBPUSD": PositionState(...),
                                        #    }
                                        
    # ------------------------------------------------------------------------- 0

    def __post_init__(self) -> None:
        self.initial_balance = float(self.initial_balance)
        self.balance = float(self.balance)
        self.equity = float(self.equity)

        self.peak_equity = float(self.peak_equity)
        self.day_start_equity = float(self.day_start_equity)

        self.used_margin = float(self.used_margin)
        self.free_margin = float(self.free_margin)

        self.realized_pnl = float(self.realized_pnl)
        self.unrealized_pnl = float(self.unrealized_pnl)

        self.step_count = int(self.step_count)

        self._validate_numeric_state()

    # ------------------------------------------------------------------------- 1
    
    def _validate_numeric_state(self) -> None:
        values = (
            self.initial_balance,
            self.balance,
            self.equity,

            self.peak_equity,
            self.day_start_equity,

            self.used_margin,
            self.free_margin,

            self.realized_pnl,
            self.unrealized_pnl,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("PortfolioState contains non-finite values")

        if self.initial_balance <= 0.0:
            raise ValueError("initial_balance must be > 0")

        if self.used_margin < 0.0:
            raise ValueError("used_margin must be >= 0")

        if self.free_margin < 0.0:
            raise ValueError("free_margin must be >= 0")

    # ------------------------------------------------------------------------- 2
    @property
    def total_drawdown(self) -> float:
        """
        Drawdown as a positive fraction.
        Formula:
            (peak_equity - equity) / peak_equity
        Example:
            peak=11000, equity=9900 -> 0.10
        """
        if self.peak_equity <= 0.0:
            return 0.0

        return max(
            0.0,
            (self.peak_equity - self.equity) / self.peak_equity,
        )

    # ------------------------------------------------------------------------- 3
    @property
    def daily_drawdown(self) -> float:
        """
        Daily drawdown as a positive fraction.
        این متد محاسبه میکند که:
            - Equity فعلی چقدر نسبت به Equity ابتدای روز کاهش پیدا کرده است.
            - و این کاهش را همیشه بصورت یک عدد کسری و مثبت ارائه میدهد.
        """
        if self.day_start_equity <= 0.0:
            return 0.0

        return max(
            0.0,
            (self.day_start_equity - self.equity) / self.day_start_equity,
        )

    # ------------------------------------------------------------------------- 4
    @property
    def margin_level(self) -> Optional[float]:
        """
        Equity / used margin.

        "None" means no margin is currently used.
        """
        if self.used_margin <= 0.0:
            return None

        return self.equity / self.used_margin

    # ------------------------------------------------------------------------- 5
    @property
    def open_position_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol
            for symbol, pos in self.positions.items()
            if pos.side != 0 and pos.lots > 0.0
        )

    # ------------------------------------------------------------------------- 6
    @property
    def open_position_count(self) -> int:
        return len(self.open_position_symbols)

    # ------------------------------------------------------------------------- 7

    def reset(self, balance: Optional[float] = None) -> None:
        """
        Reset account state while preserving configured object identity.
        """
        value = (
            self.initial_balance
            if balance is None
            else float(balance)
        )

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("reset balance must be > 0 and finite")

        self.initial_balance = value # ----------
        self.balance = value         # توسط متد update_mark_to_market بروز رسانی میشود
        self.equity = value          # توسط متد update_mark_to_market بروز رسانی میشود

        self.peak_equity = value      # توسط متد update_mark_to_market بروز رسانی میشود
        self.day_start_equity = value # ----------

        self.used_margin = 0.0      # توسط متد update_mark_to_market بروز رسانی میشود
        self.free_margin = value    # توسط متد update_mark_to_market بروز رسانی میشود

        self.realized_pnl = 0.0     # توسط متد update_mark_to_market بروز رسانی میشود
        self.unrealized_pnl = 0.0   # توسط متد update_mark_to_market بروز رسانی میشود

        self.step_count = 0      # --> توسط متد advance_step یک واحد اضافه میشود
        self.positions.clear()   # ----------

    # ------------------------------------------------------------------------- 8

    def set_position(self, position: PositionState) -> None:
        """
        این تابع میگوید:
            - من یک PositionState آماده دارم؛ آن را داخل Portfolio قرار بده.
        """
        if not isinstance(position, PositionState):
            raise TypeError("position must be PositionState")

        self.positions[position.symbol] = position

    # ------------------------------------------------------------------------- 9

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(str(symbol).upper().strip(), None)

    # ------------------------------------------------------------------------- 10

    def get_position(self, symbol: str) -> PositionState:
        """
        این تابع میگوید:
            - Position این Symbol را به من بده.
                اما اگر هنوز وجود ندارد، یک PositionState اولیه برایش بساز
                و بعد آن را بده.
        """
        key = str(symbol).upper().strip()
        if key not in self.positions:
            self.positions[key] = PositionState(symbol=key)
        return self.positions[key]

    # ------------------------------------------------------------------------- 11

    def update_mark_to_market(
        self,
        *,
        realized_delta: float = 0.0,
        unrealized_total: Optional[float] = None,
        used_margin: Optional[float] = None,
    ) -> None:
        
        """
        Update account-level financial values.

        This method performs accounting only.
        It does not calculate broker-specific PnL.
        
        در این متد متغیرهای زیر بروز میشوند:
            self.initial_balance   --> ----no----
            self.balance           --> yes (1)
            self.equity            --> yes (2)
            self.peak_equity       --> yes (3)
            self.day_start_equity  --> ----no----
            self.used_margin       --> yes (4)
            self.free_margin       --> yes (5)
            self.realized_pnl      --> yes (6)
            self.unrealized_pnl    --> yes (7)
            self.step_count        --> ----no----
            self.positions         --> ----no----
        
        وجه تسمیه این متد:
        -------------------
            اصطلاح Mark-to-Market (MTM) در حسابداری و معاملات یعنی:
            ارزش موقعیت‌های باز را با قیمت فعلی بازار به‌روزرسانی کنیم
            تا equity و unrealized_pnl وضعیت فعلی بازار را منعکس کنند.
        بنابراین این متد قصد دارد:
            وضعیت مالی حساب را با ارزش‌گذاری فعلی پوزیشن‌های باز به‌روزرسانی کند.
        نکته:
            این متد خودش Mark-to-Market را محاسبه نمی‌کند;
            چون خودش قیمت بازار را نمی‌گیرد و unrealized_pnl را محاسبه نمی‌کند.
            unrealized_total را از بیرون دریافت می‌کند.
            بنابراین نام update_mark_to_market بیشتر بیانگر نقشی است که متد در فرآیند MTM دارد،
            نه اینکه محاسبه کامل MTM را خودش انجام دهد.
        """
        self.realized_pnl += float(realized_delta)  # --> (8)

        if unrealized_total is not None:
            self.unrealized_pnl = float(unrealized_total)  # --> (9)

        self.balance = self.initial_balance + self.realized_pnl  # --> (2)
        self.equity = self.balance + self.unrealized_pnl         # --> (3)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity  # --> (4)

        if used_margin is not None:
            self.used_margin = max(0.0, float(used_margin))  # --> (6)

        self.free_margin = max(0.0, self.equity - self.used_margin)  # --> (7)

        self._validate_numeric_state()

    # ------------------------------------------------------------------------- 12

    def advance_step(self) -> None:
        self.step_count += 1

# ============================================================================= END
