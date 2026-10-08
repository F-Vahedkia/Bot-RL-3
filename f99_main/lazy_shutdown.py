# ============================================================
# Lazy-Shutdown Handler (قابل اضافه به هر برنامه)
# ============================================================
import signal
import threading
import time
# import sys
# from typing import Callable, Optional
import logging

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LazyShutdown:
    """
    مدیریت توقف نرم (graceful shutdown) با فرصت اتمام کارهای جاری
    """
    def __init__(self, timeout: float = 5.0):
        self._shutdown_event = threading.Event()
        self._timeout = timeout
        self._original_handlers = {}
        self._is_shutting_down = False

    def _signal_handler(self, signum, frame):
        if self._is_shutting_down:
            # اگر قبلاً درخواست shutdown داده شده، دوباره آن را نادیده بگیر
            return
        self._is_shutting_down = True
        logger.info(f"🛑 Shutdown signal received (signal {signum}). Finishing current work...")
        self._shutdown_event.set()

    def register(self):
        """ثبت signal handler برای SIGINT و SIGTERM"""
        self._original_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("LazyShutdown signal handlers registered.")

    def unregister(self):
        """بازگرداندن signal handler‌های قبلی"""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        logger.debug("LazyShutdown signal handlers unregistered.")

    def wait(self, check_interval: float = 0.5) -> bool:
        """
        منتظر می‌ماند تا shutdown درخواست شود.
        اگر درخواست shutdown شود، True برمی‌گرداند.
        اگر timeout تمام شود، False برمی‌گرداند.
        """
        waited = 0.0
        while waited < self._timeout:
            if self._shutdown_event.is_set():
                return True
            time.sleep(check_interval)
            waited += check_interval
        return False

    def is_requested(self) -> bool:
        """بررسی اینکه آیا shutdown درخواست شده است"""
        return self._shutdown_event.is_set()

    def reset(self):
        """بازنشانی وضعیت shutdown (برای استفاده‌ی مجدد)"""
        self._shutdown_event.clear()
        self._is_shutting_down = False