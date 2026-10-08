# f08_risk/state.py (3)
#
# Created: 1405/06/19

# Runtime state of Risk Engine.

"""
نکات مهم که chatGPT در دور دوم نوشته است:
-------------------------------------------
این فایل را در حال حاضر به‌عنوان Runtime State داخلی RiskEngine در نظر می‌گیرم.
نکتهٔ مهم این فایل این است که خودش تصمیم‌گیری ریسک نمی‌کند;
فقط وضعیت اجرای Engine را برای monitoring، audit، evaluation و diagnostics نگهداری می‌کند.

موارد اصلی آن:
    - شمارندهٔ کل evaluationها
    - تعداد APPROVED، MODIFIED و REJECTED
    - زمان آخرین evaluation
    - آخرین status و decision_id
    - تعداد کل violationها
    - شمارش violationها بر اساس code
    - record() برای ثبت نتیجهٔ هر evaluation
    - reset() برای پاک‌کردن state

از نظر طراحی، این فایل فعلاً با مسئولیتی که برایش تعریف شده سازگار است
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from f08_risk.contracts import (
    RiskDecisionStatus,
)


@dataclass(slots=True)
class RiskEngineState:
    """
    وضعیت runtime Risk Engine.
    این state فقط برای:
        - monitoring
        - audit
        - evaluation
        - diagnostics
    است.
    """
    evaluation_count: int = 0
    approved_count: int = 0
    modified_count: int = 0
    rejected_count: int = 0
    last_evaluation_at: Optional[datetime] = None
    last_status: Optional[RiskDecisionStatus] = None
    last_decision_id: Optional[str] = None
    violations_count: int = 0
    violation_counts: Dict[str, int] = field(default_factory=dict)


    def record(
        self,
        *,
        status: RiskDecisionStatus,
        decision_id: str,
        timestamp: datetime,
        violation_codes: tuple[str, ...],
    ) -> None:

        self.evaluation_count += 1

        if status == RiskDecisionStatus.APPROVED:
            self.approved_count += 1

        elif status == RiskDecisionStatus.MODIFIED:
            self.modified_count += 1

        elif status == RiskDecisionStatus.REJECTED:
            self.rejected_count += 1

        self.last_evaluation_at = timestamp
        self.last_status = status
        self.last_decision_id = decision_id
        self.violations_count += len(violation_codes)

        for code in violation_codes:
            self.violation_counts[code] = (
                self.violation_counts.get(code, 0)
                + 1
            )

    def reset(self) -> None:
        self.evaluation_count = 0
        self.approved_count = 0
        self.modified_count = 0
        self.rejected_count = 0
        self.last_evaluation_at = None
        self.last_status = None
        self.last_decision_id = None
        self.violations_count = 0
        self.violation_counts.clear()

# ============================================================================= END