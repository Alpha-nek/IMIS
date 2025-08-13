# =============================================================================
# Data Models for IMIS Scheduler
# =============================================================================

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class RuleConfig(BaseModel):
    """Global scheduling rules configuration."""
    max_consecutive_shifts: int = Field(7, ge=1, le=14)
    min_days_between_shifts: int = Field(1, ge=0, le=7)
    max_shifts_per_month: int = Field(16, ge=1, le=31)
    min_shifts_per_month: int = Field(8, ge=0, le=31)
    max_weekend_shifts_per_month: int = Field(4, ge=0, le=10)
    min_weekend_shifts_per_month: int = Field(1, ge=0, le=10)
    max_night_shifts_per_month: int = Field(8, ge=0, le=31)
    min_night_shifts_per_month: int = Field(2, ge=0, le=31)
    max_holiday_shifts_per_month: int = Field(2, ge=0, le=10)
    min_holiday_shifts_per_month: int = Field(0, ge=0, le=10)
    
    # Legacy fields for backward compatibility
    min_shifts_per_provider: int = Field(8, ge=0, le=31)
    max_shifts_per_provider: int = Field(16, ge=1, le=31)
    min_rest_days_between_shifts: float = Field(1.0, ge=0.0, le=14.0)
    min_block_size: int = Field(3, ge=1, le=7, description="Minimum consecutive days in a block")
    max_block_size: Optional[int] = 7
    require_at_least_one_weekend: bool = True
    max_nights_per_provider: Optional[int] = Field(8, ge=0, le=31)

class Provider(BaseModel):
    """Provider model with validation."""
    initials: str

    @field_validator("initials")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip().upper()

class SEvent(BaseModel):
    """Internal event schema aligned with FullCalendar."""
    id: str
    title: str
    start: datetime
    end: datetime
    backgroundColor: Optional[str] = None
    extendedProps: Dict[str, Any] = {}

    def to_json_event(self) -> Dict[str, Any]:
        """Convert to JSON-compatible dictionary for calendar display."""
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "backgroundColor": self.backgroundColor,
            "extendedProps": self.extendedProps,
        }
