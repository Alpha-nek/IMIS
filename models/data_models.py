# =============================================================================
# Data Models for IMIS Scheduler
# =============================================================================

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class ShiftTimingPreference(str, Enum):
    """Enum for shift timing preferences."""
    FRONT_LOADED = "front_loaded"  # Prefer shifts in first half of month
    BACK_LOADED = "back_loaded"    # Prefer shifts in second half of month
    EVEN_DISTRIBUTION = "even_distribution"  # No preference

class DayOfWeek(str, Enum):
    """Enum for days of the week."""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

class ProviderPreferences(BaseModel):
    """Provider-specific scheduling preferences."""
    # Shift type preferences
    shift_preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "R12": True,   # Rounder shifts
            "A12": True,   # Admitter shifts (7am-7pm)
            "A10": True,   # Admitter shifts (10am-10pm)
            "N12": False,  # Night shifts
            "NB": False,   # Night bridge
            "APP": False   # APP shifts
        }
    )
    
    # Unavailable dates (specific dates)
    unavailable_dates: List[str] = Field(default_factory=list)
    
    # Unavailable days of the week (0=Monday, 6=Sunday)
    unavailable_days_of_week: List[int] = Field(default_factory=list)
    
    # Vacation periods
    vacations: List[Dict[str, str]] = Field(default_factory=list)
    
    # FTE percentage (0.0 to 1.0)
    fte_percentage: float = Field(1.0, ge=0.0, le=1.0)
    
    # Day vs night shift percentage
    day_percentage: float = Field(80.0, ge=0.0, le=100.0)
    
    # Shift timing preference
    shift_timing_preference: ShiftTimingPreference = ShiftTimingPreference.EVEN_DISTRIBUTION
    
    # Minimum rest days between shifts
    min_rest_days: int = Field(2, ge=0, le=7)

class RuleConfig(BaseModel):
    """Global scheduling rules configuration."""
    max_consecutive_shifts: int = Field(7, ge=1, le=14)
    min_days_between_shifts: int = Field(1, ge=0, le=7)
    expected_shifts_per_month: int = Field(15, ge=1, le=31, description="Expected shifts per month (15 for 30-day months, 16 for 31-day months)")
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
