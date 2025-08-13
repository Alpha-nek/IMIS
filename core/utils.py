# =============================================================================
# Core Utilities for IMIS Scheduler
# =============================================================================

import calendar as cal
from datetime import datetime, date, timedelta, time
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from models.constants import HOLIDAY_RULES, APP_PROVIDER_INITIALS

def is_holiday(check_date: date) -> Optional[Dict]:
    """Check if a date is a holiday and return holiday info if so."""
    for holiday_name, holiday_info in HOLIDAY_RULES.items():
        holiday_date = holiday_info["date_func"](check_date.year)
        if check_date == holiday_date:
            return {
                "name": holiday_name,
                "description": holiday_info["description"],
                "capacity_multiplier": holiday_info["capacity_multiplier"]
            }
    return None

def get_holiday_adjusted_capacity(base_capacity: int, check_date: date) -> int:
    """Get capacity adjusted for holidays."""
    holiday_info = is_holiday(check_date)
    if holiday_info:
        return max(1, int(base_capacity * holiday_info["capacity_multiplier"]))
    return base_capacity

def _normalize_initials_list(items):
    """Normalize and clean a list of provider initials."""
    return sorted({str(x).strip().upper() for x in items if str(x).strip()})

def parse_time(s: str) -> time:
    """Parse time string in HH:MM format."""
    hh, mm = s.split(":")
    return time(int(hh), int(mm))

def date_range(start: date, end: date):
    """Generate date range from start to end (inclusive)."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def month_start_end(year: int, month: int):
    """Get start and end dates for a month."""
    start = date(year, month, 1)
    end = (start + relativedelta(months=1)) - timedelta(days=1)
    return start, end

def get_expected_shifts_for_month(year: int, month: int) -> int:
    """Get expected shifts for a specific month based on number of days."""
    days = cal.monthrange(year, month)[1]
    if days == 31:
        return 16
    if days == 30:
        return 15
    # For February (28/29 days), use a reasonable expected value
    return 14

def get_min_shifts_for_month(year: int, month: int) -> int:
    """Get minimum shifts required for a specific month based on number of days."""
    days = cal.monthrange(year, month)[1]
    if days == 31:
        return 16
    if days == 30:
        return 15
    # For February (28/29 days), use a reasonable minimum
    return 14

def recommended_max_shifts_for_month():
    """Recommended max shifts per provider for the current month."""
    import streamlit as st
    year = st.session_state.get("current_year", datetime.now().year)
    month = st.session_state.get("current_month", datetime.now().month)
    return get_min_shifts_for_month(year, month)

def make_month_days(year: int, month: int) -> List[date]:
    """Generate list of dates for a month."""
    start, end = month_start_end(year, month)
    return list(date_range(start, end))

def make_three_months_days(start_year: int, start_month: int) -> List[date]:
    """Generate days for three consecutive months starting from start_month."""
    all_days = []
    for i in range(3):
        year = start_year
        month = start_month + i
        if month > 12:
            year += 1
            month -= 12
        all_days.extend(make_month_days(year, month))
    return all_days

def _expand_vacation_dates(vacations: list) -> set:
    """Expand vacation date ranges to a set of individual dates."""
    out = set()
    for rng in vacations or []:
        try:
            s = pd.to_datetime(rng.get("start")).date()
            e = pd.to_datetime(rng.get("end")).date()
        except Exception:
            continue
        if e < s:
            s, e = e, s
        for d in pd.date_range(s, e):
            out.add(d.date())
    return out

def _provider_has_vacation_in_month(pr: dict) -> bool:
    """Check if provider has vacation in the currently selected month."""
    import streamlit as st
    if not pr:
        return False
    vac = pr.get("vacations", [])
    if not vac:
        return False
    year = st.session_state.get("current_year", datetime.now().year)
    month = st.session_state.get("current_month", datetime.now().month)
    ym = (year, month)
    for d in _expand_vacation_dates(vac):
        if (d.year, d.month) == ym:
            return True
    return False

def _provider_vacation_weeks_in_month(pr: dict, year: int, month: int) -> int:
    """Count vacation weeks a provider has in a specific month."""
    if not pr:
        return 0
    vac = pr.get("vacations", [])
    if not vac:
        return 0
    
    # Get all vacation dates for this month
    month_vacation_dates = set()
    for d in _expand_vacation_dates(vac):
        if (d.year, d.month) == (year, month):
            month_vacation_dates.add(d)
    
    if not month_vacation_dates:
        return 0
    
    # Count weeks (7 consecutive days = 1 week)
    weeks = 0
    sorted_dates = sorted(month_vacation_dates)
    
    i = 0
    while i < len(sorted_dates):
        # Count consecutive days starting from this date
        consecutive_count = 1
        current_date = sorted_dates[i]
        
        # Check for consecutive days
        for j in range(i + 1, len(sorted_dates)):
            if (sorted_dates[j] - current_date).days == consecutive_count:
                consecutive_count += 1
            else:
                break
        
        # Calculate weeks (7 days = 1 week)
        weeks += consecutive_count // 7
        if consecutive_count % 7 > 0:  # Partial week counts as 1 week
            weeks += 1
        
        # Skip the dates we've already counted
        i += consecutive_count
    
    return weeks

def get_shift_label_maps():
    """Get mapping between shift keys and labels."""
    import streamlit as st
    from models.constants import DEFAULT_SHIFT_TYPES
    stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
    label_for_key = {s["key"]: s["label"] for s in stypes}
    key_for_label = {v: k for k, v in label_for_key.items()}
    return label_for_key, key_for_label

def provider_weekend_count(p: str) -> int:
    """Count weekend shifts for a provider from current events."""
    import streamlit as st
    events = st.session_state.get("events", [])
    return sum(1 for e in events
               if (e.get("extendedProps") or {}).get("provider") == p and 
               pd.to_datetime(e.get("start")).weekday() >= 5)

def get_global_rules():
    """Get global rules from session state."""
    import streamlit as st
    from models.data_models import RuleConfig
    return RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))

def is_provider_unavailable_on_date(provider: str, day: date) -> bool:
    """Check if provider is unavailable on a specific date."""
    import streamlit as st
    pkey = (provider or "").strip().upper()
    pr = st.session_state.get("provider_rules", {}).get(pkey, {}) or {}

    # Check specific dates
    for tok in pr.get("unavailable_dates", []):
        try:
            if pd.to_datetime(tok).date() == day:
                return True
        except Exception:
            pass

    # Check vacation ranges
    for rng in pr.get("vacations", []) or []:
        try:
            s = pd.to_datetime(rng.get("start")).date()
            e = pd.to_datetime(rng.get("end")).date()
            if e < s: 
                s, e = e, s
            if s <= day <= e:
                return True
        except Exception:
            pass
    return False
