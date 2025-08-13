# =============================================================================
# Core Utilities for IMIS Scheduler
# =============================================================================

import calendar as cal
from datetime import datetime, date, timedelta, time
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from models.constants import HOLIDAY_RULES, APP_PROVIDER_INITIALS
from models.data_models import ShiftTimingPreference

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

def is_provider_unavailable_on_date(provider: str, check_date: date, provider_rules: Dict) -> bool:
    """
    Check if a provider is unavailable on a specific date.
    Considers specific unavailable dates, unavailable days of week, and vacation periods.
    """
    if not provider or provider not in provider_rules:
        return False
    
    provider_rule = provider_rules[provider]
    if not isinstance(provider_rule, dict):
        return False
    
    # Check specific unavailable dates
    unavailable_dates = provider_rule.get("unavailable_dates", [])
    date_str = check_date.strftime("%Y-%m-%d")
    if date_str in unavailable_dates:
        return True
    
    # Check unavailable days of the week (0=Monday, 6=Sunday)
    unavailable_days_of_week = provider_rule.get("unavailable_days_of_week", [])
    day_of_week = check_date.weekday()  # 0=Monday, 6=Sunday
    if day_of_week in unavailable_days_of_week:
        return True
    
    # Check vacation periods
    vacations = provider_rule.get("vacations", [])
    for vacation in vacations:
        if isinstance(vacation, dict) and "start" in vacation and "end" in vacation:
            try:
                vacation_start = datetime.strptime(vacation["start"], "%Y-%m-%d").date()
                vacation_end = datetime.strptime(vacation["end"], "%Y-%m-%d").date()
                if vacation_start <= check_date <= vacation_end:
                    return True
            except (ValueError, TypeError):
                continue
    
    return False

def calculate_shift_timing_score(provider: str, shift_date: date, provider_rules: Dict, 
                                year: int, month: int) -> float:
    """
    Calculate a score for shift timing preference.
    Lower score = better preference match.
    """
    if not provider or provider not in provider_rules:
        return 0.0  # Neutral score
    
    provider_rule = provider_rules[provider]
    if not isinstance(provider_rule, dict):
        return 0.0
    
    timing_preference = provider_rule.get("shift_timing_preference", ShiftTimingPreference.EVEN_DISTRIBUTION)
    
    if timing_preference == ShiftTimingPreference.EVEN_DISTRIBUTION:
        return 0.0  # No preference penalty
    
    # Get month start and end dates
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)
    
    # Calculate day position in month (0.0 to 1.0)
    total_days = (month_end - month_start).days + 1
    day_position = (shift_date - month_start).days / total_days
    
    if timing_preference == ShiftTimingPreference.FRONT_LOADED:
        # Prefer first half of month (0.0 to 0.5)
        if day_position <= 0.5:
            return 0.0  # Perfect match
        else:
            # Penalty increases as we get further from first half
            return (day_position - 0.5) * 2.0  # 0.0 to 1.0 penalty
    
    elif timing_preference == ShiftTimingPreference.BACK_LOADED:
        # Prefer second half of month (0.5 to 1.0)
        if day_position >= 0.5:
            return 0.0  # Perfect match
        else:
            # Penalty increases as we get further from second half
            return (0.5 - day_position) * 2.0  # 0.0 to 1.0 penalty
    
    return 0.0  # Default neutral score

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
    # Validate inputs
    if year is None or month is None:
        # Fallback to current date
        from datetime import date
        today = date.today()
        year = today.year
        month = today.month
    
    # Ensure month is within valid range
    if not isinstance(month, int) or month < 1 or month > 12:
        # Fallback to current month
        from datetime import date
        today = date.today()
        month = today.month
    
    # Ensure year is reasonable
    if not isinstance(year, int) or year < 2000 or year > 2100:
        # Fallback to current year
        from datetime import date
        today = date.today()
        year = today.year
    
    try:
        days = cal.monthrange(year, month)[1]
        if days == 31:
            return 16
        if days == 30:
            return 15
        # For February (28/29 days), use a reasonable expected value
        return 14
    except Exception as e:
        # Fallback to default value
        print(f"Error calculating expected shifts for {year}-{month}: {e}")
        return 15

def get_adjusted_expected_shifts(provider: str, year: int, month: int, 
                               provider_rules: Dict, global_rules) -> int:
    """
    Get expected shifts for a provider adjusted for FTE and vacation time.
    Calculates expected shifts based on FTE percentage and reduces for vacation time.
    """
    try:
        # Validate inputs
        if not provider or not isinstance(provider, str):
            return 15  # Default fallback
        
        if not isinstance(provider_rules, dict):
            provider_rules = {}
        
        # Get provider-specific rules
        provider_rule = provider_rules.get(provider, {})
        if not isinstance(provider_rule, dict):
            provider_rule = {}
        
        # Get FTE (Full Time Employment) percentage, default to 1.0 (100% full time)
        fte_percentage = provider_rule.get("fte", 1.0)
        if not isinstance(fte_percentage, (int, float)) or fte_percentage <= 0:
            fte_percentage = 1.0  # Default to full time
        
        # Get base expected shifts for full-time (1.0 FTE)
        base_expected_full_time = get_expected_shifts_for_month(year, month)
        
        # Calculate expected shifts based on FTE
        expected_shifts = int(round(base_expected_full_time * fte_percentage))
        
        # Ensure minimum of 1 shift
        expected_shifts = max(1, expected_shifts)
        
        # Calculate vacation weeks for this month
        vacation_weeks = _provider_vacation_weeks_in_month(provider_rule, year, month)
        
        if vacation_weeks == 0:
            return expected_shifts
        
        # Reduce expected shifts based on vacation weeks
        # Most providers take 1 week (reduce by 3-4 shifts), some take 2 weeks (reduce by 6-8 shifts)
        reduction_per_week = 3.5  # Average reduction per week of vacation
        total_reduction = int(vacation_weeks * reduction_per_week)
        
        adjusted_expected = max(0, expected_shifts - total_reduction)
        
        return adjusted_expected
    except Exception as e:
        print(f"Error calculating adjusted expected shifts for {provider}: {e}")
        return 15  # Default fallback

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
    try:
        # Validate inputs
        if not pr or not isinstance(pr, dict):
            return 0
        
        if year is None or month is None:
            return 0
        
        # Ensure month is within valid range
        if not isinstance(month, int) or month < 1 or month > 12:
            return 0
        
        # Ensure year is reasonable
        if not isinstance(year, int) or year < 2000 or year > 2100:
            return 0
        
        vac = pr.get("vacations", [])
        if not vac or not isinstance(vac, list):
            return 0
        
        # Get all vacation dates for this month
        month_vacation_dates = set()
        try:
            for d in _expand_vacation_dates(vac):
                if (d.year, d.month) == (year, month):
                    month_vacation_dates.add(d)
        except Exception as e:
            print(f"Error expanding vacation dates: {e}")
            return 0
        
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
    except Exception as e:
        print(f"Error calculating vacation weeks: {e}")
        return 0

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

def count_shifts_on_date(day: date, shift_type: str, provider_shifts: Dict) -> int:
    """
    Count how many shifts of a specific type are already assigned on a given day.
    """
    count = 0
    for provider, shifts in provider_shifts.items():
        for shift in shifts:
            if hasattr(shift, 'start'):
                shift_date = shift.start.date()
                shift_type_actual = shift.extendedProps.get("shift_type")
            elif isinstance(shift, dict) and 'start' in shift:
                try:
                    shift_date = datetime.fromisoformat(shift['start']).date()
                    shift_type_actual = shift.get('extendedProps', {}).get("shift_type")
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            if shift_date == day and shift_type_actual == shift_type:
                count += 1
    
    return count
