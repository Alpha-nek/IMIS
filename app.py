# =============================================================================
# Hospitalist Monthly Scheduler - Streamlit Application
# =============================================================================
# 
# A comprehensive scheduling system for hospitalist providers with the following
# features:
# - Multi-provider shift scheduling with APP support
# - Interactive calendar interface
# - Rule-based validation and optimization
# - Google Calendar integration
# - Provider request management
# - Grid-based editing
# - Holiday-aware scheduling
#
# Author: Yazan Al-Fanek, MD and ChatGPT5
# Requirements: streamlit, pandas, numpy, pydantic, streamlit-calendar, google-auth
#
# Run with: streamlit run app.py
# =============================================================================

import uuid
import json
import os
import calendar as cal
from datetime import datetime, date, timedelta, time
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, field_validator

# Import streamlit-calendar with error handling
try:
    from streamlit_calendar import calendar as st_calendar
except Exception:
    st_calendar = None

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Google Calendar API Configuration
GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
GCAL_TOKEN_FILE = 'token.json'          # Created on first successful auth
GCAL_CREDENTIALS_FILE = 'credentials.json'  # Download from Google Cloud
APP_TIMEZONE = 'America/New_York'       # Application timezone

# Default Shift Types Configuration
DEFAULT_SHIFT_TYPES = [
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
    {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "07:00", "color": "#06b6d4"},
    {"key": "APP", "label": "APP Provider",      "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
]

# Provider Rosters
PROVIDER_INITIALS_DEFAULT = [
    "AA","AD","AM","FS","JM","JT","KA","LN","SM","OI","NP","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","EB","RS","RG","CJ","MS","AT",
    "YH","XL","MA","LM","MQ","CM","AI"
]

# APP Provider roster - these providers can only take APP shifts
APP_PROVIDER_INITIALS = ["JA", "DN", "KP", "AR", "JL"]

# Default shift capacities
DEFAULT_SHIFT_CAPACITY = {"N12": 4, "NB": 1, "R12": 13, "A12": 1, "A10": 2, "APP": 2}

# Holiday rules - reduced capacity on major holidays
HOLIDAY_RULES = {
    "thanksgiving": {
        "date_func": lambda year: date(year, 11, 4) + timedelta(days=(3 - date(year, 11, 4).weekday()) % 7 + 21),
        "capacity_multiplier": 0.5,  # 50% of normal capacity
        "description": "Thanksgiving Day"
    },
    "christmas": {
        "date_func": lambda year: date(year, 12, 25),
        "capacity_multiplier": 0.3,  # 30% of normal capacity
        "description": "Christmas Day"
    },
    "new_years": {
        "date_func": lambda year: date(year, 1, 1),
        "capacity_multiplier": 0.4,  # 40% of normal capacity
        "description": "New Year's Day"
    }
}

# =============================================================================
# DATA MODELS (Pydantic)
# =============================================================================

class RuleConfig(BaseModel):
    """Global scheduling rules configuration."""
    min_shifts_per_provider: int = 15
    max_shifts_per_provider: int = Field(15, ge=1, le=31)
    min_rest_days_between_shifts: float = Field(1.0, ge=0.0, le=14.0)
    min_block_size: int = Field(3, ge=1, le=7, description="Minimum consecutive days in a block")
    max_block_size: Optional[int] = 7
    require_at_least_one_weekend: bool = True
    max_nights_per_provider: Optional[int] = Field(6, ge=0, le=31)

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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _normalize_initials_list(items) -> Set[str]:
    """Normalize and clean a list of provider initials."""
    return sorted({str(x).strip().upper() for x in items if str(x).strip()})

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

def get_min_shifts_for_month(year: int, month: int) -> int:
    """Get minimum shifts required for a specific month based on number of days."""
    days = cal.monthrange(year, month)[1]
    if days == 31:
        return 16
    if days == 30:
        return 15
    return 14  # For February (28/29 days)

def get_shift_label_maps():
    """Get mapping between shift keys and labels."""
    stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
    label_for_key = {s["key"]: s["label"] for s in stypes}
    key_for_label = {v: k for k, v in label_for_key.items()}
    return label_for_key, key_for_label

def get_global_rules():
    """Get global rules from session state."""
    return RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))

def recommended_max_shifts_for_month() -> int:
    """Recommended max shifts per provider for the current month."""
    year = st.session_state.month.year
    month = st.session_state.month.month
    return get_min_shifts_for_month(year, month)

# =============================================================================
# VACATION AND AVAILABILITY HELPERS
# =============================================================================

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
    if not pr:
        return False
    vac = pr.get("vacations", [])
    if not vac:
        return False
    ym = (st.session_state.month.year, st.session_state.month.month)
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
        consecutive_count = 1
        current_date = sorted_dates[i]
        
        for j in range(i + 1, len(sorted_dates)):
            if (sorted_dates[j] - current_date).days == consecutive_count:
                consecutive_count += 1
            else:
                break
        
        weeks += consecutive_count // 7
        if consecutive_count % 7 > 0:
            weeks += 1
        
        i += consecutive_count
    
    return weeks

def is_provider_unavailable_on_date(provider: str, day: date) -> bool:
    """Check if provider is unavailable on a specific date."""
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

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state with default values."""
    st.set_page_config(page_title="Scheduling", layout="wide", initial_sidebar_state="collapsed")
    
    # Ensure data directory exists
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load provider rules from file if it exists
    provider_rules_path = os.path.join(data_dir, "provider_rules.json")
    if os.path.exists(provider_rules_path):
        try:
            with open(provider_rules_path, "r") as f:
                loaded_rules = json.load(f)
            existing_rules = st.session_state.get("provider_rules", {})
            existing_rules.update(loaded_rules)
            st.session_state["provider_rules"] = existing_rules
        except Exception as e:
            st.error(f"Failed to load provider_rules.json: {e}")
    
    # Load provider caps from file if it exists
    provider_caps_path = os.path.join(data_dir, "provider_caps.json")
    if os.path.exists(provider_caps_path):
        try:
            with open(provider_caps_path, "r") as f:
                st.session_state["provider_caps"] = json.load(f)
        except Exception as e:
            st.error(f"Failed to load provider_caps.json: {e}")

    # Initialize session state with defaults
    st.session_state.setdefault("month", date.today().replace(day=1))
    
    # Load default providers from CSV file if providers_df is empty
    if "providers_df" not in st.session_state or st.session_state.providers_df.empty:
        try:
            if os.path.exists("IMIS_initials.csv"):
                providers_df = pd.read_csv("IMIS_initials.csv")
                providers_df = providers_df.dropna()
                providers_df["initials"] = providers_df["initials"].astype(str).str.strip().str.upper()
                providers_df = providers_df[providers_df["initials"] != ""]
                providers_df = providers_df[providers_df["initials"] != "nan"]
                providers_df = providers_df[providers_df["initials"] != "NO"]
                if not providers_df.empty:
                    st.session_state["providers_df"] = providers_df
                    st.session_state["providers_loaded"] = True
                else:
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
            else:
                default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                st.session_state["providers_df"] = default_providers
                st.session_state["providers_loaded"] = True
        except Exception as e:
            st.error(f"Failed to load providers: {e}")
            default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
            st.session_state["providers_df"] = default_providers
            st.session_state["providers_loaded"] = True
    
    # Set other defaults
    st.session_state.setdefault("shift_types", DEFAULT_SHIFT_TYPES.copy())
    st.session_state.setdefault("shift_capacity", DEFAULT_SHIFT_CAPACITY.copy())
    st.session_state.setdefault("provider_caps", {})
    st.session_state.setdefault("provider_rules", {})
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("comments", {})
    st.session_state.setdefault("highlight_provider", "")
    st.session_state.setdefault("rules", RuleConfig().model_dump())
    st.session_state.setdefault("providers_loaded", False)
    st.session_state.setdefault("generation_count", 0)
    st.session_state.setdefault("saved_months", {})

# =============================================================================
# EVENT CONVERSION AND UTILITIES
# =============================================================================

def events_for_calendar(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert events to calendar-compatible format."""
    return [_event_to_dict(e) for e in events]

def _event_to_dict(e):
    """Convert SEvent to dictionary format for calendar display."""
    if isinstance(e, dict):
        out = dict(e)
        # Convert start/end to ISO strings
        for k in ("start", "end"):
            v = out.get(k)
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            elif hasattr(v, "to_pydatetime"):  # pandas Timestamp
                out[k] = v.to_pydatetime().isoformat()
        out.setdefault("extendedProps", {})
        return out

    if hasattr(e, "to_json_event"):
        return _event_to_dict(e.to_json_event())

    # Best-effort generic object
    try:
        return {
            "id": getattr(e, "id", None),
            "title": getattr(e, "title", None),
            "start": getattr(getattr(e, "start", None), "isoformat", lambda: None)(),
            "end": getattr(getattr(e, "end", None), "isoformat", lambda: None)(),
            "backgroundColor": getattr(e, "backgroundColor", None),
            "extendedProps": getattr(e, "extendedProps", {}) or {},
        }
    except Exception:
        return {"raw": str(e)}

def _serialize_events_for_download(events):
    """Serialize events for download."""
    return [_event_to_dict(e) for e in (events or [])]

# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================

@st.cache_data
def make_month_days(year: int, month: int) -> List[date]:
    """Generate list of dates for a specific month."""
    start, end = month_start_end(year, month)
    return list(date_range(start, end))

def make_three_months_days(start_year: int, start_month: int) -> List[date]:
    """Generate days for three consecutive months."""
    all_days = []
    for i in range(3):
        year = start_year
        month = start_month + i
        if month > 12:
            year += 1
            month -= 12
        all_days.extend(make_month_days(year, month))
    return all_days

def parse_time(s: str) -> time:
    """Parse time string (HH:MM) to time object."""
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

def filter_events_for_current_month():
    """Return JSON-safe events only for the month in st.session_state.month."""
    year = st.session_state.month.year
    month = st.session_state.month.month
    evs = events_for_calendar(st.session_state.get("events", []))
    out = []
    for e in evs:
        try:
            d = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        if d.year == year and d.month == month:
            out.append(e)
    return out

# =============================================================================
# STATISTICS AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_provider_statistics(events: List[SEvent]) -> Dict[str, Any]:
    """Calculate comprehensive provider statistics from events."""
    provider_stats = {}
    coverage_by_day = {}
    
    for event in events:
        provider = (event.extendedProps.get("provider") or "").strip().upper()
        if not provider:
            continue
            
        # Initialize provider stats if not exists
        if provider not in provider_stats:
            provider_stats[provider] = {
                "total_shifts": 0,
                "weekend_shifts": 0,
                "shift_types": {},
                "dates": set()
            }
        
        # Count total shifts
        provider_stats[provider]["total_shifts"] += 1
        
        # Count shift types
        shift_type = event.extendedProps.get("shift_type", "Unknown")
        provider_stats[provider]["shift_types"][shift_type] = provider_stats[provider]["shift_types"].get(shift_type, 0) + 1
        
        # Check if weekend shift
        event_date = event.start.date()
        if event_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            provider_stats[provider]["weekend_shifts"] += 1
        
        # Track dates for coverage analysis
        provider_stats[provider]["dates"].add(event_date)
        
        # Track coverage by day
        if event_date not in coverage_by_day:
            coverage_by_day[event_date] = []
        coverage_by_day[event_date].append({
            "provider": provider,
            "shift_type": shift_type,
            "start": event.start,
            "end": event.end
        })
    
    # Convert dates set to list for JSON serialization
    for provider in provider_stats:
        provider_stats[provider]["dates"] = list(provider_stats[provider]["dates"])
    
    return {
        "provider_stats": provider_stats,
        "coverage_by_day": coverage_by_day
    }

def identify_coverage_gaps(events: List[SEvent], shift_types: List[Dict], shift_capacity: Dict) -> List[Dict]:
    """Identify days with insufficient provider coverage."""
    coverage_by_day = {}
    
    # Get the month range from events to check all days
    if not events:
        return []
    
    # Find the month range from events
    event_dates = [event.start.date() for event in events]
    min_date = min(event_dates)
    max_date = max(event_dates)
    
    # Initialize coverage tracking for all days in the month
    current_date = min_date
    while current_date <= max_date:
        coverage_by_day[current_date] = {shift["key"]: [] for shift in shift_types}
        current_date += timedelta(days=1)
    
    # Count actual coverage from events
    for event in events:
        event_date = event.start.date()
        shift_type = event.extendedProps.get("shift_type", "")
        provider = event.extendedProps.get("provider", "").strip().upper()
        
        if event_date in coverage_by_day and shift_type in coverage_by_day[event_date]:
            if provider:  # Only count if provider is not empty
                coverage_by_day[event_date][shift_type].append(provider)
    
    gaps = []
    for date, day_coverage in coverage_by_day.items():
        # Only check days that have at least one event
        day_has_events = any(len(providers) > 0 for providers in day_coverage.values())
        if not day_has_events:
            continue
            
        for shift_type, providers in day_coverage.items():
            # Get the actual expected capacity from the provided shift_capacity dict
            expected_capacity = shift_capacity.get(shift_type, 1)
            
            # Special handling for APP shifts
            if shift_type == "APP":
                if date.weekday() < 5:  # Weekday
                    expected_capacity = 2
                else:  # Weekend
                    expected_capacity = 1
            
            # Apply holiday adjustments
            holiday_info = is_holiday(date)
            if holiday_info:
                expected_capacity = max(1, int(expected_capacity * holiday_info["capacity_multiplier"]))
            
            actual_coverage = len(providers)
            if actual_coverage < expected_capacity:
                gaps.append({
                    "date": date,
                    "shift_type": shift_type,
                    "expected": expected_capacity,
                    "actual": actual_coverage,
                    "providers": providers,
                    "shortage": expected_capacity - actual_coverage
                })
    
    return gaps

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def _contiguous_blocks(dates: List[date]) -> List[Tuple[date, date, int]]:
    """Find contiguous blocks of dates and return (start, end, length) tuples."""
    if not dates:
        return []
    
    blocks = []
    start = prev = dates[0]
    length = 1
    
    for d in dates[1:]:
        if (d - prev).days == 1:
            prev = d
            length += 1
        else:
            blocks.append((start, prev, length))
            start = prev = d
            length = 1
    
    blocks.append((start, prev, length))
    return blocks

def validate_rules(events: list[SEvent], rules: RuleConfig) -> dict[str, list[str]]:
    """Validate scheduling rules and return violations."""
    violations: dict[str, list[str]] = {}

    cap_map: dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict] = st.session_state.get("provider_rules", {})

    # Helper function to check unavailability
    def _is_unavailable(p_upper: str, day: date) -> bool:
        """True if provider p_upper is unavailable on 'day' due to specific dates or vacation ranges."""
        pr = prov_rules.get(p_upper, {}) or {}
        # specific dates
        for tok in pr.get("unavailable_dates", []):
            try:
                if pd.to_datetime(tok).date() == day:
                    return True
            except Exception:
                pass
        # vacation ranges
        for rng in pr.get("vacations", []) or []:
            try:
                s = pd.to_datetime(rng.get("start")).date()
                e = pd.to_datetime(rng.get("end")).date()
            except Exception:
                continue
            if e < s:
                s, e = e, s
            if s <= day <= e:
                return True
        return False

    # Group events by provider and month for validation
    provider_month_events = {}
    for ev in events:
        p_upper = (ev.extendedProps.get("provider") or "").strip().upper()
        if not p_upper:
            continue
        month_key = (ev.start.year, ev.start.month)
        if p_upper not in provider_month_events:
            provider_month_events[p_upper] = {}
        if month_key not in provider_month_events[p_upper]:
            provider_month_events[p_upper][month_key] = []
        provider_month_events[p_upper][month_key].append(ev)

    # Validate each provider's events per month
    for p_upper, month_events in provider_month_events.items():
        for (year, month), month_evs in month_events.items():
            # Get provider rules
            pr = prov_rules.get(p_upper, {}) or {}
            
            # Check if this is an APP provider
            is_app_provider = p_upper in [ap.upper() for ap in APP_PROVIDER_INITIALS]
            
            if is_app_provider:
                # APP providers don't have max shift requirements - they just fill available spots
                # But we still check for other rules like rest periods
                pass
            else:
                # Regular providers: check max shifts using individual provider rules
                # First check if provider has specific max_shifts rule
                if "max_shifts" in pr:
                    eff_max = pr["max_shifts"]
                else:
                    # Use recommended max only if provider doesn't have specific rule
                    eff_max = recommended_max_shifts_for_month()
                
                vacation_weeks = _provider_vacation_weeks_in_month(pr, year, month)
                if vacation_weeks > 0:
                    eff_max = max(0, (eff_max or 0) - (vacation_weeks * 3))
                
                # Validate max shifts for this month
                if eff_max is not None and len(month_evs) > eff_max:
                    violations.setdefault(p_upper, []).append(
                        f"Month {year}-{month:02d}: {len(month_evs)} shifts exceeds max {eff_max}"
                    )

            # Validate minimum shifts for this month (only for regular providers)
            if not is_app_provider:
                # Check if provider has specific min_shifts rule
                if "min_shifts" in pr:
                    min_required = pr["min_shifts"]
                else:
                    # Use default minimum only if provider doesn't have specific rule
                    min_required = get_min_shifts_for_month(year, month)
                
                if len(month_evs) < min_required:
                    violations.setdefault(p_upper, []).append(
                        f"Month {year}-{month:02d}: {len(month_evs)} shifts below minimum {min_required}"
                    )

    # Validate rest periods and block rules
    for ev in events:
        p_upper = (ev.extendedProps.get("provider") or "").strip().upper()
        if not p_upper:
            continue

        # Check if provider is unavailable on this date
        if _is_unavailable(p_upper, ev.start.date()):
            violations.setdefault(p_upper, []).append(
                f"Assigned on unavailable date {ev.start.date()}"
            )

        # Check rest periods
        pr = prov_rules.get(p_upper, {}) or {}
        min_rest_days = float(pr.get("min_rest_days", rules.min_rest_days_between_shifts))
        
        if min_rest_days > 0:
            # Find other events for this provider
            other_events = [e for e in events if 
                           (e.extendedProps.get("provider") or "").strip().upper() == p_upper and 
                           e.id != ev.id]
            
            for other_ev in other_events:
                days_between = abs((ev.start.date() - other_ev.start.date()).days)
                if days_between < min_rest_days:
                    violations.setdefault(p_upper, []).append(
                        f"Insufficient rest: {days_between} days between {ev.start.date()} and {other_ev.start.date()}"
                    )

    # Validate block rules
    for p_upper in provider_month_events.keys():
        p_events = [e for e in events if (e.extendedProps.get("provider") or "").strip().upper() == p_upper]
        if not p_events:
            continue
            
        # Find contiguous blocks
        dates = sorted([e.start.date() for e in p_events])
        blocks = _contiguous_blocks(dates)
        
        for block_start, block_end, block_length in blocks:
            # Check minimum block size
            if block_length < rules.min_block_size:
                violations.setdefault(p_upper, []).append(
                    f"Block {block_start} to {block_end} ({block_length} days) below minimum {rules.min_block_size}"
                )
            
            # Check maximum block size
            if rules.max_block_size and block_length > rules.max_block_size:
                violations.setdefault(p_upper, []).append(
                    f"Block {block_start} to {block_end} ({block_length} days) exceeds maximum {rules.max_block_size}"
                )

    return violations

# =============================================================================
# SCHEDULING ENGINE (Greedy Algorithm)
# =============================================================================

def build_empty_roster(days: List[date], shift_types: List[Dict[str, Any]]) -> Dict[date, Dict[str, Optional[str]]]:
    """Build an empty roster structure for the given days and shift types."""
    return {d: {s["key"]: None for s in shift_types} for d in days}

def shifts_to_events(roster: Dict[date, Dict[str, Optional[str]]], shift_types: List[Dict[str, Any]]) -> List[SEvent]:
    """Convert roster assignments to SEvent objects."""
    sdefs = {s["key"]: s for s in shift_types}
    events: List[SEvent] = []
    
    for d, shifts in roster.items():
        for skey, provider in shifts.items():
            sdef = sdefs[skey]
            # Compute start/end datetimes (handle overnight)
            start_dt = datetime.combine(d, parse_time(sdef["start"]))
            end_dt = datetime.combine(d, parse_time(sdef["end"]))
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            
            title = f"{sdef['label']} — {provider if provider else 'UNASSIGNED'}"
            ev = SEvent(
                id=str(uuid.uuid4()),
                title=title,
                start=start_dt,
                end=end_dt,
                backgroundColor=sdef.get("color"),
                extendedProps={"provider": provider, "shift_key": skey, "label": sdef["label"]},
            )
            events.append(ev)
    return events

def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    """
    Greedy scheduling algorithm that assigns providers to shifts based on rules and preferences.
    
    This is the core scheduling engine that:
    - Respects provider availability and vacation dates
    - Enforces shift type restrictions (APP vs regular providers)
    - Applies holiday capacity adjustments
    - Prefers block scheduling for better continuity
    - Balances workload across providers
    - Handles APP providers with special rules
    """
    # Build lookup for shifts
    sdefs = {s["key"]: s for s in shift_types}
    stypes = [s["key"] for s in shift_types]

    # Session-config maps
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: Dict[str, List[str]] = st.session_state.get("provider_caps", {})
    prov_rules: Dict[str, Dict[str, Any]] = st.session_state.get("provider_rules", {})

    # Get APP providers
    app_providers = APP_PROVIDER_INITIALS.copy()
    
    # Counters and accumulator
    counts: Dict[str, int] = {p: 0 for p in providers}
    nights: Dict[str, int] = {p: 0 for p in providers}
    events: List[SEvent] = []

    # Month-aware/global knobs
    base_max = recommended_max_shifts_for_month()
    mbs = int(getattr(rules, "min_block_size", 1) or 1)
    mbx = getattr(rules, "max_block_size", None)
    min_rest_days_global = float(getattr(rules, "min_rest_days_between_shifts", 1.0))

    # ---------- Helper functions that read what we've already assigned ----------
    def day_shift_count(d: date, skey: str) -> int:
        """Count how many shifts of a given type are assigned on a specific day."""
        return sum(1 for e in events if e.extendedProps.get("shift_key") == skey and e.start.date() == d)

    def provider_has_shift_on_day(p: str, d: date) -> bool:
        """Check if a provider already has a shift on a specific day."""
        return any((e.extendedProps.get("provider") or "").upper() == p.upper() and e.start.date() == d for e in events)

    def provider_days(p: str) -> Set[date]:
        """Get all days a provider is currently assigned to."""
        pu = (p or "").upper()
        return {e.start.date() for e in events if (e.extendedProps.get("provider") or "").upper() == pu}

    def left_run_len(days_set: Set[date], d: date) -> int:
        """Count consecutive days to the left of the given date."""
        run = 0
        cur = d - timedelta(days=1)
        while cur in days_set:
            run += 1
            cur -= timedelta(days=1)
        return run

    def right_run_len(days_set: Set[date], d: date) -> int:
        """Count consecutive days to the right of the given date."""
        run = 0
        cur = d + timedelta(days=1)
        while cur in days_set:
            run += 1
            cur += timedelta(days=1)
        return run

    def total_block_len_if_assigned(p: str, d: date) -> int:
        """Calculate total block length if provider were assigned to this day."""
        ds = provider_days(p)
        return left_run_len(ds, d) + 1 + right_run_len(ds, d)

    def provider_weekend_count(p: str) -> int:
        """Count weekend shifts for a provider."""
        pu = (p or "").upper()
        return sum(1 for e in events if (e.extendedProps.get("provider") or "").upper() == pu and e.start.weekday() >= 5)

    # ---------- APP shift helpers ----------
    def is_weekday(d: date) -> bool:
        """Check if date is a weekday (Monday-Friday)."""
        return d.weekday() < 5
    
    def is_weekend(d: date) -> bool:
        """Check if date is a weekend (Saturday-Sunday)."""
        return d.weekday() >= 5
    
    def get_app_shift_capacity(d: date) -> int:
        """Get APP shift capacity for a given day (2 on weekdays, 1 on weekends)."""
        if is_weekday(d):
            return 2
        else:
            return 1
    
    def day_app_shift_count(d: date) -> int:
        """Count how many APP shifts are already assigned on a given day."""
        return day_shift_count(d, "APP")
    
    def is_app_provider(p: str) -> bool:
        """Check if a provider is an APP provider."""
        return p.upper() in [ap.upper() for ap in app_providers]

    # ---------- Feasibility and scoring functions ----------
    def ok(p: str, d: date, skey: str) -> bool:
        """
        Check if a provider can be assigned to a specific shift on a specific day.
        This is the main feasibility checker that enforces all scheduling rules.
        """
        p_upper = (p or "").upper()

        # Special handling for APP providers
        if is_app_provider(p):
            # APP providers can ONLY take APP shifts
            if skey != "APP":
                return False
            # APP providers don't have min/max shift requirements - they just fill available spots
            # But they still need to follow rest and block rules
        else:
            # Regular providers cannot take APP shifts
            if skey == "APP":
                return False

        # 1) Eligibility (for regular providers)
        if not is_app_provider(p):
            allowed = prov_caps.get(p_upper, [])
            if allowed and skey not in allowed:
                return False

        # 2) Provider overrides / month defaults (different for APP vs regular)
        pr = prov_rules.get(p_upper, {}) or {}
        
        if is_app_provider(p):
            # APP providers: no min/max requirements, just fill spots
            eff_max = None  # No maximum for APP providers
            max_nights = None  # APP providers don't take night shifts
        else:
            # Regular providers: normal min/max logic
            eff_max = pr.get("max_shifts", base_max)
            # Calculate vacation weeks for the current month being generated
            current_month = d.month
            current_year = d.year
            vacation_weeks = _provider_vacation_weeks_in_month(pr, current_year, current_month)
            if vacation_weeks > 0:
                eff_max = max(0, (eff_max or 0) - (vacation_weeks * 3))
            max_nights = pr.get("max_nights", rules.max_nights_per_provider)
        
        min_rest_days = float(pr.get("min_rest_days", min_rest_days_global))

        # 3) Hard unavailability
        if is_provider_unavailable_on_date(p_upper, d):
            return False

        # 4) Per-day caps & one-shift-per-day
        if skey == "APP":
            # APP shift specific rules
            # Check APP shift capacity (2 on weekdays, 1 on weekends)
            if day_app_shift_count(d) >= get_app_shift_capacity(d):
                return False
        else:
            # Regular shift capacity check
            if day_shift_count(d, skey) >= cap_map.get(skey, 1):
                return False
        
        if provider_has_shift_on_day(p, d):
            return False

        # 5) Max totals & nights (only for regular providers)
        if not is_app_provider(p):
            if eff_max is not None and counts[p] + 1 > eff_max:
                return False
            if skey == "N12" and max_nights is not None and nights[p] + 1 > max_nights:
                return False

        # 6) Block cap (applies to both APP and regular providers)
        if mbx and mbx > 0 and total_block_len_if_assigned(p, d) > mbx:
            return False

        return True

    def score(provider_id: str, day: date, shift_key: str) -> float:
        """
        Score a potential assignment to determine the best provider for a shift.
        Higher scores are better. This implements the scheduling preferences.
        """
        sc = 0.0
        
        # Different scoring for APP vs regular providers
        if is_app_provider(provider_id):
            # APP provider scoring - focus on weekend coverage and block consistency
            ds = provider_days(provider_id)
            
            # 1. Weekend coverage priority (APP providers should cover weekends)
            if is_weekend(day):
                sc += 8.0  # High priority for weekend coverage
            
            # 2. Block consistency - prefer longer blocks
            L = left_run_len(ds, day)
            if L > 0:
                sc += 3.0  # Bonus for continuing a block
            if L < mbs:
                sc += 2.0  # Bonus for building up to minimum block size
            
            # 3. Avoid standalone days
            if L == 0 and right_run_len(ds, day) == 0:
                sc -= 2.0  # Penalty for standalone days
            
            # 4. Prefer longer blocks (4-7 days)
            total_block_len = total_block_len_if_assigned(provider_id, day)
            if 4 <= total_block_len <= 7:
                sc += 3.0  # Bonus for optimal block size
            elif total_block_len > 7:
                sc -= 1.0  # Slight penalty for very long blocks
            
        else:
            # Regular provider scoring - original logic
            # Get dynamic minimum for the current month
            current_month_min = get_min_shifts_for_month(day.year, day.month)
            
            # toward minimum for the current month
            if counts[provider_id] < current_month_min:
                sc += 4.0
                # contiguous blocks up to preferred min size
                ds = provider_days(provider_id)
                L = left_run_len(ds, day)
                if L > 0:
                    sc += 2.0
                if L < mbs:
                    sc += 4.0
            
            # Stretch preference: prefer 4-7 day stretches, avoid 1-2 day stretches
            ds = provider_days(provider_id)
            if ds:
                # Check if this would create a short stretch (1-2 days)
                left_run = left_run_len(ds, day)
                right_run = right_run_len(ds, day)
                
                # If this would be a standalone day or very short stretch
                if left_run == 0 and right_run == 0:
                    # Standalone day - strong penalty
                    sc -= 6.0
                elif left_run + right_run + 1 <= 2:
                    # Short stretch (1-2 days) - moderate penalty
                    sc -= 3.0
                elif 4 <= left_run + right_run + 1 <= 7:
                    # Optimal stretch length - bonus
                    sc += 2.0
        
        # gentle load balance
        sc += max(0, 20 - counts[provider_id]) * 0.01
        # soft penalty if this hits the max block size
        if mbx and mbx > 0 and total_block_len_if_assigned(provider_id, day) == mbx:
            sc -= 0.2
        # weekend incentive if required & none yet
        weekend_required = prov_rules.get(provider_id, {}).get("require_weekend", rules.require_at_least_one_weekend)
        if day.weekday() >= 5 and weekend_required and provider_weekend_count(provider_id) == 0:
            sc += 3.0
    
        # soft incentive to meet provider-specific day/night ratio if configured
        try:
            pr = prov_rules.get(provider_id, {}) or {}
            ratio = pr.get("day_night_ratio", None)  # percent of day shifts
            if ratio is not None:
                desired_night_frac = max(0.0, (100.0 - float(ratio)) / 100.0)
                cur_nights = nights.get(provider_id, 0)
                cur_total = counts.get(provider_id, 0)
                est_total = cur_total + 1
                est_nights = cur_nights + (1 if shift_key == "N12" else 0)
                est_night_frac = est_nights / max(1, est_total)
                # penalize if assigning a night would push above desired fraction
                if shift_key == "N12" and est_night_frac > desired_night_frac + 0.05:
                    sc -= 2.0
                # small bonus if assigning a day reduces night fraction toward target
                if shift_key != "N12":
                    if est_night_frac < desired_night_frac - 0.10:
                        sc += 0.5
        except Exception:
            pass

        # Half-month shift preference scoring
        try:
            pr = prov_rules.get(provider_id, {}) or {}
            half_month_pref = pr.get("half_month_preference", None)
            if half_month_pref is not None:
                day_of_month = day.day
                if half_month_pref == 0:  # First half preference
                    if day_of_month <= 15:
                        sc += 1.5  # Bonus for first half
                    else:
                        sc -= 0.5  # Small penalty for second half
                elif half_month_pref == 1:  # Last half preference
                    if day_of_month > 15:
                        sc += 1.5  # Bonus for second half
                    else:
                        sc -= 0.5  # Small penalty for first half
        except Exception:
            pass

        # Shift type consistency within blocks scoring - ENHANCED
        try:
            # Always enforce shift consistency within blocks (not just when provider preference is set)
            ds = provider_days(provider_id)
            if ds:
                # Find the block this day would be part of
                left_run = left_run_len(ds, day)
                right_run = right_run_len(ds, day)
                
                # Check shifts in the existing block
                block_start = day - timedelta(days=left_run)
                block_end = day + timedelta(days=right_run)
                
                # Get shift types in the existing block
                block_shift_types = set()
                for ev in events:
                    if (ev.extendedProps.get("provider") or "").upper() == provider_id.upper():
                        ev_date = ev.start.date()
                        if block_start <= ev_date <= block_end:
                            block_shift_types.add(ev.extendedProps.get("shift_key"))
                
                # Classify shift types
                night_shifts = {"N12", "NB"}  # Night shifts
                day_shifts = {"R12", "A12", "A10"}  # Day shifts
                
                current_shift_type = "night" if shift_key in night_shifts else "day"
                
                # Check if block is consistent
                block_has_nights = any(s in night_shifts for s in block_shift_types)
                block_has_days = any(s in day_shifts for s in block_shift_types)
                
                # Strong preference for shift consistency within blocks
                if block_has_nights and block_has_days:
                    # Mixed block - strong penalty for adding different type
                    if (current_shift_type == "night" and not block_has_nights) or \
                       (current_shift_type == "day" and not block_has_days):
                        sc -= 5.0  # Strong penalty for breaking consistency
                elif block_has_nights and current_shift_type == "day":
                    # Adding day shift to night block - very strong penalty
                    sc -= 8.0
                elif block_has_days and current_shift_type == "night":
                    # Adding night shift to day block - very strong penalty
                    sc -= 8.0
                else:
                    # Consistent block - strong bonus
                    sc += 3.0
                    
                # Additional bonus for extending existing consistent blocks
                if left_run > 0 or right_run > 0:
                    if (current_shift_type == "night" and block_has_nights and not block_has_days) or \
                       (current_shift_type == "day" and block_has_days and not block_has_nights):
                        sc += 2.0  # Bonus for extending consistent block
        except Exception:
            pass

        return sc

    # ---------- Main scheduling loop ----------
    total_assignments = 0
    
    # Add randomness to provider selection for different schedules
    import random
    providers_shuffled = providers.copy()
    random.shuffle(providers_shuffled)
    
    for current_day in days:
        for shift_key in stypes:
            # Get base capacity
            if shift_key == "APP":
                base_capacity = get_app_shift_capacity(current_day)
            else:
                base_capacity = cap_map.get(shift_key, 1)
            
            # Apply holiday adjustments
            capacity = get_holiday_adjusted_capacity(base_capacity, current_day)
            
            for _ in range(capacity):
                candidates = [prov for prov in providers_shuffled if ok(prov, current_day, shift_key)]
                if not candidates:
                    continue
                # Add some randomness to candidate selection when scores are close
                if len(candidates) > 1:
                    scores = [(prov, score(prov, current_day, shift_key)) for prov in candidates]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    # If top 2 scores are within 10% of each other, randomly choose between them
                    if len(scores) >= 2 and scores[0][1] > 0 and (scores[0][1] - scores[1][1]) / scores[0][1] < 0.1:
                        best = random.choice(scores[:2])[0]
                    else:
                        best = scores[0][0]
                else:
                    best = candidates[0]
                total_assignments += 1

                sdef = sdefs[shift_key]
                start_dt = datetime.combine(current_day, parse_time(sdef["start"]))
                end_dt = datetime.combine(current_day, parse_time(sdef["end"]))
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)

                ev = SEvent(
                    id=str(uuid.uuid4()),
                    title=f"{sdef['label']} — {best}",
                    start=start_dt,
                    end=end_dt,
                    backgroundColor=sdef.get("color"),
                    extendedProps={"provider": best, "shift_key": shift_key, "label": sdef["label"]},
                )
                events.append(ev)
                counts[best] += 1
                if shift_key == "N12":
                    nights[best] += 1

    return events

# =============================================================================
# GOOGLE CALENDAR INTEGRATION
# =============================================================================

def get_gcal_service():
    """Get authenticated Google Calendar service."""
    creds = None
    if os.path.exists(GCAL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GCAL_TOKEN_FILE, GCAL_SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GCAL_CREDENTIALS_FILE):
                st.error(f"Missing {GCAL_CREDENTIALS_FILE}. Download from Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(GCAL_CREDENTIALS_FILE, GCAL_SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(GCAL_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return build('calendar', 'v3', credentials=creds)

def gcal_list_calendars(service):
    """List available calendars."""
    try:
        calendars = service.calendarList().list().execute()
        return [(c['id'], c['summary']) for c in calendars.get('items', [])]
    except HttpError:
        return []

def local_event_to_gcal_body(E: dict) -> dict:
    """Map a local event dict to a Google Calendar event body."""
    ext = E.get("extendedProps") or {}
    prov = (ext.get("provider") or "").strip().upper()
    skey = ext.get("shift_key") or ""
    label = ext.get("label") or ""
    title = E.get("title") or f"{label} — {prov}" if prov else label or "Shift"
    return {
        "summary": title,
        "description": f"Provider: {prov}\nShift: {label} ({skey})\nSource: Streamlit Scheduler",
        "start": {"dateTime": E["start"], "timeZone": APP_TIMEZONE},
        "end": {"dateTime": E["end"], "timeZone": APP_TIMEZONE},
        "extendedProperties": {
            "private": {
                "app_event_id": E.get("id",""),
                "shift_key": skey,
                "provider": prov,
            }
        },
    }

def gcal_find_by_app_id(service, calendar_id: str, app_event_id: str):
    """Find a GCal event that matches our local app_event_id (using private extendedProperties)."""
    try:
        resp = service.events().list(
            calendarId=calendar_id,
            privateExtendedProperty=f"app_event_id={app_event_id}",
            maxResults=1,
            singleEvents=True,
        ).execute()
        items = resp.get('items', [])
        return items[0] if items else None
    except HttpError as e:
        # Some accounts may not permit this filter; in that case we skip matching.
        return None

def _is_same_event_times(g_ev: dict, local: dict) -> bool:
    """Shallow compare start/end times between Google Calendar and local events."""
    g_start = (g_ev.get("start") or {}).get("dateTime") or (g_ev.get("start") or {}).get("date")
    g_end = (g_ev.get("end") or {}).get("dateTime") or (g_ev.get("end") or {}).get("date")
    return (str(g_start) == str(local["start"]["dateTime"])) and (str(g_end) == str(local["end"]["dateTime"]))

# =============================================================================
# UI COMPONENTS AND CALENDAR RENDERING
# =============================================================================

def provider_selector():
    """
    Provider dropdown that updates global selection with separate sections for Physicians and APPs.
    
    This creates a unified dropdown that separates physicians and APP providers
    with clear visual separators for better organization.
    """
    physician_roster = (
        st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
        if not st.session_state.providers_df.empty else []
    )
    physician_roster = sorted(physician_roster)
    app_roster = sorted(APP_PROVIDER_INITIALS)
    
    # Create options with separators
    options = ["(All providers)"]
    if physician_roster:
        options.append("--- Physicians ---")
        options.extend(physician_roster)
    if app_roster:
        options.append("--- APPs ---")
        options.extend(app_roster)
    
    cur = st.session_state.get("highlight_provider", "") or ""
    idx = options.index(cur) if cur and cur in options else 0

    sel = st.selectbox("Provider", options=options, index=idx, key="provider_selector")
    st.session_state.highlight_provider = "" if sel == "(All providers)" else sel

def render_calendar():
    """
    Render the main calendar interface with navigation and interaction capabilities.
    
    This function handles:
    - Month navigation controls
    - Google Calendar sync integration
    - Holiday indicators
    - Calendar event display and interaction
    - Provider highlighting
    - Event editing and comments
    """
    st.subheader(f"Calendar — {st.session_state.month:%B %Y}")
    
    # Add month navigation controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("← Previous Month"):
            st.session_state.month = st.session_state.month - relativedelta(months=1)
            st.rerun()
    with col2:
        if st.button("Next Month →"):
            st.session_state.month = st.session_state.month + relativedelta(months=1)
            st.rerun()
    with col3:
        if st.button("Today"):
            st.session_state.month = date.today().replace(day=1)
            st.rerun()
    with col4:
        st.caption("💡 Navigate to change which month the Generate button will create schedules for")
    
    # Add Google Calendar sync button
    if st.button("📅 Sync to Google Calendar", help="Sync current month's schedule to Google Calendar"):
        # Show provider selection for sync
        st.subheader("👤 Select Provider to Sync")
        
        # Get all providers
        if not st.session_state.providers_df.empty:
            all_providers = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
            app_providers = sorted(APP_PROVIDER_INITIALS)
            
            # Filter out APP providers from the physician list
            physician_providers = [p for p in all_providers if p not in app_providers]
            
            # Create provider options with separators
            provider_options = ["(Select Provider)"]
            if physician_providers:
                provider_options.append("--- Physicians ---")
                provider_options.extend(physician_providers)
            if app_providers:
                provider_options.append("--- APPs ---")
                provider_options.extend(app_providers)
            
            # Provider selection
            selected_provider = st.selectbox(
                "Provider to Sync",
                options=provider_options,
                key="quick_sync_provider"
            )
            
            if selected_provider != "(Select Provider)" and not selected_provider.startswith("---"):
                # Initialize provider-specific session state
                provider_key = f"gcal_provider_{selected_provider}"
                if provider_key not in st.session_state:
                    st.session_state[provider_key] = {
                        "connected": False,
                        "calendar_id": "primary",
                        "calendar_name": "Primary Calendar"
                    }
                
                provider_state = st.session_state[provider_key]
                
                # Connect to Google Calendar
                svc = get_gcal_service()
                if svc:
                    provider_state["connected"] = True
                    
                    # Choose calendar
                    calendars = gcal_list_calendars(svc)
                    if calendars:
                        cal_ids = [c[0] for c in calendars]
                        cal_labels = [c[1] for c in calendars]
                        
                        default_cal = provider_state.get("calendar_id", "primary")
                        if default_cal not in cal_ids:
                            default_cal = cal_ids[0]
                        
                        sel_idx = cal_ids.index(default_cal)
                        sel_label = st.selectbox(
                            f"{selected_provider}'s Calendar",
                            options=cal_labels,
                            index=sel_idx,
                            key=f"quick_calendar_{selected_provider}"
                        )
                        sel_id = cal_ids[cal_labels.index(sel_label)]
                        provider_state["calendar_id"] = sel_id
                        provider_state["calendar_name"] = sel_label
                        
                        # Filter events for this provider in current month
                        provider_events = []
                        current_year = st.session_state.month.year
                        current_month = st.session_state.month.month
                        
                        for event in st.session_state.get("events", []):
                            ext = event.get("extendedProps", {})
                            event_provider = (ext.get("provider") or "").strip().upper()
                            if event_provider == selected_provider:
                                try:
                                    event_date = pd.to_datetime(event["start"]).date()
                                    if event_date.year == current_year and event_date.month == current_month:
                                        provider_events.append(event)
                                except Exception:
                                    continue
                        
                        if provider_events:
                            st.write(f"**{selected_provider}**: {len(provider_events)} shifts in {st.session_state.month.strftime('%B %Y')}")
                            
                            if st.button(f"Sync {selected_provider}'s Shifts to Google Calendar", key=f"quick_sync_execute_{selected_provider}"):
                                created, updated = 0, 0
                                for event in provider_events:
                                    body = local_event_to_gcal_body(event)
                                    # Try to find existing GCal event by our app_event_id
                                    g_ev = gcal_find_by_app_id(svc, sel_id, event.get("id", ""))
                                    if g_ev is None:
                                        # Create
                                        svc.events().insert(calendarId=sel_id, body=body).execute()
                                        created += 1
                                    else:
                                        # Update if changed
                                        if (g_ev.get("summary") != body["summary"]) or (not _is_same_event_times(g_ev, body)):
                                            g_ev["summary"] = body["summary"]
                                            g_ev["start"] = body["start"]
                                            g_ev["end"] = body["end"]
                                            g_ev["description"] = body["description"]
                                            g_ev.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                                                body["extendedProperties"]["private"]
                                            )
                                            svc.events().update(calendarId=sel_id, eventId=g_ev["id"], body=g_ev).execute()
                                            updated += 1
                                st.success(f"✅ Synced {selected_provider}: created {created}, updated {updated} events to {sel_label}")
                        else:
                            st.info(f"No shifts found for {selected_provider} in {st.session_state.month.strftime('%B %Y')}")
                    else:
                        st.warning("No calendars available for this account.")
                else:
                    st.error("Failed to connect to Google Calendar. Please check your credentials.")
            else:
                st.info("Please select a provider to sync their shifts.")
        else:
            st.warning("No providers loaded. Please load providers first.")
    
    # Holiday indicator for current month
    current_month_holidays = []
    for day in range(1, 32):  # Check all possible days
        try:
            check_date = date(st.session_state.month.year, st.session_state.month.month, day)
            holiday_info = is_holiday(check_date)
            if holiday_info:
                current_month_holidays.append((day, holiday_info))
        except ValueError:
            break  # Invalid date (e.g., Feb 30)
    
    if current_month_holidays:
        st.info("🎄 **Holiday Schedule**: Reduced capacity will be applied on:")
        for day, holiday_info in current_month_holidays:
            st.write(f"• **{holiday_info['description']}** (Day {day}): {holiday_info['capacity_multiplier']*100:.0f}% of normal capacity")
    
    if st_calendar is None:
        st.warning("streamlit-calendar is not installed or failed to import. Please install and restart.")
        return

    # FullCalendar options
    cal_options = {
        "initialDate": st.session_state.month.isoformat(),
        "height": 780,
        "selectable": True,
        "editable": True,
        "navLinks": True,
        "initialView": "dayGridMonth",
        "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth,timeGridWeek"},
        "eventTimeFormat": {"hour": "2-digit", "minute": "2-digit", "hour12": False},
    }

    # Custom CSS to dim non-highlighted events
    st.markdown(
        """
        <style>
        .fc-event.dim { opacity: 0.25 !important; filter: grayscale(0.8); }
        .comment-badge { font-size: 10px; padding: 2px 6px; border-radius: 8px; background:#111827; color:white; margin-left:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Prepare JSON-safe events
    events = events_for_calendar(st.session_state.get("events", []))
    
    # Filter calendar by the global provider selection
    hi = (st.session_state.get("highlight_provider", "") or "").strip().upper()
    if hi:
        events = [
            e for e in events
            if (e.get("extendedProps", {}).get("provider", "") or "").upper() == hi
        ]
    
    # Render the calendar
    state = st_calendar(
        events=events,
        options=cal_options,
        key="calendar",
    )

    # Handle calendar interactions
    if state.get("eventClick"):
        ev = state["eventClick"]["event"]
        st.info(f"Selected event: {ev['title']}")
        with st.expander("Edit Event"):
            new_title = st.text_input("Title", value=ev["title"], key=f"ttl_{ev['id']}")
            prov = (ev.get("extendedProps") or {}).get("provider", "")
            new_prov = st.text_input("Provider", value=prov, key=f"prov_{ev['id']}").upper()
            if st.button("Save changes", key=f"save_{ev['id']}"):
                for E in st.session_state.events:
                    if E["id"] == ev["id"]:
                        E["title"] = new_title
                        E.setdefault("extendedProps", {})["provider"] = new_prov
                        break
                st.success("Updated.")
        with st.expander("Comments"):
            eid = ev["id"]
            st.session_state.comments.setdefault(eid, [])
            for i, c in enumerate(st.session_state.comments[eid]):
                st.markdown(f"- {c}")
            new_c = st.text_input("Add a comment", key=f"cmt_{eid}")
            if st.button("Add comment", key=f"addc_{eid}") and new_c.strip():
                st.session_state.comments[eid].append(new_c.strip())
                st.success("Comment added.")

    # Update on drop/resize/create/delete
    changed = False

    for k in ["eventDrop", "eventResize"]:
        if state.get(k):
            ev = state[k]["event"]
            for E in st.session_state.events:
                if E["id"] == ev["id"]:
                    E["start"] = ev["start"]
                    E["end"] = ev["end"]
                    changed = True
                    break

    if state.get("select"):
        # Create a new event via selection
        sel = state["select"]
        new_id = str(uuid.uuid4())
        e = {
            "id": new_id,
            "title": "UNASSIGNED",
            "start": sel["startStr"],
            "end": sel["endStr"],
            "allDay": False,
            "extendedProps": {"provider": "", "shift_key": None, "label": "Custom"},
        }
        st.session_state.events.append(e)
        st.session_state.comments[new_id] = []
        changed = True

    if state.get("eventRemove"):
        ev = state["eventRemove"]["event"]
        st.session_state.events = [E for E in st.session_state.events if E["id"] != ev["id"]]
        st.session_state.comments.pop(ev["id"], None)
        changed = True

    if changed:
        st.toast("Calendar updated", icon="✅")

# =============================================================================
# PROVIDER RULES MANAGEMENT
# =============================================================================

def provider_rules_panel():
    """
    Panel for managing provider-specific rules and preferences.
    
    This function provides a comprehensive interface for:
    - Setting allowed shift types per provider
    - Configuring min/max shifts and nights
    - Managing vacation dates and unavailable dates
    - Setting rest period requirements
    - Configuring shift preferences and ratios
    """
    import pandas as pd
    st.header("Provider-specific rules")

    # Roster
    roster = (
        st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
        if not st.session_state.providers_df.empty else []
    )
    if not roster:
        st.info("Add providers first.")
        return

    sel = (st.session_state.get("highlight_provider", "") or "").strip().upper()
    if not sel:
        st.info("Select a provider in the Engine to edit rules.")
        return
    if sel not in roster:
        st.warning(f"{sel} not in current roster.")
        return

    rules_map = st.session_state.setdefault("provider_rules", {})
    st.session_state.setdefault("provider_caps", {})

    # Shift maps
    stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
    label_for_key = {s["key"]: s["label"] for s in stypes}
    key_for_label = {v: k for k, v in label_for_key.items()}

    # Allowed shift types
    current_allowed = st.session_state["provider_caps"].get(sel, [])
    default_labels = [label_for_key[k] for k in current_allowed if k in label_for_key]

    st.subheader(f"Allowed shift types — {sel}")
    picked_labels = st.multiselect(
        "Assign only these shift types (leave empty to allow ALL)",
        options=list(label_for_key.values()),
        default=default_labels,
        key=f"pr_allowed_{sel}",
    )
    if len(picked_labels) == 0:
        st.session_state["provider_caps"].pop(sel, None)
    else:
        st.session_state["provider_caps"][sel] = [key_for_label[lbl] for lbl in picked_labels]

    # ----- Provider-specific rules
    st.markdown("---")
    st.subheader("Provider-specific rules")

    base_default = recommended_max_shifts_for_month()
    curr = rules_map.get(sel, {}).copy()  # work on a copy

    # Show current assigned shifts & weekend count for selected provider
    all_events = st.session_state.get("events", [])
    shift_count = sum(1 for e in all_events if (e.get("extendedProps") or {}).get("provider",""
                       ).strip().upper() == sel)
    weekend_count = sum(1 for e in all_events if (e.get("extendedProps") or {}).get("provider",""
                        ).strip().upper() == sel and pd.to_datetime(e.get("start")).weekday() >= 5)
    st.markdown(f"**Current month shifts:** {shift_count} | **Weekend shifts:** {weekend_count}")

    # Max shifts and nights
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"Recommended default this month: **{base_default}**")
        max_sh = st.number_input(
            "Max shifts (this month)",
            1, 50,
            value=int(curr.get("max_shifts", base_default)),
            key=f"pr_max_{sel}",
        )
    with c2:
        min_sh = st.number_input(
            "Min shifts (this month)",
            1, 50,
            value=int(curr.get("min_shifts", get_min_shifts_for_month(st.session_state.month.year, st.session_state.month.month))),
            key=f"pr_min_{sel}",
        )
    with c3:
        global_rules = get_global_rules()
        default_max_n = global_rules.max_nights_per_provider if global_rules.max_nights_per_provider is not None else 0
        max_n = st.number_input(
            "Max nights (this month)",
            0, 50,
            value=int(curr.get("max_nights", default_max_n)),
            key=f"pr_max_n_{sel}",
        )

    # Weekend requirement
    wk_idx = 0 if curr.get("require_weekend", True) else 1
    wk_choice = st.radio(
        "Weekend requirement",
        options=["Require at least one", "No weekend required"],
        index=wk_idx,
        key=f"pr_weekend_choice_{sel}",
        horizontal=True,
    )

    # Min rest days
    # Backward-compat default: prefer min_rest_days; fall back to converting hours → days
    if "min_rest_days" in curr:
        default_rest_days = float(curr.get("min_rest_days", 1.0))
    elif "min_rest_hours" in curr:
        default_rest_days = float(curr.get("min_rest_hours", 24.0)) / 24.0
    else:
        default_rest_days = float(getattr(global_rules, "min_rest_days_between_shifts", 1.0))
    
    min_rest_days = st.number_input(
        "Min rest days between shifts",
        min_value=0.0, max_value=14.0, step=0.5,
        value=float(default_rest_days),
        key=f"pr_min_rest_{sel}",
    )

    # Day/night ratio per provider (percent day shifts)
    ratio_val = st.slider(
        "Percent day shifts",
        min_value=0, max_value=100,
        value=int(curr.get("day_night_ratio", 70)),
        key=f"pr_ratio_val_{sel}",
    )

    # Half-month shift preference
    half_month_choice = st.radio(
        "Preferred half of month",
        options=["First half (1-15)", "Last half (16-31)", "No preference"],
        index=curr.get("half_month_preference", 2),  # 0=first, 1=last, 2=none
        key=f"pr_half_month_choice_{sel}",
        horizontal=True,
    )
    half_month_val = {"First half (1-15)": 0, "Last half (16-31)": 1, "No preference": 2}[half_month_choice]

    # Shift type consistency within blocks
    consistency_strength = st.slider(
        "Shift consistency preference strength",
        min_value=1, max_value=5,
        value=int(curr.get("shift_consistency_strength", 3)),
        help="1=weak preference, 5=strong preference to avoid mixing night/day shifts",
        key=f"pr_consistency_strength_{sel}",
    )
    
    # Info about stretch preferences
    st.info("💡 **Stretch Preferences**: The system automatically prefers 4-7 day stretches and avoids 1-2 day stretches to reduce provider fatigue.")

    st.markdown("---")
    st.subheader("Unavailable specific dates")
    dates_txt = st.text_input(
        "YYYY-MM-DD, comma-separated",
        value=",".join(curr.get("unavailable_dates", [])),
        key=f"pr_unavail_{sel}",
    )

    st.markdown("---")
    st.subheader("Vacations (date ranges)")
    vac_list = curr.get("vacations", [])
    if not isinstance(vac_list, list):
        vac_list = []

    vc1, vc2, vc3 = st.columns([1, 1, 1])
    with vc1:
        v_start = st.date_input("Start", key=f"pr_vac_start_{sel}")
    with vc2:
        v_end = st.date_input("End", key=f"pr_vac_end_{sel}")
    with vc3:
        if st.button("Add vacation", key=f"pr_vac_add_{sel}"):
            if v_start and v_end:
                s = min(v_start, v_end)
                e = max(v_start, v_end)
                vac_list.append({"start": str(s), "end": str(e)})
                curr["vacations"] = vac_list
                rules_map[sel] = curr
                st.success(f"Added vacation {s} → {e}")
            else:
                st.warning("Pick both start and end.")

    if vac_list:
        for i, rng in enumerate(list(vac_list)):
            rr1, rr2, rr3 = st.columns([2, 2, 1])
            rr1.markdown(f"**Start:** {rng.get('start','')}")
            rr2.markdown(f"**End:** {rng.get('end','')}")
            if rr3.button("Remove", key=f"pr_vac_del_{sel}_{i}"):
                vac_list.pop(i)
                curr["vacations"] = vac_list
                rules_map[sel] = curr
                st.experimental_rerun()

    notes_val = st.text_area("Notes (optional)", value=curr.get("notes", ""), key=f"pr_notes_{sel}")

    # Save (MERGE — never wipe unrelated keys)
    if st.button("Save provider rules", key=f"pr_save_{sel}"):
        new_entry = rules_map.get(sel, {}).copy()

        # Always save all provider-specific rules
        new_entry["max_shifts"] = int(max_sh)
        new_entry["min_shifts"] = int(min_sh)
        new_entry["max_nights"] = int(max_n)
        new_entry["require_weekend"] = (wk_choice == "Require at least one")
        new_entry["min_rest_days"] = float(min_rest_days)
        new_entry["day_night_ratio"] = int(ratio_val)
        new_entry["half_month_preference"] = int(half_month_val)
        new_entry["prefer_shift_consistency"] = True
        new_entry["shift_consistency_strength"] = int(consistency_strength)

        # normalize dates
        import pandas as pd
        toks = [t.strip() for t in dates_txt.split(",") if t.strip()]
        if toks:
            clean = []
            for tok in toks:
                try: 
                    clean.append(str(pd.to_datetime(tok).date()))
                except Exception: 
                    pass
            if clean:
                new_entry["unavailable_dates"] = clean
            else:
                new_entry.pop("unavailable_dates", None)
        else:
            new_entry.pop("unavailable_dates", None)

        # vacations
        if vac_list:
            new_entry["vacations"] = vac_list
        else:
            new_entry.pop("vacations", None)

        # notes
        if notes_val.strip():
            new_entry["notes"] = notes_val.strip()
        else:
            new_entry.pop("notes", None)

        if new_entry:
            rules_map[sel] = new_entry
        else:
            rules_map.pop(sel, None)

        # persist provider rules & caps to disk
        try:
            # Ensure the rules are properly saved to session state first
            st.session_state["provider_rules"] = rules_map.copy()
            
            # Use a more robust path for Streamlit deployment
            import os
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Then save to disk
            provider_rules_path = os.path.join(data_dir, "provider_rules.json")
            with open(provider_rules_path, "w") as _f:
                json.dump(rules_map, _f)
            st.success(f"Saved provider_rules.json to {data_dir} with {len(rules_map)} providers")
        except Exception as e:
            st.error(f"Failed to save provider_rules.json: {e}")
        try:
            import os
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            provider_caps_path = os.path.join(data_dir, "provider_caps.json")
            with open(provider_caps_path, "w") as _f:
                json.dump(st.session_state.get("provider_caps", {}), _f)
            st.success(f"Saved provider_caps.json to {data_dir}")
        except Exception as e:
            st.error(f"Failed to save provider_caps.json: {e}")
        
        st.success("Saved provider rules.")


