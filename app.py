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

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

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
        "date_func": lambda year: date(year, 11, 4) + timedelta(days=(3 - date(year, 11, 4).weekday()) % 7 + 21),  # 4th Thursday
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
# UTILITY FUNCTIONS
# =============================================================================

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

def get_min_shifts_for_month(year: int, month: int) -> int:
    """Get minimum shifts required for a specific month based on number of days."""
    days = cal.monthrange(year, month)[1]
    if days == 31:
        return 16
    if days == 30:
        return 15
    # For February (28/29 days), use a reasonable minimum
    return 14

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
    stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
    label_for_key = {s["key"]: s["label"] for s in stypes}
    key_for_label = {v: k for k, v in label_for_key.items()}
    return label_for_key, key_for_label

def provider_weekend_count(p: str) -> int:
    """Count weekend shifts for a provider from current events."""
    events = st.session_state.get("events", [])
    return sum(1 for e in events
               if (e.get("extendedProps") or {}).get("provider") == p and 
               pd.to_datetime(e.get("start")).weekday() >= 5)

def get_global_rules():
    """Get global rules from session state."""
    return RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))

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
# SCHEDULING ENGINE AND VALIDATION
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
    """Validate scheduling rules and return violations by provider."""
    violations: dict[str, list[str]] = {}

    cap_map: dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict] = st.session_state.get("provider_rules", {})

    # Helper function to check provider availability
    def _is_unavailable(p_upper: str, day: date) -> bool:
        """Check if provider is unavailable on a specific date."""
        pr = prov_rules.get(p_upper, {}) or {}
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
# GREEDY SCHEDULING ALGORITHM
# =============================================================================

def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    """Main greedy scheduling algorithm that assigns providers to shifts."""
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

    # ---------- Helper functions that read what we've already assigned in `events` ----------
    def day_shift_count(d: date, skey: str) -> int:
        return sum(1 for e in events if e.extendedProps.get("shift_key") == skey and e.start.date() == d)

    def provider_has_shift_on_day(p: str, d: date) -> bool:
        return any((e.extendedProps.get("provider") or "").upper() == p.upper() and e.start.date() == d for e in events)

    def provider_days(p: str) -> Set[date]:
        pu = (p or "").upper()
        return {e.start.date() for e in events if (e.extendedProps.get("provider") or "").upper() == pu}

    def left_run_len(days_set: Set[date], d: date) -> int:
        run = 0; cur = d - timedelta(days=1)
        while cur in days_set: run += 1; cur -= timedelta(days=1)
        return run

    def right_run_len(days_set: Set[date], d: date) -> int:
        run = 0; cur = d + timedelta(days=1)
        while cur in days_set: run += 1; cur += timedelta(days=1)
        return run

    def total_block_len_if_assigned(p: str, d: date) -> int:
        ds = provider_days(p)
        return left_run_len(ds, d) + 1 + right_run_len(ds, d)

    def provider_weekend_count(p: str) -> int:
        pu = (p or "").upper()
        return sum(1 for e in events if (e.extendedProps.get("provider") or "").upper() == pu and e.start.weekday() >= 5)

    # ---------- APP shift helpers ----------
    def is_weekday(d: date) -> bool:
        return d.weekday() < 5  # Monday = 0, Friday = 4
    
    def is_weekend(d: date) -> bool:
        return d.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def get_app_shift_capacity(d: date) -> int:
        """Get APP shift capacity for a given day (2 on weekdays, 1 on weekends)"""
        if is_weekday(d):
            return 2
        else:
            return 1
    
    def day_app_shift_count(d: date) -> int:
        """Count how many APP shifts are already assigned on a given day"""
        return day_shift_count(d, "APP")
    
    def is_app_provider(p: str) -> bool:
        """Check if a provider is an APP provider"""
        return p.upper() in [ap.upper() for ap in app_providers]

    # ---------- Feasibility and scoring functions ----------
    def ok(p: str, d: date, skey: str) -> bool:
        """Check if a provider can be assigned to a shift on a given day."""
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
        """Score a potential assignment (higher is better)."""
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
       
    # ---------- Build schedule ----------
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
# UTILITY FUNCTIONS FOR SCHEDULING
# =============================================================================

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

def build_empty_roster(days: List[date], shift_types: List[Dict[str, Any]]):
    """Build an empty roster structure for the given days and shift types."""
    roster = {d: {s["key"]: None for s in shift_types} for d in days}
    return roster

def shifts_to_events(roster: Dict[date, Dict[str, Optional[str]]], shift_types: List[Dict[str, Any]]):
    """Convert roster assignments to SEvent objects."""
    stypes = {s["key"]: s for s in shift_types}
    events: List[SEvent] = []
    for d, shifts in roster.items():
        for skey, provider in shifts.items():
            sdef = stypes[skey]
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

def recommended_max_shifts_for_month() -> int:
    """Recommended max shifts per provider for the current month."""
    year = st.session_state.month.year
    month = st.session_state.month.month
    return get_min_shifts_for_month(year, month)

@st.cache_data
def make_month_days(year: int, month: int) -> List[date]:
    """Generate list of dates for a month (cached for performance)."""
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

