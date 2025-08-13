
# app.py — Interactive Monthly Scheduler for Multi-Doctor Shifts (Streamlit)
# ---------------------------------------------------------------
# 
# Vibe Coded By Yazan Al-Fanek, MD and Chat GPT5
# 
# Features
# - Upload or paste a provider list (initials)
# - Define/confirm shift types
# - Pick a month/year
# - Auto-generate a draft schedule from rules (greedy round-robin)
# - FullCalendar-based interactive calendar (select, drag, edit, delete)
# - Filter/highlight by provider
# - Per-event comments (stored alongside events)
# - Validate rules & show violations
# - Save/Load as CSV or JSON
#
# Requirements (install):
#   pip install streamlit pandas numpy pydantic streamlit-calendar python-dateutil
#   # If streamlit-calendar fails to install, see: https://pypi.org/project/streamlit-calendar/
#
# Run:
#   streamlit run app.py

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

try:
    from streamlit_calendar import calendar as st_calendar
except Exception:
    st_calendar = None

# -------------------------
# Utilities & Data Models
# -------------------------

GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
GCAL_TOKEN_FILE = 'token.json'          # created on first successful auth
GCAL_CREDENTIALS_FILE = 'credentials.json'  # download from Google Cloud
APP_TIMEZONE = 'America/New_York'       # your timezone

DEFAULT_SHIFT_TYPES = [
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
    {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "07:00", "color": "#06b6d4"},
    {"key": "APP", "label": "APP Provider",      "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ----- Default provider roster -----
PROVIDER_INITIALS_DEFAULT = [
    "AA","AD","AM","FS","JM","JT","KA","LN","SM","OI","NP","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","EB","RS","RG","CJ","MS","AT",
    "YH","XL","MA","LM","MQ","CM","AI"
]

# ----- APP Provider roster -----
APP_PROVIDER_INITIALS = [
    "JA", "DN", "KP", "AR", "JL"
]

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

def is_holiday(check_date: date) -> Optional[Dict]:
    """Check if a date is a holiday and return holiday info if so"""
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
    """Get capacity adjusted for holidays"""
    holiday_info = is_holiday(check_date)
    if holiday_info:
        return max(1, int(base_capacity * holiday_info["capacity_multiplier"]))
    return base_capacity


def _normalize_initials_list(items):
    return sorted({str(x).strip().upper() for x in items if str(x).strip()})


class RuleConfig(BaseModel):
    # GLOBAL defaults
    min_shifts_per_provider: int = 15
    max_shifts_per_provider: int = Field(15, ge=1, le=31)

    # CHANGED: rest is now measured in DAYS (float)
    min_rest_days_between_shifts: float = Field(1.0, ge=0.0, le=14.0)

    min_block_size: int = Field(3, ge=1, le=7, description="Minimum consecutive days in a block when possible")
    max_block_size: Optional[int] = 7

    require_at_least_one_weekend: bool = True
    max_nights_per_provider: Optional[int] = Field(6, ge=0, le=31)


class Provider(BaseModel):
    initials: str

    @field_validator("initials")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip().upper()

# Internal event schema aligned with FullCalendar
class SEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    backgroundColor: Optional[str] = None
    extendedProps: Dict[str, Any] = {}

    def to_json_event(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "backgroundColor": self.backgroundColor,
            "extendedProps": self.extendedProps,
        }


# -------------------------
# Utility Functions
# -------------------------

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


# --- Vacation helpers ---
def _expand_vacation_dates(vacations: list) -> set:
    """Expand [{'start':'YYYY-MM-DD','end':'YYYY-MM-DD'}, ...] to a set of date objects."""
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
    """True if any vacation day falls in the currently selected month."""
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
    """Count the number of vacation weeks a provider has in a specific month."""
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
    return RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))


def is_provider_unavailable_on_date(provider: str, day: date) -> bool:
    """Returns True if provider is unavailable (specific date or any vacation range) on 'day'."""
    pkey = (provider or "").strip().upper()
    pr = st.session_state.get("provider_rules", {}).get(pkey, {}) or {}

    # Specific dates
    for tok in pr.get("unavailable_dates", []):
        try:
            if pd.to_datetime(tok).date() == day:
                return True
        except Exception:
            pass

    # Vacation ranges
    for rng in pr.get("vacations", []) or []:
        try:
            s = pd.to_datetime(rng.get("start")).date()
            e = pd.to_datetime(rng.get("end")).date()
            if e < s: s, e = e, s
            if s <= day <= e:
                return True
        except Exception:
            pass
    return False


# -------------------------
# State helpers
# -------------------------

# --- Session bootstrap: make sure all keys exist before anything touches them ---
def init_session_state():
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
            # Merge with existing session state rules
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
            # Try to load from IMIS_initials.csv
            if os.path.exists("IMIS_initials.csv"):
                providers_df = pd.read_csv("IMIS_initials.csv")
                # Clean up the data - remove empty rows and normalize initials
                providers_df = providers_df.dropna()
                providers_df["initials"] = providers_df["initials"].astype(str).str.strip().str.upper()
                providers_df = providers_df[providers_df["initials"] != ""]
                providers_df = providers_df[providers_df["initials"] != "nan"]
                providers_df = providers_df[providers_df["initials"] != "NO"]  # Remove problematic entry
                if not providers_df.empty:
                    st.session_state["providers_df"] = providers_df
                    st.session_state["providers_loaded"] = True
                else:
                    # If CSV is empty or has no valid data, use defaults
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
            else:
                # Fallback to default providers if CSV doesn't exist
                default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                st.session_state["providers_df"] = default_providers
                st.session_state["providers_loaded"] = True
        except Exception as e:
            st.error(f"Failed to load providers: {e}")
            # Fallback to default providers
            default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
            st.session_state["providers_df"] = default_providers
            st.session_state["providers_loaded"] = True
    
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


def recommended_max_shifts_for_month() -> int:
    """Recommended max shifts per provider for the current month."""
    year = st.session_state.month.year
    month = st.session_state.month.month
    return get_min_shifts_for_month(year, month)


def events_for_calendar(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert events to calendar-compatible format."""
    return [_event_to_dict(e) for e in events]


def _event_to_dict(e):
    # Convert SEvent -> dict, and coerce datetimes to ISO strings
    if isinstance(e, dict):
        out = dict(e)
        # start / end may be datetime or pandas Timestamp
        for k in ("start", "end"):
            v = out.get(k)
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            elif hasattr(v, "to_pydatetime"):  # pandas Timestamp
                out[k] = v.to_pydatetime().isoformat()
            elif isinstance(v, str):
                # leave as-is
                pass
        # ensure extendedProps exists
        out.setdefault("extendedProps", {})
        return out

    # If it's an SEvent-like object
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
        # last resort: string-ify
        return {"raw": str(e)}


def _serialize_events_for_download(events):
    return [_event_to_dict(e) for e in (events or [])]


@st.cache_data
def make_month_days(year: int, month: int) -> List[date]:
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


# -------------------------
# Google Calendar Integration
# -------------------------

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
        "end":   {"dateTime": E["end"],   "timeZone": APP_TIMEZONE},
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
    """Shallow compare start/end; assumes timeZone handling via body."""
    g_start = (g_ev.get("start") or {}).get("dateTime") or (g_ev.get("start") or {}).get("date")
    g_end   = (g_ev.get("end")   or {}).get("dateTime") or (g_ev.get("end")   or {}).get("date")
    return (str(g_start) == str(local["start"]["dateTime"])) and (str(g_end) == str(local["end"]["dateTime"]))


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


# -------------------------
# Scheduling Engine (Greedy Draft)
# -------------------------

def parse_time(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def month_start_end(year: int, month: int):
    start = date(year, month, 1)
    end = (start + relativedelta(months=1)) - timedelta(days=1)
    return start, end


def build_empty_roster(days: List[date], shift_types: List[Dict[str, Any]]):
    # For each day, each shift key needs 1 provider by default (can be extended later)
    roster = {d: {s["key"]: None for s in shift_types} for d in days}
    return roster


def shifts_to_events(roster: Dict[date, Dict[str, Optional[str]]], shift_types: List[Dict[str, Any]]):
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


# -------------------------
# Validation Rules
# -------------------------

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
    violations: dict[str, list[str]] = {}

    cap_map: dict[str, int]   = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict]      = st.session_state.get("provider_rules", {})

    # --- helpers ---
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
            
            # Debug: Show provider rules being used
            if pr:
                st.caption(f"🔍 Debug: Provider {p_upper} rules: {pr}")
            
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


# Assignment Logic
def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    # Build lookup for shifts
    sdefs  = {s["key"]: s for s in shift_types}
    stypes = [s["key"] for s in shift_types]

    # Session-config maps
    cap_map: Dict[str, int]         = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
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
    # Use dynamic minimum based on month length for the current month being processed
    # We'll calculate this per month during assignment
    mbs = int(getattr(rules, "min_block_size", 1) or 1)
    mbx = getattr(rules, "max_block_size", None)
    min_rest_days_global = float(getattr(rules, "min_rest_days_between_shifts", 1.0))

    # ---------- helpers that read what we've already assigned in `events` ----------
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

    # ---------- feasibility + scoring ----------
    def ok(p: str, d: date, skey: str) -> bool:
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
       

    # ---------- build schedule ----------
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
                end_dt   = datetime.combine(current_day, parse_time(sdef["end"]))
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








def google_calendar_panel():
    st.subheader("Google Calendar Sync")

    # Connect / Authenticate
    svc = None
    if st.button("Connect Google Calendar"):
        svc = get_gcal_service()
        st.session_state["gcal_connected"] = bool(svc)
        if svc:
            st.success("Connected to Google Calendar.")
    else:
        # Try to reuse previous session silently
        if st.session_state.get("gcal_connected"):
            svc = get_gcal_service()

    if not svc:
        st.caption("Click **Connect Google Calendar** to authenticate.")
        return

    # Choose calendar
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    cal_ids = [c[0] for c in calendars]
    cal_labels = [c[1] for c in calendars]

    default_cal = st.session_state.get("gcal_calendar_id", "primary")
    if default_cal not in cal_ids:
        default_cal = cal_ids[0]

    sel_idx = cal_ids.index(default_cal)
    sel_label = st.selectbox("Calendar", options=cal_labels, index=sel_idx)
    sel_id = cal_ids[cal_labels.index(sel_label)]
    st.session_state["gcal_calendar_id"] = sel_id

    st.caption(f"Target: **{sel_label}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Push current month → Google"):
            to_push = filter_events_for_current_month()
            created, updated = 0, 0
            for E in to_push:
                body = local_event_to_gcal_body(E)
                # Try to find existing GCal event by our app_event_id
                g_ev = gcal_find_by_app_id(svc, sel_id, E.get("id",""))
                if g_ev is None:
                    # Create
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
                    # Update if changed
                    if (g_ev.get("summary") != body["summary"]) or (not _is_same_event_times(g_ev, body)):
                        g_ev["summary"] = body["summary"]
                        g_ev["start"]   = body["start"]
                        g_ev["end"]     = body["end"]
                        g_ev["description"] = body["description"]
                        g_ev.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                            body["extendedProperties"]["private"]
                        )
                        svc.events().update(calendarId=sel_id, eventId=g_ev["id"], body=g_ev).execute()
                        updated += 1
            st.success(f"Pushed month: created {created}, updated {updated}")

    with c2:
        if st.button("Remove this month's pushed events from Google"):
            # We'll look for events in this month that have our app_event_id private property and delete them
            year = st.session_state.month.year
            month = st.session_state.month.month
            start = datetime(year, month, 1)
            end = (start + relativedelta(months=1))
            time_min = start.isoformat() + "Z"
            time_max = end.isoformat() + "Z"

            removed = 0
            # Fetch all events in window and filter by privateExtendedProperty via app_event_id of local events
            local_ids = {e["id"] for e in filter_events_for_current_month()}
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id, timeMin=time_min, timeMax=time_max,
                    singleEvents=True, showDeleted=False, pageToken=page_token
                ).execute()
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            st.success(f"Removed {removed} events from Google for this month.")

def provider_selector():
    """One provider dropdown that updates global selection with separate sections for Physicians and APPs."""
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

    # Handle interactions
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



# provider rules section
# make sure this version is in your codebase
def provider_rules_panel():
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
                s = min(v_start, v_end); e = max(v_start, v_end)
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
                try: clean.append(str(pd.to_datetime(tok).date()))
                except Exception: pass
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
    



def schedule_grid_view():
    st.subheader("Monthly Grid — Shifts × Days (one provider per cell)")
    
    import pandas as pd

    if not st.session_state.shift_types:
        st.info("No shift types configured.")
        return

    def tod_group_and_order(skey: str, sdef: Dict[str, Any]):
        # New order: Rounders, 7am Admitter, 10am Admitter, Night Shift, Bridge Shift, APP Shift
        if skey == "R12":          return "Rounders (07:00–19:00)", 1
        if skey == "A12":          return "7am Admitter (07:00–19:00)", 2
        if skey == "A10":          return "10am Admitter (10:00–22:00)", 3
        if skey == "N12":          return "Night Shift (19:00–07:00)", 4
        if skey == "NB":           return "Bridge Shift (23:00–07:00)", 5
        if skey == "APP":          return "APP Shift (07:00–19:00)", 6
        # Fallback for any other shift types
        start = parse_time(sdef["start"])
        if 5 <= start.hour < 12:   return "Day", 1
        if 12 <= start.hour < 18:  return "Evening", 2
        return "Night", 3

    def start_minutes(sdef):
        t = parse_time(sdef["start"])
        return t.hour * 60 + t.minute

    def _hex_to_rgb(h):
        h = (h or "").lstrip("#")
        if len(h) == 3: h = "".join([c*2 for c in h])
        try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except Exception: return (102,102,102)

    def _rgb_to_hue(r,g,b):
        import colorsys
        h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
        return int(h*360)

    def emoji_for_hex(hex_color: str) -> str:
        r,g,b = _hex_to_rgb(hex_color); hue = _rgb_to_hue(r,g,b)
        if hue < 15 or hue >= 345: return "🔴"
        if 15 <= hue < 40:         return "🟠"
        if 40 <= hue < 70:         return "🟡"
        if 70 <= hue < 170:        return "🟢"
        if 170 <= hue < 250:       return "🔵"
        if 250 <= hue < 320:       return "🟣"
        return "🟤"

    # month context - allow viewing any month that has events
    all_events = st.session_state.get("events", [])
    
    # Find all months that have events
    months_with_events = set()
    for e in all_events:
        try:
            d = pd.to_datetime(e["start"]).date()
            months_with_events.add((d.year, d.month))
        except Exception:
            continue
    
    if not months_with_events:
        st.info("No events found. Generate a schedule first.")
        return
    
    # Default to current month, but allow selection
    default_year, default_month = st.session_state.month.year, st.session_state.month.month
    if (default_year, default_month) not in months_with_events:
        # If current month has no events, pick the first available month
        default_year, default_month = sorted(months_with_events)[0]
    
    # Month selector
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", options=sorted(set(y for y, m in months_with_events)), index=sorted(set(y for y, m in months_with_events)).index(default_year))
    with col2:
        month_options = [m for y, m in months_with_events if y == year]
        month = st.selectbox("Month", options=month_options, index=month_options.index(default_month))
    
    days = make_month_days(year, month)
    day_cols = [str(d.day) for d in days]
    
    # Show summary of available months
    month_names = []
    for y, m in sorted(months_with_events):
        month_name = date(y, m, 1).strftime('%B %Y')
        month_names.append(month_name)
    
    st.caption(f"📅 Available months: {', '.join(month_names)}")
    st.caption(f"📊 Viewing: {date(year, month, 1).strftime('%B %Y')} ({len(days)} days)")

    stypes  = st.session_state.shift_types
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)

    # build row meta (one row per capacity slot)
    row_meta = []
    for s in stypes:
        skey = s["key"]; cap = int(cap_map.get(skey, 1))
        group_label, gorder = tod_group_and_order(skey, s)
        for slot in range(1, cap + 1):
            row_label = f"{skey} — {s['label']} (slot {slot})"
            row_meta.append({
                "row_label": row_label, "skey": skey, "sdef": s,
                "slot": slot, "group": group_label, "gorder": gorder
            })
    row_meta.sort(key=lambda r: (r["gorder"], start_minutes(r["sdef"]), r["skey"], r["slot"]))

    import pandas as pd
    row_labels = [rm["row_label"] for rm in row_meta]
    grid_raw = pd.DataFrame("", index=row_labels, columns=day_cols, dtype="object")
    color_tags = [emoji_for_hex(rm["sdef"].get("color")) for rm in row_meta]
    grid_raw.insert(0, "Color", color_tags)  # first column

    # fill from events (first empty slot per shift/day)
    rows_for_key = {}
    for rm in row_meta:
        rows_for_key.setdefault(rm["skey"], []).append(rm["row_label"])

    for e in st.session_state.events:
        ext = (e.get("extendedProps") or {}); skey = ext.get("shift_key")
        if not skey: continue
        try:
            d = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        # Filter to selected month
        if d.year != year or d.month != month:
            continue
        prov = (ext.get("provider") or "").strip().upper() or "UNASSIGNED"
        col = str(d.day)
        for row_label in rows_for_key.get(skey, []):
            if grid_raw.at[row_label, col] == "":
                grid_raw.at[row_label, col] = prov
                break

    # height to avoid vertical scroll
    height_px = min(2200, 110 + len(row_meta) * 38)


    hi = (st.session_state.get("highlight_provider", "") or "").strip().upper()
    enable_highlight = hi != ""
    edit_mode = st.toggle("Edit grid (disables highlighting)", value=False, disabled=not enable_highlight)

    if enable_highlight and not edit_mode:
        # Styled, read-only grid with light background highlight
        day_only_cols = [c for c in grid_raw.columns if c.isdigit()]

        def _style_fn(val):
            try:
                return (
                    "background-color:#fff3bf; color:#111111; font-weight:700;"
                    if str(val).strip().upper() == hi else ""
                )
            except Exception:
                return ""

        styled = grid_raw.style.applymap(_style_fn, subset=day_only_cols)
        st.dataframe(styled, use_container_width=True, height=height_px)
        st.caption(f"Highlighting cells for **{hi}**. Toggle *Edit grid* to make changes.")
    else:
        # Editable grid - separate providers by type
        physician_provs = sorted(
            st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()
        ) if not st.session_state.providers_df.empty else []
        app_provs = sorted(APP_PROVIDER_INITIALS)
        
        col_config = {"Color": st.column_config.TextColumn(disabled=True, help="Shift color tag")}
        try:
            for c in day_cols:
                # Determine which providers can be assigned to this column based on row metadata
                # Check what shift types are available in this column based on the grid rows
                shift_types_in_col = set()
                for rm in row_meta:
                    if rm["row_label"] in grid_raw.index:
                        shift_types_in_col.add(rm["skey"])
                
                # Set options based on shift types in this column
                if "APP" in shift_types_in_col and len(shift_types_in_col) == 1:
                    # If ONLY APP shifts are available, only APP providers can be assigned
                    options = [""] + app_provs
                    help_text = f"Assignments for day {c} (APP providers only)"
                elif "APP" not in shift_types_in_col:
                    # If NO APP shifts are available, only physician providers
                    options = [""] + physician_provs
                    help_text = f"Assignments for day {c} (Physicians only)"
                else:
                    # Mixed shift types - allow both provider types
                    all_providers = physician_provs + app_provs
                    options = [""] + sorted(all_providers)
                    help_text = f"Assignments for day {c} (All providers)"
                
                col_config[c] = st.column_config.SelectboxColumn(
                    options=options,
                    help=help_text
                )
        except Exception:
            pass

        edited_grid = st.data_editor(
            grid_raw,
            num_rows="fixed",
            use_container_width=True,
            height=height_px,
            column_config=col_config,
            key="grid_editor",
        )

        # Auto-apply when grid changes
        if edited_grid is not None and not edited_grid.equals(grid_raw):
            # Validate for double assignments first
            conflicts = validate_grid_for_double_assignments(edited_grid)
            if conflicts:
                st.error("❌ Double assignments detected:\n" + "\n".join(conflicts))
            else:
                # Apply changes automatically
                apply_grid_to_calendar(edited_grid, year, month, row_meta)
                st.success("✅ Grid changes applied automatically!")
                # Remove automatic rerun to improve performance
                # st.rerun()

        # Manual apply button (for backup)
        if st.button("Apply grid to calendar (Manual)"):
            apply_grid_to_calendar(edited_grid, year, month, row_meta)
            st.success("✅ Grid changes applied manually!")
            # Only rerun on manual apply if needed
            # st.rerun()
        
        # Save functionality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Month"):
                save_month_to_file(year, month)
        with col2:
            if st.button("📁 Load Saved Month"):
                load_month_from_file(year, month)
        
        # Shift Swapping Functionality
        st.subheader("🔄 Shift Swapping")
        st.caption("Swap shifts between providers for the selected month")
        
        # Get all providers for swapping
        all_providers = sorted(physician_provs + app_provs)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            provider1 = st.selectbox("Provider 1", options=all_providers, key="swap_provider1")
        with col2:
            day1 = st.selectbox("Day 1", options=day_cols, key="swap_day1")
        with col3:
            provider2 = st.selectbox("Provider 2", options=all_providers, key="swap_provider2")
        with col4:
            day2 = st.selectbox("Day 2", options=day_cols, key="swap_day2")
        
        if st.button("🔄 Execute Shift Swap", help="Swap shifts between the selected providers and days"):
            if provider1 == provider2 and day1 == day2:
                st.warning("⚠️ Please select different providers or days for swapping")
            else:
                success = execute_shift_swap(provider1, int(day1), provider2, int(day2), year, month)
                if success:
                    st.success("✅ Shift swap completed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Shift swap failed. Check the console for details.")

def apply_grid_to_calendar(edited_grid, target_year, target_month, row_meta=None):
    """Apply grid changes to calendar events"""
    # Always normalize existing events before processing
    st.session_state.events = events_for_calendar(st.session_state.get("events", []))

    sdefs = {s["key"]: s for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())}
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps = st.session_state.get("provider_caps", {})
    prov_rules = st.session_state.get("provider_rules", {})
    global_rules = get_global_rules()
    base_max = recommended_max_shifts_for_month()
    mbx = getattr(global_rules, "max_block_size", None)
    
    # If row_meta is not provided, we need to build it
    if row_meta is None:
        stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
        row_meta = []
        for s in stypes:
            skey = s["key"]
            cap = int(cap_map.get(skey, 1))
            for slot in range(1, cap + 1):
                row_label = f"{skey} — {s['label']} (slot {slot})"
                row_meta.append({
                    "row_label": row_label, "skey": skey, "sdef": s,
                    "slot": slot
                })

    # keep comments by (date, shift_key, provider)
    comments_by_key = {}
    for e in st.session_state.events:
        ext = (e.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey or skey not in sdefs:
            continue
        try:
            d0 = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        if d0.year == target_year and d0.month == target_month:
            prov0 = (ext.get("provider") or "").strip().upper()
            comments_by_key[(d0, skey, prov0)] = list(st.session_state.comments.get(e["id"], []))

    # identify grid-controlled events for this month
    def is_grid_event(E: dict) -> bool:
        ext = (E.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey or skey not in sdefs:
            return False
        try:
            d0 = pd.to_datetime(E["start"]).date()
        except Exception:
            return False
        return d0.year == target_year and d0.month == target_month

    preserved = [E for E in st.session_state.events if not is_grid_event(E)]

    new_events = []
    seen_day_provider = set()  # {(date, provider)}
    conflicts = []

    # helpers that look at what's been added so far (new_events)
    def day_shift_count(dy, key):
        return sum(1 for E in new_events
                   if pd.to_datetime(E["start"]).date() == dy and
                      (E.get("extendedProps") or {}).get("shift_key") == key)

    def provider_has_shift_on_day(provider, dy):
        return any(
            (E.get("extendedProps") or {}).get("provider", "").upper() == provider and
            pd.to_datetime(E["start"]).date() == dy
            for E in new_events
        )

    def provider_days(provider):
        return {pd.to_datetime(E["start"]).date()
                for E in new_events
                if (E.get("extendedProps") or {}).get("provider", "").upper() == provider}

    def left_run_len(days_set, d0):
        run = 0; cur = d0 - timedelta(days=1)
        while cur in days_set:
            run += 1; cur -= timedelta(days=1)
        return run

    def right_run_len(days_set, d0):
        run = 0; cur = d0 + timedelta(days=1)
        while cur in days_set:
            run += 1; cur += timedelta(days=1)
        return run

    def total_block_len_if_assigned(provider, d0):
        ds = provider_days(provider)
        return left_run_len(ds, d0) + 1 + right_run_len(ds, d0)

    # live counters for per-provider totals/nights in this month build
    counts = {}
    nights = {}

    row_to_key = {rm["row_label"]: rm["skey"] for rm in row_meta}
    day_only_cols = [c for c in edited_grid.columns if c.isdigit()]

    for row_label in edited_grid.index:
        skey = row_to_key.get(row_label)
        if not skey:
            continue
        sdef = sdefs.get(skey)
        if not sdef:
            continue

        for col in day_only_cols:
            prov = edited_grid.at[row_label, col]
            prov = ("" if prov is None else str(prov)).strip().upper()
            if not prov:
                continue

            day_date = date(target_year, target_month, int(col))

            # one shift per provider per day
            key_dp = (day_date, prov)
            if key_dp in seen_day_provider or provider_has_shift_on_day(prov, day_date):
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (duplicate same-day assignment; skipped)")
                continue

            # per-shift daily capacity (in case capacity < number of rows filled)
            if day_shift_count(day_date, skey) >= int(cap_map.get(skey, 1)):
                conflicts.append(f"{day_date:%Y-%m-%d} {skey} over capacity; skipped")
                continue

            # hard block: unavailable (vacation or specific date)
            if is_provider_unavailable_on_date(prov, day_date):
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (on vacation/unavailable; skipped)")
                continue

            # eligibility (allowed shift types)
            allowed = prov_caps.get(prov, [])
            if allowed and skey not in allowed:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} not eligible for {skey}; skipped")
                continue

            # effective max shifts (month default, minus 3 if any vacation in month)
            pr = prov_rules.get(prov, {}) or {}
            eff_max = pr.get("max_shifts", base_max)
            if _provider_has_vacation_in_month(pr):
                eff_max = max(0, (eff_max or 0) - 3)

            counts.setdefault(prov, 0)
            nights.setdefault(prov, 0)

            if eff_max is not None and counts[prov] + 1 > eff_max:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} exceeds max shifts {eff_max}; skipped")
                continue

            # max nights
            max_nights = pr.get("max_nights", global_rules.max_nights_per_provider)
            if skey == "N12" and max_nights is not None and nights[prov] + 1 > max_nights:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} exceeds max nights {max_nights}; skipped")
                continue

            # max block size (if set)
            if mbx and mbx > 0 and total_block_len_if_assigned(prov, day_date) > mbx:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} would exceed max block {mbx}; skipped")
                continue

            # build event
            def _parse(hhmm: str):
                hh, mm = hhmm.split(":")
                return time(int(hh), int(mm))
            start_dt = datetime.combine(day_date, _parse(sdef["start"]))
            end_dt   = datetime.combine(day_date, _parse(sdef["end"]))
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            eid = str(uuid.uuid4())
            ev = {
                "id": eid,
                "title": f"{sdef['label']} — {prov}",
                "start": start_dt.isoformat(),
                "end":   end_dt.isoformat(),
                "allDay": False,
                "backgroundColor": sdef.get("color"),
                "extendedProps": {"provider": prov, "shift_key": skey, "label": sdef["label"]},
            }
            new_events.append(ev)
            seen_day_provider.add(key_dp)

            # carry comments forward if any mapping existed
            k = (day_date, skey, prov)
            if k in st.session_state.comments:
                st.session_state.comments[eid] = st.session_state.comments[k]
            elif k in comments_by_key:
                st.session_state.comments[eid] = comments_by_key[k]

            # update counters
            counts[prov] += 1
            if skey == "N12":
                nights[prov] += 1

    st.session_state.events = events_for_calendar(preserved + new_events)

    if conflicts:
        st.warning("Some cells were skipped:\n- " + "\n- ".join(conflicts))
    else:
        st.success("Applied grid to calendar.")

def save_month_to_file(year: int, month: int):
    """Save the current month's events to a file"""
    try:
        # Get events for the specific month
        month_events = []
        for e in st.session_state.get("events", []):
            try:
                d = pd.to_datetime(e["start"]).date()
                if d.year == year and d.month == month:
                    month_events.append(e)
            except Exception:
                continue
        
        # Create filename
        filename = f"saved_month_{year}_{month:02d}.json"
        
        # Save to session state and file
        month_key = f"{year}_{month:02d}"
        st.session_state["saved_months"][month_key] = {
            "events": month_events,
            "comments": {eid: comments for eid, comments in st.session_state.get("comments", {}).items() 
                        if any(pd.to_datetime(e["start"]).date().year == year and 
                               pd.to_datetime(e["start"]).date().month == month 
                               for e in month_events if e["id"] == eid)},
            "saved_at": datetime.now().isoformat()
        }
        
        # Also save to disk
        import os
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(st.session_state["saved_months"][month_key], f, indent=2)
        
        st.success(f"✅ Saved {len(month_events)} events for {date(year, month, 1).strftime('%B %Y')} to {filename}")
        
    except Exception as e:
        st.error(f"❌ Failed to save month: {e}")

def load_month_from_file(year: int, month: int):
    """Load a saved month's events"""
    try:
        month_key = f"{year}_{month:02d}"
        saved_data = st.session_state.get("saved_months", {}).get(month_key)
        
        if not saved_data:
            # Try to load from disk
            import os
            data_dir = os.path.join(os.getcwd(), "data")
            filename = f"saved_month_{year}_{month:02d}.json"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    saved_data = json.load(f)
                st.session_state["saved_months"][month_key] = saved_data
            else:
                st.warning(f"❌ No saved data found for {date(year, month, 1).strftime('%B %Y')}")
                return
        
        # Remove existing events for this month
        existing_events = st.session_state.get("events", [])
        filtered_events = []
        for e in existing_events:
            try:
                d = pd.to_datetime(e["start"]).date()
                if not (d.year == year and d.month == month):
                    filtered_events.append(e)
            except Exception:
                filtered_events.append(e)
        
        # Add saved events
        saved_events = saved_data.get("events", [])
        st.session_state["events"] = filtered_events + saved_events
        
        # Restore comments
        saved_comments = saved_data.get("comments", {})
        st.session_state["comments"].update(saved_comments)
        
        st.success(f"✅ Loaded {len(saved_events)} events for {date(year, month, 1).strftime('%B %Y')}")
        
    except Exception as e:
        st.error(f"❌ Failed to load month: {e}")



# -------------------------
# App entry
# -------------------------
def main():
    init_session_state()
    

    
    # Main header
    st.title("🏥 Hospitalist Monthly Scheduler")
    
    # Provider status indicator
    if st.session_state.get("providers_loaded", False) and not st.session_state.providers_df.empty:
        provider_count = len(st.session_state.providers_df)
        st.success(f"✅ {provider_count} providers loaded and ready")
    else:
        st.error("❌ No providers loaded. Please go to the Providers tab to load providers.")
    
    st.markdown("---")
    
    # Navigation tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Calendar", "⚙️ Settings", "👥 Providers", "📊 Grid View", "📅 Google Sync", "📝 Requests"])
    
    with tab1:
        # Calendar tab - main scheduling interface
        st.header("Monthly Calendar")
        
        # Top controls in a clean layout
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            # Ensure providers are loaded and get the list
            if not st.session_state.providers_df.empty:
                physician_provs = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist())
                app_provs = sorted(APP_PROVIDER_INITIALS)
                
                # Filter out APP providers from the physician list
                physician_providers = [p for p in physician_provs if p not in app_provs]
                
                # Create provider options with separators
                provider_options = ["(Select Provider)"]
                if physician_providers:
                    provider_options.append("--- Physicians ---")
                    provider_options.extend(physician_providers)
                if app_providers:
                    provider_options.append("--- APPs ---")
                    provider_options.extend(app_provs)
                
                default = st.session_state.highlight_provider if st.session_state.highlight_provider in provider_options else "(All providers)"
                idx = provider_options.index(default) if default in provider_options else 0
                sel = st.selectbox("Highlight provider", options=provider_options, index=idx)
                st.session_state.highlight_provider = "" if sel == "(All providers)" else sel
            else:
                st.warning("No providers loaded. Please check the Providers tab.")
                st.session_state.highlight_provider = ""
        with col2:
            st.caption(f"📅 Currently viewing: {st.session_state.month.strftime('%B %Y')}")
        with col3:
            st.caption("💡 Use navigation buttons above calendar to change month")
        with col4:
            st.caption("🔄 Generate button creates schedules for the displayed month")
        
        # Generation info
        if st.session_state.get("generation_count", 0) > 0:
            st.caption(f"📊 Generated {st.session_state.generation_count} schedule(s) so far. Each generation creates a different schedule!")
        
        # Action buttons
        g1, g2, g3 = st.columns(3)
        with g1:
            if st.button("🔄 Generate Draft", help="Generate schedule for the displayed month"):
                if st.session_state.providers_df.empty:
                    st.error("❌ No providers loaded! Please go to the Providers tab and load providers first.")
                else:
                    providers = st.session_state.providers_df["initials"].tolist()
                    # Add APP providers to the list (avoiding duplicates)
                    csv_providers_upper = [p.upper() for p in providers]
                    for app_provider in APP_PROVIDER_INITIALS:
                        if app_provider.upper() not in csv_providers_upper:
                            providers.append(app_provider)
                    if not providers:
                        st.error("❌ Provider list is empty! Please add providers in the Providers tab.")
                    else:
                        # Count how many APP providers were actually added
                        app_providers_added = sum(1 for app_provider in APP_PROVIDER_INITIALS 
                                                if app_provider.upper() not in csv_providers_upper)
                        st.info(f"🔄 Generating schedule for {len(providers)} providers (including {app_providers_added} additional APP providers)...")
                        rules = RuleConfig(**st.session_state.rules)
                        # Generate days for the current month only
                        days = make_month_days(st.session_state.month.year, st.session_state.month.month)
                        # Add randomness seed for different schedules on each generation
                        import random
                        random_seed = random.randint(1, 10000)
                        random.seed(random_seed)
                        # Generate new events using the greedy algorithm
                        new_events = assign_greedy(providers, days, st.session_state.shift_types, rules)
                        # Convert SEvent objects to dictionary format for calendar
                        st.session_state.events = [_event_to_dict(e) for e in new_events]
                        st.session_state.comments = {}
                        st.session_state.generation_count += 1
                        st.success(f"✅ Draft schedule generated for {st.session_state.month.strftime('%B %Y')} (use navigation buttons above calendar to change month) with {len(new_events)} events! (Generation #{st.session_state.generation_count}, Seed: {random_seed})")
        with g2:
            if st.button("✅ Validate Schedule", help="Check for rule violations"):
                rules = RuleConfig(**st.session_state.rules)
                evs = [SEvent(**{**e, "start": datetime.fromisoformat(e["start"]), "end": datetime.fromisoformat(e["end"])}) for e in st.session_state.events]
                
                # Show shift counts for debugging
                provider_counts = {}
                for ev in evs:
                    provider = (ev.extendedProps.get("provider") or "").strip().upper()
                    if provider:
                        if provider not in provider_counts:
                            provider_counts[provider] = 0
                        provider_counts[provider] += 1
                
                viols = validate_rules(evs, rules)
                
                # Create a comprehensive validation results interface
                st.markdown("## 📋 Schedule Validation Results")
                
                # Summary metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", len(evs))
                with col2:
                    st.metric("Providers", len(provider_counts))
                with col3:
                    st.metric("Violations", len(viols) if viols else 0)
                with col4:
                    status = "✅ Valid" if not viols else "❌ Issues Found"
                    st.metric("Status", status)
                
                # Create tabs for different views
                if viols:
                    tab_summary, tab_details, tab_stats = st.tabs(["📊 Summary", "🔍 Details", "📈 Statistics"])
                    
                    with tab_summary:
                        st.subheader("🚨 Violation Summary")
                        
                        # Quick violation overview
                        violation_types = {}
                        for provider, violations in viols.items():
                            for violation in violations:
                                # Extract violation type from message
                                if "exceeds max" in violation.lower():
                                    violation_types["Max Shifts"] = violation_types.get("Max Shifts", 0) + 1
                                elif "below minimum" in violation.lower():
                                    violation_types["Min Shifts"] = violation_types.get("Min Shifts", 0) + 1
                                elif "insufficient rest" in violation.lower():
                                    violation_types["Rest Periods"] = violation_types.get("Rest Periods", 0) + 1
                                elif "unavailable" in violation.lower():
                                    violation_types["Unavailable Dates"] = violation_types.get("Unavailable Dates", 0) + 1
                                else:
                                    violation_types["Other"] = violation_types.get("Other", 0) + 1
                        
                        # Display violation types
                        if violation_types:
                            cols = st.columns(len(violation_types))
                            for i, (violation_type, count) in enumerate(violation_types.items()):
                                with cols[i]:
                                    st.metric(violation_type, count)
                        
                        # Providers with most violations
                        provider_violation_counts = {p: len(v) for p, v in viols.items()}
                        if provider_violation_counts:
                            st.subheader("⚠️ Providers with Issues")
                            sorted_providers = sorted(provider_violation_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            for provider, count in sorted_providers[:5]:  # Show top 5
                                st.warning(f"**{provider}**: {count} violation(s)")
                    
                    with tab_details:
                        st.subheader("🔍 Detailed Violations")
                        
                        # Group violations by provider in expandable sections
                        for provider, violations in viols.items():
                            with st.expander(f"**{provider}** ({len(violations)} violations)", expanded=False):
                                for i, violation in enumerate(violations, 1):
                                    # Color code violations
                                    if "exceeds max" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    elif "below minimum" in violation.lower():
                                        st.warning(f"{i}. {violation}")
                                    elif "insufficient rest" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    elif "unavailable" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    else:
                                        st.info(f"{i}. {violation}")
                    
                    with tab_stats:
                        st.subheader("📈 Provider Statistics")
                        
                        # Calculate statistics
                        stats_data = calculate_provider_statistics(evs)
                        provider_stats = stats_data["provider_stats"]
                        
                        if provider_stats:
                            # 1. Total shifts per provider
                            st.subheader("📊 Total Shifts per Provider")
                            shifts_data = {provider: data["total_shifts"] for provider, data in provider_stats.items()}
                            shifts_df = pd.DataFrame(list(shifts_data.items()), columns=["Provider", "Total Shifts"])
                            shifts_df = shifts_df.sort_values("Total Shifts", ascending=False)
                            
                            st.dataframe(shifts_df, use_container_width=True)
                            
                            # 2. Weekend shifts per provider
                            st.subheader("🌅 Weekend Shifts per Provider")
                            weekend_data = {provider: data["weekend_shifts"] for provider, data in provider_stats.items()}
                            weekend_df = pd.DataFrame(list(weekend_data.items()), columns=["Provider", "Weekend Shifts"])
                            weekend_df = weekend_df.sort_values("Weekend Shifts", ascending=False)
                            
                            st.dataframe(weekend_df, use_container_width=True)
                            
                            # 3. Coverage gaps
                            st.subheader("⚠️ Coverage Gaps")
                            gaps = identify_coverage_gaps(evs, st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES), 
                                                        st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
                            
                            # Debug information
                            st.caption(f"📊 **Debug Info**: Analyzing {len(evs)} events across {len(set(event.start.date() for event in evs))} unique days")
                            
                            if gaps:
                                # Group gaps by date for better display
                                gaps_by_date = {}
                                for gap in gaps:
                                    date_str = gap["date"].strftime("%Y-%m-%d")
                                    if date_str not in gaps_by_date:
                                        gaps_by_date[date_str] = []
                                    gaps_by_date[date_str].append(gap)
                                
                                # Create a summary table
                                summary_data = []
                                for date_str, date_gaps in gaps_by_date.items():
                                    # Get the date object from the first gap in this date group
                                    date_obj = date_gaps[0]["date"]
                                    day_name = date_obj.strftime("%A")
                                    total_shortage = sum(gap["shortage"] for gap in date_gaps)
                                    missing_shifts = [f"{gap['shift_type']} (-{gap['shortage']})" for gap in date_gaps]
                                    summary_data.append({
                                        "Date": date_str,
                                        "Day": day_name,
                                        "Total Shortage": total_shortage,
                                        "Missing Shifts": ", ".join(missing_shifts)
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                summary_df = summary_df.sort_values("Date")
                                
                                st.dataframe(summary_df, use_container_width=True)
                                
                                # Summary of gaps
                                total_gaps = len(gaps)
                                total_shortage = sum(gap["shortage"] for gap in gaps)
                                st.info(f"📋 **Coverage Summary**: {len(gaps_by_date)} days have insufficient coverage ({total_shortage} total missing providers)")
                                
                                # Show detailed breakdown
                                with st.expander("🔍 Detailed Coverage Analysis"):
                                    st.write("**Shift Type Breakdown:**")
                                    shift_breakdown = pd.DataFrame(gaps).groupby("shift_type").agg({
                                        "expected": "sum",
                                        "actual": "sum", 
                                        "shortage": "sum"
                                    }).reset_index()
                                    st.dataframe(shift_breakdown, use_container_width=True)
                                    
                                    # Show detailed gaps table
                                    st.write("**Detailed Gaps:**")
                                    gaps_df = pd.DataFrame(gaps)
                                    gaps_df["Date"] = gaps_df["date"].dt.strftime("%Y-%m-%d")
                                    gaps_df["Day"] = gaps_df["date"].dt.strftime("%A")
                                    gaps_df = gaps_df[["Date", "Day", "shift_type", "expected", "actual", "shortage"]]
                                    gaps_df.columns = ["Date", "Day", "Shift Type", "Expected", "Actual", "Shortage"]
                                    gaps_df = gaps_df.sort_values("Date")
                                    st.dataframe(gaps_df, use_container_width=True)
                            else:
                                st.success("✅ **No coverage gaps detected!** All shifts are properly staffed.")
                        else:
                            st.warning("⚠️ No provider data available for statistics.")
                else:
                    # No violations - show success with details
                    st.success("🎉 **Schedule is Valid!** No rule violations detected.")
                    
                    # Show statistics even when no violations
                    st.subheader("📈 Provider Statistics")
                    
                    # Calculate statistics
                    stats_data = calculate_provider_statistics(evs)
                    provider_stats = stats_data["provider_stats"]
                    
                    if provider_stats:
                        # 1. Total shifts per provider
                        st.subheader("📊 Total Shifts per Provider")
                        shifts_data = {provider: data["total_shifts"] for provider, data in provider_stats.items()}
                        shifts_df = pd.DataFrame(list(shifts_data.items()), columns=["Provider", "Total Shifts"])
                        shifts_df = shifts_df.sort_values("Total Shifts", ascending=False)
                        
                        st.dataframe(shifts_df, use_container_width=True)
                        
                        # 2. Weekend shifts per provider
                        st.subheader("🌅 Weekend Shifts per Provider")
                        weekend_data = {provider: data["weekend_shifts"] for provider, data in provider_stats.items()}
                        weekend_df = pd.DataFrame(list(weekend_data.items()), columns=["Provider", "Weekend Shifts"])
                        weekend_df = weekend_df.sort_values("Weekend Shifts", ascending=False)
                        
                        st.dataframe(weekend_df, use_container_width=True)
                        
                        # 3. Coverage gaps
                        st.subheader("⚠️ Coverage Gaps")
                        gaps = identify_coverage_gaps(evs, st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES), 
                                                    st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
                        
                        # Debug information
                        st.caption(f"📊 **Debug Info**: Analyzing {len(evs)} events across {len(set(event.start.date() for event in evs))} unique days")
                        
                        if gaps:
                            # Group gaps by date for better display
                            gaps_by_date = {}
                            for gap in gaps:
                                date_str = gap["date"].strftime("%Y-%m-%d")
                                if date_str not in gaps_by_date:
                                    gaps_by_date[date_str] = []
                                gaps_by_date[date_str].append(gap)
                            
                            # Create a summary table
                            summary_data = []
                            for date_str, date_gaps in gaps_by_date.items():
                                # Get the date object from the first gap in this date group
                                date_obj = date_gaps[0]["date"]
                                day_name = date_obj.strftime("%A")
                                total_shortage = sum(gap["shortage"] for gap in date_gaps)
                                missing_shifts = [f"{gap['shift_type']} (-{gap['shortage']})" for gap in date_gaps]
                                summary_data.append({
                                    "Date": date_str,
                                    "Day": day_name,
                                    "Total Shortage": total_shortage,
                                    "Missing Shifts": ", ".join(missing_shifts)
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            summary_df = summary_df.sort_values("Date")
                            
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Summary of gaps
                            total_gaps = len(gaps)
                            total_shortage = sum(gap["shortage"] for gap in gaps)
                            st.info(f"📋 **Coverage Summary**: {len(gaps_by_date)} days have insufficient coverage ({total_shortage} total missing providers)")
                            
                            # Show detailed breakdown
                            with st.expander("🔍 Detailed Coverage Analysis"):
                                st.write("**Shift Type Breakdown:**")
                                shift_breakdown = pd.DataFrame(gaps).groupby("shift_type").agg({
                                    "expected": "sum",
                                    "actual": "sum", 
                                    "shortage": "sum"
                                }).reset_index()
                                st.dataframe(shift_breakdown, use_container_width=True)
                                
                                # Show detailed gaps table
                                st.write("**Detailed Gaps:**")
                                gaps_df = pd.DataFrame(gaps)
                                gaps_df["Date"] = gaps_df["date"].dt.strftime("%Y-%m-%d")
                                gaps_df["Day"] = gaps_df["date"].dt.strftime("%A")
                                gaps_df = gaps_df[["Date", "Day", "shift_type", "expected", "actual", "shortage"]]
                                gaps_df.columns = ["Date", "Day", "Shift Type", "Expected", "Actual", "Shortage"]
                                gaps_df = gaps_df.sort_values("Date")
                                st.dataframe(gaps_df, use_container_width=True)
                        else:
                            st.success("✅ **No coverage gaps detected!** All shifts are properly staffed.")
                    else:
                        st.warning("⚠️ No provider data available for statistics.")
        with g3:
            if st.button("🗑️ Clear All", help="Clear all events"):
                st.session_state.events = []
                st.session_state.comments = {}
                st.success("All events cleared!")
        
        # Calendar display
        render_calendar()
    
    with tab2:
        # Settings tab - global rules and shift types
        st.header("Global Settings")
        
        # Global rules section
        st.subheader("📋 Scheduling Rules")
        
        # Info about dynamic minimum shifts and enhanced features
        st.info("💡 **Enhanced Features**:\n"
                "• **Dynamic Minimum Shifts**: Automatically enforced based on month length\n"
                "• **Shift Consistency**: Providers stay on same shift type within blocks\n"
                "• **Random Generation**: Each generate creates a different schedule\n"
                "• **Smart Month Generation**: Generates for the month currently displayed in calendar")
        
        rc = RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))
        
        col1, col2 = st.columns(2)
        with col1:
            rc.max_shifts_per_provider = st.number_input("Max shifts/provider", 1, 31, value=int(rc.max_shifts_per_provider))
            rc.min_rest_days_between_shifts = st.number_input("Min rest (days) between shifts", min_value=0.0, max_value=14.0, step=0.5, value=float(getattr(rc, "min_rest_days_between_shifts", 1.0)))
            rc.min_block_size = st.number_input("Preferred block size (days)", 1, 7, value=int(rc.min_block_size))
        with col2:
            rc.require_at_least_one_weekend = st.checkbox("Require at least one weekend shift", value=bool(rc.require_at_least_one_weekend))
            limit_nights = st.checkbox("Limit 7pm–7am (N12) nights per provider", value=st.session_state.rules.get("max_nights_per_provider", 6) is not None)
            if limit_nights:
                default_nights = int(st.session_state.rules.get("max_nights_per_provider", 6) or 0)
                rc.max_nights_per_provider = st.number_input("Max nights/provider", 0, 31, value=default_nights)
            else:
                rc.max_nights_per_provider = None
        
        st.session_state.rules = rc.model_dump()
        
        # Shift types section
        st.subheader("🕐 Shift Types")
        st.caption("Edit labels, times, and colors for each shift type.")
        for i, s in enumerate(st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())):
            with st.expander(f"{s['label']} ({s['key']})", expanded=False):
                s["label"] = st.text_input("Label", value=s["label"], key=f"s_lbl_{i}")
                s["start"] = st.text_input("Start (HH:MM)", value=s["start"], key=f"s_st_{i}")
                s["end"]   = st.text_input("End (HH:MM)",   value=s["end"],   key=f"s_en_{i}")
                s["color"] = st.color_picker("Color", value=s.get("color", "#3388ff"), key=f"s_co_{i}")
        
        # Daily capacities section
        st.subheader("📊 Daily Shift Capacities")
        if st.button("Reset to default capacities"):
            st.session_state["shift_capacity"] = DEFAULT_SHIFT_CAPACITY.copy()
            st.toast("Capacities reset to defaults.", icon="♻️")
        
        cap_map = dict(st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
        for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy()):
            key = s["key"]; label = s["label"]
            default_cap = int(cap_map.get(key, DEFAULT_SHIFT_CAPACITY.get(key, 1)))
            cap_map[key] = int(st.number_input(f"{label} ({key}) capacity/day", min_value=0, max_value=50, value=default_cap, key=f"cap_{key}"))
        st.session_state["shift_capacity"] = cap_map
    
    with tab3:
        # Providers tab - manage provider roster and individual rules
        st.header("Provider Management")
        
        # Provider roster management
        st.subheader("👥 Provider Roster")
        current_list = st.session_state.providers_df["initials"].astype(str).tolist()
        st.caption(f"Currently loaded: {len(current_list)} providers")
        
        # Add a button to load default providers if none are loaded
        if len(current_list) == 0:
            st.warning("No providers loaded. Please load default providers or add providers manually.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Default Providers"):
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
                    st.success(f"Loaded {len(PROVIDER_INITIALS_DEFAULT)} default providers!")
                    st.rerun()
            with col2:
                if st.button("Load from CSV"):
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
                                st.success(f"Loaded {len(providers_df)} providers from CSV!")
                                st.rerun()
                            else:
                                st.error("CSV file is empty or has no valid data.")
                        else:
                            st.error("IMIS_initials.csv file not found.")
                    except Exception as e:
                        st.error(f"Failed to load CSV: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("➕ Add Providers", expanded=False):
                new_one = st.text_input("Add single provider (initials)", key="add_single_init")
                if st.button("Add", key="btn_add_single"):
                    cand = _normalize_initials_list([new_one])
                    if not cand:
                        st.warning("Enter initials to add.")
                    else:
                        initial = list(cand)[0]
                        if initial in current_list:
                            st.info(f"{initial} is already in the list.")
                        else:
                            st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(current_list + [initial])})
                            st.toast(f"Added {initial}", icon="✅")
                
                st.markdown("---")
                batch = st.text_area("Add multiple (comma/space/newline separated)", key="add_batch_area")
                if st.button("Add batch", key="btn_add_batch"):
                    tokens = _normalize_initials_list(batch.replace(",", "\n").split())
                    if not tokens:
                        st.warning("Nothing to add.")
                    else:
                        merged = _normalize_initials_list(current_list + list(tokens))
                        st.session_state.providers_df = pd.DataFrame({"initials": merged})
                        st.toast(f"Added {len(merged) - len(current_list)} new provider(s).", icon="✅")
        
        with col2:
            with st.expander("➖ Remove Providers", expanded=False):
                to_remove = st.multiselect("Select providers to remove", options=current_list, key="rm_multi")
                if st.button("Remove selected", key="btn_rm"):
                    if not to_remove:
                        st.info("No providers selected.")
                    else:
                        remaining = [p for p in current_list if p not in set(to_remove)]
                        st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(remaining)})
                        st.session_state["provider_caps"] = {k: v for k, v in st.session_state.provider_caps.items() if k in remaining}
                        st.toast(f"Removed {len(to_remove)} provider(s).", icon="🗑️")
        
        # Provider-specific rules
        st.subheader("⚙️ Provider-Specific Rules")
        provider_selector()
        provider_rules_panel()

    with tab4:
        # Grid view tab
        st.header("📊 Schedule Grid View")
        st.caption("Edit assignments directly in the grid below")
        schedule_grid_view()

    with tab5:
        # Google Calendar Sync tab
        provider_google_calendar_sync()

    with tab6:
        # Provider Requests tab
        st.header("📝 Provider Requests")
        st.caption("Providers can submit vacation requests, blackout dates, and shift swap requests")
        provider_requests_panel()

def execute_shift_swap(provider1: str, day1: int, provider2: str, day2: int, year: int, month: int) -> bool:
    """Execute a shift swap between two providers on different days"""
    try:
        # Normalize events
        st.session_state.events = events_for_calendar(st.session_state.get("events", []))
        
        # Find events for the specified days and providers
        events_to_swap = []
        target_events = []
        
        for event in st.session_state.events:
            ext = event.get("extendedProps", {})
            event_provider = ext.get("provider", "").strip().upper()
            try:
                event_date = pd.to_datetime(event["start"]).date()
            except Exception:
                continue
                
            if event_date.year == year and event_date.month == month:
                if event_date.day == day1 and event_provider == provider1:
                    events_to_swap.append(event)
                elif event_date.day == day2 and event_provider == provider2:
                    target_events.append(event)
        
        if not events_to_swap or not target_events:
            st.error(f"No shifts found for the specified providers and days")
            return False
        
        # Swap the providers
        for event in events_to_swap:
            event["extendedProps"]["provider"] = provider2
            event["title"] = event["title"].replace(provider1, provider2)
        
        for event in target_events:
            event["extendedProps"]["provider"] = provider1
            event["title"] = event["title"].replace(provider2, provider1)
        
        return True
        
    except Exception as e:
        st.error(f"Error during shift swap: {e}")
        return False

def provider_google_calendar_sync():
    """Allow each provider to sync their shifts to their own Google Calendar."""
    st.subheader("👤 Provider Google Calendar Sync")
    st.caption("Each provider can connect to their own Google Calendar and sync their shifts.")
    
    # Get all providers
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded. Please load providers first.")
        return
    
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
        "Select Provider to Sync",
        options=provider_options,
        key="provider_sync_select"
    )
    
    if selected_provider == "(Select Provider)" or selected_provider.startswith("---"):
        st.info("Please select a provider to sync their shifts to Google Calendar.")
        return
    
    # Initialize provider-specific session state
    provider_key = f"gcal_provider_{selected_provider}"
    if provider_key not in st.session_state:
        st.session_state[provider_key] = {
            "connected": False,
            "calendar_id": "primary",
            "calendar_name": "Primary Calendar"
        }
    
    provider_state = st.session_state[provider_key]
    
    # Connect to Google Calendar for this provider
    svc = None
    if st.button(f"Connect {selected_provider}'s Google Calendar", key=f"connect_{selected_provider}"):
        svc = get_gcal_service()
        if svc:
            provider_state["connected"] = True
            st.success(f"Connected {selected_provider} to Google Calendar.")
        else:
            st.error("Failed to connect to Google Calendar.")
    
    # Try to reuse previous connection
    if provider_state.get("connected"):
        svc = get_gcal_service()
    
    if not svc:
        st.caption(f"Click **Connect {selected_provider}'s Google Calendar** to authenticate.")
        return
    
    # Choose calendar for this provider
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    
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
        key=f"calendar_{selected_provider}"
    )
    sel_id = cal_ids[cal_labels.index(sel_label)]
    provider_state["calendar_id"] = sel_id
    provider_state["calendar_name"] = sel_label
    
    st.caption(f"Target: **{sel_label}** for {selected_provider}")
    
    # Filter events for this provider
    provider_events = []
    for event in st.session_state.get("events", []):
        ext = event.get("extendedProps", {})
        event_provider = (ext.get("provider") or "").strip().upper()
        if event_provider == selected_provider:
            provider_events.append(event)
    
    if not provider_events:
        st.info(f"No shifts found for {selected_provider}.")
        return
    
    # Show provider's shifts summary
    st.subheader(f"📊 {selected_provider}'s Shifts Summary")
    shifts_by_month = {}
    for event in provider_events:
        try:
            event_date = pd.to_datetime(event["start"]).date()
            month_key = (event_date.year, event_date.month)
            if month_key not in shifts_by_month:
                shifts_by_month[month_key] = []
            shifts_by_month[month_key].append(event)
        except Exception:
            continue
    
    # Display shifts by month
    for (year, month), month_events in sorted(shifts_by_month.items()):
        month_name = date(year, month, 1).strftime('%B %Y')
        st.write(f"**{month_name}**: {len(month_events)} shifts")
    
    # Sync options
    st.subheader(f"🔄 Sync Options for {selected_provider}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Sync {selected_provider}'s Shifts → Google", key=f"sync_{selected_provider}"):
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
            st.success(f"Synced {selected_provider}: created {created}, updated {updated} events")
    
    with col2:
        if st.button(f"Remove {selected_provider}'s Events from Google", key=f"remove_{selected_provider}"):
            removed = 0
            # Get all local event IDs for this provider
            local_ids = {event["id"] for event in provider_events}
            
            # Find and remove events from Google Calendar
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id,
                    singleEvents=True,
                    showDeleted=False,
                    pageToken=page_token
                ).execute()
                
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            
            st.success(f"Removed {removed} events for {selected_provider} from Google Calendar")
    
    # Show sync status
    st.subheader(f"📋 Sync Status for {selected_provider}")
    st.info(f"**Connected**: {provider_state['connected']}")
    st.info(f"**Calendar**: {provider_state['calendar_name']}")
    st.info(f"**Total Shifts**: {len(provider_events)}")
    
    # Quick sync for current month only
    st.subheader(f"📅 Quick Sync - Current Month")
    current_month_events = []
    current_year = st.session_state.month.year
    current_month = st.session_state.month.month
    
    for event in provider_events:
        try:
            event_date = pd.to_datetime(event["start"]).date()
            if event_date.year == current_year and event_date.month == current_month:
                current_month_events.append(event)
        except Exception:
            continue
    
    if current_month_events:
        st.write(f"**{current_month_events[0]['start'][:7]}**: {len(current_month_events)} shifts")
        if st.button(f"Sync {selected_provider}'s Current Month Only", key=f"quick_sync_{selected_provider}"):
            created, updated = 0, 0
            for event in current_month_events:
                body = local_event_to_gcal_body(event)
                g_ev = gcal_find_by_app_id(svc, sel_id, event.get("id", ""))
                if g_ev is None:
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
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
            st.success(f"Quick sync for {selected_provider}: created {created}, updated {updated} events")
    else:
        st.info(f"No shifts for {selected_provider} in the current month.")


def google_calendar_panel():
    st.subheader("Google Calendar Sync")

    # Connect / Authenticate
    svc = None
    if st.button("Connect Google Calendar"):
        svc = get_gcal_service()
        st.session_state["gcal_connected"] = bool(svc)
        if svc:
            st.success("Connected to Google Calendar.")
    else:
        # Try to reuse previous session silently
        if st.session_state.get("gcal_connected"):
            svc = get_gcal_service()

    if not svc:
        st.caption("Click **Connect Google Calendar** to authenticate.")
        return

    # Choose calendar
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    cal_ids = [c[0] for c in calendars]
    cal_labels = [c[1] for c in calendars]

    default_cal = st.session_state.get("gcal_calendar_id", "primary")
    if default_cal not in cal_ids:
        default_cal = cal_ids[0]

    sel_idx = cal_ids.index(default_cal)
    sel_label = st.selectbox("Calendar", options=cal_labels, index=sel_idx)
    sel_id = cal_ids[cal_labels.index(sel_label)]
    st.session_state["gcal_calendar_id"] = sel_id

    st.caption(f"Target: **{sel_label}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Push current month → Google"):
            to_push = filter_events_for_current_month()
            created, updated = 0, 0
            for E in to_push:
                body = local_event_to_gcal_body(E)
                # Try to find existing GCal event by our app_event_id
                g_ev = gcal_find_by_app_id(svc, sel_id, E.get("id",""))
                if g_ev is None:
                    # Create
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
                    # Update if changed
                    if (g_ev.get("summary") != body["summary"]) or (not _is_same_event_times(g_ev, body)):
                        g_ev["summary"] = body["summary"]
                        g_ev["start"]   = body["start"]
                        g_ev["end"]     = body["end"]
                        g_ev["description"] = body["description"]
                        g_ev.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                            body["extendedProperties"]["private"]
                        )
                        svc.events().update(calendarId=sel_id, eventId=g_ev["id"], body=g_ev).execute()
                        updated += 1
            st.success(f"Pushed month: created {created}, updated {updated}")

    with c2:
        if st.button("Remove this month's pushed events from Google"):
            # We'll look for events in this month that have our app_event_id private property and delete them
            year = st.session_state.month.year
            month = st.session_state.month.month
            start = datetime(year, month, 1)
            end = (start + relativedelta(months=1))
            time_min = start.isoformat() + "Z"
            time_max = end.isoformat() + "Z"

            removed = 0
            # Fetch all events in window and filter by privateExtendedProperty via app_event_id of local events
            local_ids = {e["id"] for e in filter_events_for_current_month()}
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id, timeMin=time_min, timeMax=time_max,
                    singleEvents=True, showDeleted=False, pageToken=page_token
                ).execute()
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            st.success(f"Removed {removed} events from Google for this month.")

def provider_requests_panel():
    """Panel for managing provider requests (vacations, blackout dates, shift swaps)."""
    st.subheader("📝 Provider Requests Management")
    
    # Initialize requests in session state
    if "provider_requests" not in st.session_state:
        st.session_state.provider_requests = {
            "vacations": [],
            "blackout_dates": [],
            "shift_swaps": []
        }
    
    # Get all providers
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded. Please load providers first.")
        return
    
    all_providers = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
    app_providers = sorted(APP_PROVIDER_INITIALS)
    
    # Create provider options with separators
    provider_options = ["(Select Provider)"]
    if all_providers:
        provider_options.append("--- Physicians ---")
        provider_options.extend(all_providers)
    if app_providers:
        provider_options.append("--- APPs ---")
        provider_options.extend(app_providers)
    
    # Request type selection
    request_type = st.selectbox(
        "Request Type",
        options=["Vacation Request", "Blackout Date Request", "Shift Swap Request"],
        key="request_type_select"
    )
    
    if request_type == "Vacation Request":
        vacation_request_form(provider_options)
    elif request_type == "Blackout Date Request":
        blackout_date_request_form(provider_options)
    elif request_type == "Shift Swap Request":
        shift_swap_request_form(provider_options)
    
    # Display existing requests
    display_existing_requests()


def vacation_request_form(provider_options):
    """Form for submitting vacation requests."""
    st.subheader("🏖️ Vacation Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="vacation_provider")
        start_date = st.date_input("Start Date", key="vacation_start")
    with col2:
        end_date = st.date_input("End Date", key="vacation_end")
        reason = st.text_area("Reason (optional)", key="vacation_reason", height=100)
    
    if st.button("Submit Vacation Request", key="submit_vacation"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        elif start_date > end_date:
            st.error("Start date must be before or equal to end date.")
        else:
            request = {
                "id": f"vacation_{len(st.session_state.provider_requests['vacations'])}",
                "type": "vacation",
                "provider": provider,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat()
            }
            st.session_state.provider_requests["vacations"].append(request)
            st.success(f"Vacation request submitted for {provider} from {start_date} to {end_date}")


def blackout_date_request_form(provider_options):
    """Form for submitting blackout date requests."""
    st.subheader("🚫 Blackout Date Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="blackout_provider")
        blackout_date = st.date_input("Blackout Date", key="blackout_date")
    with col2:
        reason = st.text_area("Reason (optional)", key="blackout_reason", height=100)
    
    if st.button("Submit Blackout Date Request", key="submit_blackout"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        else:
            request = {
                "id": f"blackout_{len(st.session_state.provider_requests['blackout_dates'])}",
                "type": "blackout_date",
                "provider": provider,
                "date": blackout_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat()
            }
            st.session_state.provider_requests["blackout_dates"].append(request)
            st.success(f"Blackout date request submitted for {provider} on {blackout_date}")


def shift_swap_request_form(provider_options):
    """Form for submitting shift swap requests."""
    st.subheader("🔄 Shift Swap Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider1 = st.selectbox("Provider 1", options=provider_options, key="swap_provider1")
        provider1_date = st.date_input("Provider 1 Date", key="swap_date1")
    with col2:
        provider2 = st.selectbox("Provider 2", options=provider_options, key="swap_provider2")
        provider2_date = st.date_input("Provider 2 Date", key="swap_date2")
    
    reason = st.text_area("Reason for swap (optional)", key="swap_reason", height=100)
    
    # Check for violations if both providers are selected
    if (provider1 != "(Select Provider)" and not provider1.startswith("---") and 
        provider2 != "(Select Provider)" and not provider2.startswith("---") and
        provider1 != provider2):
        
        violations = check_shift_swap_violations(
            provider1, 
            provider1_date.isoformat(), 
            provider2, 
            provider2_date.isoformat()
        )
        
        if violations:
            st.warning("⚠️ **Potential Violations Detected:**")
            for violation in violations:
                st.caption(violation)
            st.info("💡 You can still submit the request, but it will be flagged for review.")
        else:
            st.success("✅ No violations detected for this swap")
    
    if st.button("Submit Shift Swap Request", key="submit_swap"):
        if (provider1 == "(Select Provider)" or provider1.startswith("---") or 
            provider2 == "(Select Provider)" or provider2.startswith("---")):
            st.error("Please select both providers.")
        elif provider1 == provider2:
            st.error("Please select different providers for the swap.")
        else:
            # Check violations again before submitting
            violations = check_shift_swap_violations(
                provider1, 
                provider1_date.isoformat(), 
                provider2, 
                provider2_date.isoformat()
            )
            
            request = {
                "id": f"swap_{len(st.session_state.provider_requests['shift_swaps'])}",
                "type": "shift_swap",
                "provider1": provider1,
                "provider1_date": provider1_date.isoformat(),
                "provider2": provider2,
                "provider2_date": provider2_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat(),
                "violations": violations
            }
            st.session_state.provider_requests["shift_swaps"].append(request)
            
            if violations:
                st.warning(f"⚠️ Shift swap request submitted with {len(violations)} potential violation(s). Request will be reviewed.")
            else:
                st.success(f"✅ Shift swap request submitted between {provider1} ({provider1_date}) and {provider2} ({provider2_date})")


def display_existing_requests():
    """Display all existing requests with management options."""
    st.subheader("📋 Existing Requests")
    
    if not any(st.session_state.provider_requests.values()):
        st.info("No requests submitted yet.")
        return
    
    # Vacation requests
    if st.session_state.provider_requests["vacations"]:
        st.write("**🏖️ Vacation Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["vacations"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider']}**: {request['start_date']} to {request['end_date']}")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_vacation_{i}"):
                        st.session_state.provider_requests["vacations"][i]["status"] = "approved"
                        st.session_state.provider_requests["vacations"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_vacation_{i}"):
                        st.session_state.provider_requests["vacations"][i]["status"] = "rejected"
                        st.session_state.provider_requests["vacations"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")
    
    # Blackout date requests
    if st.session_state.provider_requests["blackout_dates"]:
        st.write("**🚫 Blackout Date Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["blackout_dates"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider']}**: {request['date']}")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_blackout_{i}"):
                        st.session_state.provider_requests["blackout_dates"][i]["status"] = "approved"
                        st.session_state.provider_requests["blackout_dates"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_blackout_{i}"):
                        st.session_state.provider_requests["blackout_dates"][i]["status"] = "rejected"
                        st.session_state.provider_requests["blackout_dates"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")
    
    # Shift swap requests
    if st.session_state.provider_requests["shift_swaps"]:
        st.write("**🔄 Shift Swap Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["shift_swaps"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider1']}** ({request['provider1_date']}) ↔ **{request['provider2']}** ({request['provider2_date']})")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Check for violations if pending
                if request['status'] == "pending":
                    violations = check_shift_swap_violations(
                        request['provider1'], 
                        request['provider1_date'], 
                        request['provider2'], 
                        request['provider2_date']
                    )
                    if violations:
                        st.warning("⚠️ **Potential Violations:**")
                        for violation in violations:
                            st.caption(violation)
                    else:
                        st.success("✅ No violations detected")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
                    if request['status'] == "approved":
                        st.info("🔄 Shift swap was executed")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_swap_{i}"):
                        st.session_state.provider_requests["shift_swaps"][i]["status"] = "approved"
                        st.session_state.provider_requests["shift_swaps"][i]["decision_date"] = date.today().isoformat()
                        # Execute the swap
                        success = execute_shift_swap(
                            request['provider1'], 
                            int(request['provider1_date'].split('-')[2]), 
                            request['provider2'], 
                            int(request['provider2_date'].split('-')[2]),
                            int(request['provider1_date'].split('-')[0]),
                            int(request['provider1_date'].split('-')[1])
                        )
                        if success:
                            st.session_state.provider_requests["shift_swaps"][i]["executed"] = True
                            st.success("Shift swap executed successfully!")
                        else:
                            st.session_state.provider_requests["shift_swaps"][i]["executed"] = False
                            st.error("Shift swap failed to execute!")
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_swap_{i}"):
                        st.session_state.provider_requests["shift_swaps"][i]["status"] = "rejected"
                        st.session_state.provider_requests["shift_swaps"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")

def check_shift_swap_violations(provider1: str, date1: str, provider2: str, date2: str) -> list:
    """Check for violations in a proposed shift swap."""
    violations = []
    
    try:
        # Parse dates
        date1_obj = pd.to_datetime(date1).date()
        date2_obj = pd.to_datetime(date2).date()
        
        # Get current events
        current_events = st.session_state.get("events", [])
        
        # Check for double assignments
        for event in current_events:
            ext = event.get("extendedProps", {})
            event_provider = ext.get("provider", "").strip().upper()
            try:
                event_date = pd.to_datetime(event["start"]).date()
            except Exception:
                continue
            
            # Check if provider1 is already assigned on date2
            if event_date == date2_obj and event_provider == provider1:
                violations.append(f"⚠️ {provider1} is already assigned on {date2_obj}")
            
            # Check if provider2 is already assigned on date1
            if event_date == date1_obj and event_provider == provider2:
                violations.append(f"⚠️ {provider2} is already assigned on {date1_obj}")
        
        # Check for 7+ shifts in a block
        def count_shifts_in_block(provider: str, target_date: date) -> int:
            """Count shifts for a provider in a 7-day block around the target date."""
            start_date = target_date - timedelta(days=3)
            end_date = target_date + timedelta(days=3)
            
            shift_count = 0
            for event in current_events:
                ext = event.get("extendedProps", {})
                event_provider = ext.get("provider", "").strip().upper()
                try:
                    event_date = pd.to_datetime(event["start"]).date()
                except Exception:
                    continue
                
                if event_provider == provider and start_date <= event_date <= end_date:
                    shift_count += 1
            
            return shift_count
        
        # Check provider1's block after swap
        provider1_shifts_after = count_shifts_in_block(provider1, date2_obj)
        if provider1_shifts_after > 7:
            violations.append(f"⚠️ {provider1} would have {provider1_shifts_after} shifts in 7-day block around {date2_obj}")
        
        # Check provider2's block after swap
        provider2_shifts_after = count_shifts_in_block(provider2, date1_obj)
        if provider2_shifts_after > 7:
            violations.append(f"⚠️ {provider2} would have {provider2_shifts_after} shifts in 7-day block around {date1_obj}")
        
        return violations
        
    except Exception as e:
        violations.append(f"⚠️ Error checking violations: {e}")
        return violations

if __name__ == "__main__":
    main()
#   pip install streamlit pandas numpy pydantic streamlit-calendar python-dateutil
#   # If streamlit-calendar fails to install, see: https://pypi.org/project/streamlit-calendar/
#
# Run:
#   streamlit run app.py

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

try:
    from streamlit_calendar import calendar as st_calendar
except Exception:
    st_calendar = None

# -------------------------
# Utilities & Data Models
# -------------------------

GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
GCAL_TOKEN_FILE = 'token.json'          # created on first successful auth
GCAL_CREDENTIALS_FILE = 'credentials.json'  # download from Google Cloud
APP_TIMEZONE = 'America/New_York'       # your timezone

DEFAULT_SHIFT_TYPES = [
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
    {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "07:00", "color": "#06b6d4"},
    {"key": "APP", "label": "APP Provider",      "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ----- Default provider roster -----
PROVIDER_INITIALS_DEFAULT = [
    "AA","AD","AM","FS","JM","JT","KA","LN","SM","OI","NP","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","EB","RS","RG","CJ","MS","AT",
    "YH","XL","MA","LM","MQ","CM","AI"
]

# ----- APP Provider roster -----
APP_PROVIDER_INITIALS = [
    "JA", "DN", "KP", "AR", "JL"
]

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

def is_holiday(check_date: date) -> Optional[Dict]:
    """Check if a date is a holiday and return holiday info if so"""
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
    """Get capacity adjusted for holidays"""
    holiday_info = is_holiday(check_date)
    if holiday_info:
        return max(1, int(base_capacity * holiday_info["capacity_multiplier"]))
    return base_capacity


def _normalize_initials_list(items):
    return sorted({str(x).strip().upper() for x in items if str(x).strip()})


class RuleConfig(BaseModel):
    # GLOBAL defaults
    min_shifts_per_provider: int = 15
    max_shifts_per_provider: int = Field(15, ge=1, le=31)

    # CHANGED: rest is now measured in DAYS (float)
    min_rest_days_between_shifts: float = Field(1.0, ge=0.0, le=14.0)

    min_block_size: int = Field(3, ge=1, le=7, description="Minimum consecutive days in a block when possible")
    max_block_size: Optional[int] = 7

    require_at_least_one_weekend: bool = True
    max_nights_per_provider: Optional[int] = Field(6, ge=0, le=31)


class Provider(BaseModel):
    initials: str

    @field_validator("initials")
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.strip().upper()

# Internal event schema aligned with FullCalendar
class SEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    backgroundColor: Optional[str] = None
    extendedProps: Dict[str, Any] = {}

    def to_json_event(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "backgroundColor": self.backgroundColor,
            "extendedProps": self.extendedProps,
        }


# -------------------------
# Utility Functions
# -------------------------

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


# --- Vacation helpers ---
def _expand_vacation_dates(vacations: list) -> set:
    """Expand [{'start':'YYYY-MM-DD','end':'YYYY-MM-DD'}, ...] to a set of date objects."""
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
    """True if any vacation day falls in the currently selected month."""
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
    """Count the number of vacation weeks a provider has in a specific month."""
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
    return RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))


def is_provider_unavailable_on_date(provider: str, day: date) -> bool:
    """Returns True if provider is unavailable (specific date or any vacation range) on 'day'."""
    pkey = (provider or "").strip().upper()
    pr = st.session_state.get("provider_rules", {}).get(pkey, {}) or {}

    # Specific dates
    for tok in pr.get("unavailable_dates", []):
        try:
            if pd.to_datetime(tok).date() == day:
                return True
        except Exception:
            pass

    # Vacation ranges
    for rng in pr.get("vacations", []) or []:
        try:
            s = pd.to_datetime(rng.get("start")).date()
            e = pd.to_datetime(rng.get("end")).date()
            if e < s: s, e = e, s
            if s <= day <= e:
                return True
        except Exception:
            pass
    return False


# -------------------------
# State helpers
# -------------------------

# --- Session bootstrap: make sure all keys exist before anything touches them ---
def init_session_state():
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
            # Merge with existing session state rules
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
            # Try to load from IMIS_initials.csv
            if os.path.exists("IMIS_initials.csv"):
                providers_df = pd.read_csv("IMIS_initials.csv")
                # Clean up the data - remove empty rows and normalize initials
                providers_df = providers_df.dropna()
                providers_df["initials"] = providers_df["initials"].astype(str).str.strip().str.upper()
                providers_df = providers_df[providers_df["initials"] != ""]
                providers_df = providers_df[providers_df["initials"] != "nan"]
                providers_df = providers_df[providers_df["initials"] != "NO"]  # Remove problematic entry
                if not providers_df.empty:
                    st.session_state["providers_df"] = providers_df
                    st.session_state["providers_loaded"] = True
                else:
                    # If CSV is empty or has no valid data, use defaults
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
            else:
                # Fallback to default providers if CSV doesn't exist
                default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                st.session_state["providers_df"] = default_providers
                st.session_state["providers_loaded"] = True
        except Exception as e:
            st.error(f"Failed to load providers: {e}")
            # Fallback to default providers
            default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
            st.session_state["providers_df"] = default_providers
            st.session_state["providers_loaded"] = True
    
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


def recommended_max_shifts_for_month() -> int:
    """Recommended max shifts per provider for the current month."""
    year = st.session_state.month.year
    month = st.session_state.month.month
    return get_min_shifts_for_month(year, month)


def events_for_calendar(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert events to calendar-compatible format."""
    return [_event_to_dict(e) for e in events]


def _event_to_dict(e):
    # Convert SEvent -> dict, and coerce datetimes to ISO strings
    if isinstance(e, dict):
        out = dict(e)
        # start / end may be datetime or pandas Timestamp
        for k in ("start", "end"):
            v = out.get(k)
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            elif hasattr(v, "to_pydatetime"):  # pandas Timestamp
                out[k] = v.to_pydatetime().isoformat()
            elif isinstance(v, str):
                # leave as-is
                pass
        # ensure extendedProps exists
        out.setdefault("extendedProps", {})
        return out

    # If it's an SEvent-like object
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
        # last resort: string-ify
        return {"raw": str(e)}


def _serialize_events_for_download(events):
    return [_event_to_dict(e) for e in (events or [])]


@st.cache_data
def make_month_days(year: int, month: int) -> List[date]:
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


# -------------------------
# Google Calendar Integration
# -------------------------

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
        "end":   {"dateTime": E["end"],   "timeZone": APP_TIMEZONE},
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
    """Shallow compare start/end; assumes timeZone handling via body."""
    g_start = (g_ev.get("start") or {}).get("dateTime") or (g_ev.get("start") or {}).get("date")
    g_end   = (g_ev.get("end")   or {}).get("dateTime") or (g_ev.get("end")   or {}).get("date")
    return (str(g_start) == str(local["start"]["dateTime"])) and (str(g_end) == str(local["end"]["dateTime"]))


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


# -------------------------
# Scheduling Engine (Greedy Draft)
# -------------------------

def parse_time(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def month_start_end(year: int, month: int):
    start = date(year, month, 1)
    end = (start + relativedelta(months=1)) - timedelta(days=1)
    return start, end


def build_empty_roster(days: List[date], shift_types: List[Dict[str, Any]]):
    # For each day, each shift key needs 1 provider by default (can be extended later)
    roster = {d: {s["key"]: None for s in shift_types} for d in days}
    return roster


def shifts_to_events(roster: Dict[date, Dict[str, Optional[str]]], shift_types: List[Dict[str, Any]]):
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


# -------------------------
# Validation Rules
# -------------------------

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
    violations: dict[str, list[str]] = {}

    cap_map: dict[str, int]   = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict]      = st.session_state.get("provider_rules", {})

    # --- helpers ---
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
            
            # Debug: Show provider rules being used
            if pr:
                st.caption(f"🔍 Debug: Provider {p_upper} rules: {pr}")
            
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


# Assignment Logic
def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    # Build lookup for shifts
    sdefs  = {s["key"]: s for s in shift_types}
    stypes = [s["key"] for s in shift_types]

    # Session-config maps
    cap_map: Dict[str, int]         = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
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
    # Use dynamic minimum based on month length for the current month being processed
    # We'll calculate this per month during assignment
    mbs = int(getattr(rules, "min_block_size", 1) or 1)
    mbx = getattr(rules, "max_block_size", None)
    min_rest_days_global = float(getattr(rules, "min_rest_days_between_shifts", 1.0))

    # ---------- helpers that read what we've already assigned in `events` ----------
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

    # ---------- feasibility + scoring ----------
    def ok(p: str, d: date, skey: str) -> bool:
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
       

    # ---------- build schedule ----------
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
                end_dt   = datetime.combine(current_day, parse_time(sdef["end"]))
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








def google_calendar_panel():
    st.subheader("Google Calendar Sync")

    # Connect / Authenticate
    svc = None
    if st.button("Connect Google Calendar"):
        svc = get_gcal_service()
        st.session_state["gcal_connected"] = bool(svc)
        if svc:
            st.success("Connected to Google Calendar.")
    else:
        # Try to reuse previous session silently
        if st.session_state.get("gcal_connected"):
            svc = get_gcal_service()

    if not svc:
        st.caption("Click **Connect Google Calendar** to authenticate.")
        return

    # Choose calendar
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    cal_ids = [c[0] for c in calendars]
    cal_labels = [c[1] for c in calendars]

    default_cal = st.session_state.get("gcal_calendar_id", "primary")
    if default_cal not in cal_ids:
        default_cal = cal_ids[0]

    sel_idx = cal_ids.index(default_cal)
    sel_label = st.selectbox("Calendar", options=cal_labels, index=sel_idx)
    sel_id = cal_ids[cal_labels.index(sel_label)]
    st.session_state["gcal_calendar_id"] = sel_id

    st.caption(f"Target: **{sel_label}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Push current month → Google"):
            to_push = filter_events_for_current_month()
            created, updated = 0, 0
            for E in to_push:
                body = local_event_to_gcal_body(E)
                # Try to find existing GCal event by our app_event_id
                g_ev = gcal_find_by_app_id(svc, sel_id, E.get("id",""))
                if g_ev is None:
                    # Create
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
                    # Update if changed
                    if (g_ev.get("summary") != body["summary"]) or (not _is_same_event_times(g_ev, body)):
                        g_ev["summary"] = body["summary"]
                        g_ev["start"]   = body["start"]
                        g_ev["end"]     = body["end"]
                        g_ev["description"] = body["description"]
                        g_ev.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                            body["extendedProperties"]["private"]
                        )
                        svc.events().update(calendarId=sel_id, eventId=g_ev["id"], body=g_ev).execute()
                        updated += 1
            st.success(f"Pushed month: created {created}, updated {updated}")

    with c2:
        if st.button("Remove this month's pushed events from Google"):
            # We'll look for events in this month that have our app_event_id private property and delete them
            year = st.session_state.month.year
            month = st.session_state.month.month
            start = datetime(year, month, 1)
            end = (start + relativedelta(months=1))
            time_min = start.isoformat() + "Z"
            time_max = end.isoformat() + "Z"

            removed = 0
            # Fetch all events in window and filter by privateExtendedProperty via app_event_id of local events
            local_ids = {e["id"] for e in filter_events_for_current_month()}
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id, timeMin=time_min, timeMax=time_max,
                    singleEvents=True, showDeleted=False, pageToken=page_token
                ).execute()
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            st.success(f"Removed {removed} events from Google for this month.")

def provider_selector():
    """One provider dropdown that updates global selection with separate sections for Physicians and APPs."""
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

    # Handle interactions
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



# provider rules section
# make sure this version is in your codebase
def provider_rules_panel():
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
                s = min(v_start, v_end); e = max(v_start, v_end)
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
                try: clean.append(str(pd.to_datetime(tok).date()))
                except Exception: pass
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
    



def schedule_grid_view():
    st.subheader("Monthly Grid — Shifts × Days (one provider per cell)")
    
    import pandas as pd

    if not st.session_state.shift_types:
        st.info("No shift types configured.")
        return

    def tod_group_and_order(skey: str, sdef: Dict[str, Any]):
        # New order: Rounders, 7am Admitter, 10am Admitter, Night Shift, Bridge Shift, APP Shift
        if skey == "R12":          return "Rounders (07:00–19:00)", 1
        if skey == "A12":          return "7am Admitter (07:00–19:00)", 2
        if skey == "A10":          return "10am Admitter (10:00–22:00)", 3
        if skey == "N12":          return "Night Shift (19:00–07:00)", 4
        if skey == "NB":           return "Bridge Shift (23:00–07:00)", 5
        if skey == "APP":          return "APP Shift (07:00–19:00)", 6
        # Fallback for any other shift types
        start = parse_time(sdef["start"])
        if 5 <= start.hour < 12:   return "Day", 1
        if 12 <= start.hour < 18:  return "Evening", 2
        return "Night", 3

    def start_minutes(sdef):
        t = parse_time(sdef["start"])
        return t.hour * 60 + t.minute

    def _hex_to_rgb(h):
        h = (h or "").lstrip("#")
        if len(h) == 3: h = "".join([c*2 for c in h])
        try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except Exception: return (102,102,102)

    def _rgb_to_hue(r,g,b):
        import colorsys
        h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
        return int(h*360)

    def emoji_for_hex(hex_color: str) -> str:
        r,g,b = _hex_to_rgb(hex_color); hue = _rgb_to_hue(r,g,b)
        if hue < 15 or hue >= 345: return "🔴"
        if 15 <= hue < 40:         return "🟠"
        if 40 <= hue < 70:         return "🟡"
        if 70 <= hue < 170:        return "🟢"
        if 170 <= hue < 250:       return "🔵"
        if 250 <= hue < 320:       return "🟣"
        return "🟤"

    # month context - allow viewing any month that has events
    all_events = st.session_state.get("events", [])
    
    # Find all months that have events
    months_with_events = set()
    for e in all_events:
        try:
            d = pd.to_datetime(e["start"]).date()
            months_with_events.add((d.year, d.month))
        except Exception:
            continue
    
    if not months_with_events:
        st.info("No events found. Generate a schedule first.")
        return
    
    # Default to current month, but allow selection
    default_year, default_month = st.session_state.month.year, st.session_state.month.month
    if (default_year, default_month) not in months_with_events:
        # If current month has no events, pick the first available month
        default_year, default_month = sorted(months_with_events)[0]
    
    # Month selector
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", options=sorted(set(y for y, m in months_with_events)), index=sorted(set(y for y, m in months_with_events)).index(default_year))
    with col2:
        month_options = [m for y, m in months_with_events if y == year]
        month = st.selectbox("Month", options=month_options, index=month_options.index(default_month))
    
    days = make_month_days(year, month)
    day_cols = [str(d.day) for d in days]
    
    # Show summary of available months
    month_names = []
    for y, m in sorted(months_with_events):
        month_name = date(y, m, 1).strftime('%B %Y')
        month_names.append(month_name)
    
    st.caption(f"📅 Available months: {', '.join(month_names)}")
    st.caption(f"📊 Viewing: {date(year, month, 1).strftime('%B %Y')} ({len(days)} days)")

    stypes  = st.session_state.shift_types
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)

    # build row meta (one row per capacity slot)
    row_meta = []
    for s in stypes:
        skey = s["key"]; cap = int(cap_map.get(skey, 1))
        group_label, gorder = tod_group_and_order(skey, s)
        for slot in range(1, cap + 1):
            row_label = f"{skey} — {s['label']} (slot {slot})"
            row_meta.append({
                "row_label": row_label, "skey": skey, "sdef": s,
                "slot": slot, "group": group_label, "gorder": gorder
            })
    row_meta.sort(key=lambda r: (r["gorder"], start_minutes(r["sdef"]), r["skey"], r["slot"]))

    import pandas as pd
    row_labels = [rm["row_label"] for rm in row_meta]
    grid_raw = pd.DataFrame("", index=row_labels, columns=day_cols, dtype="object")
    color_tags = [emoji_for_hex(rm["sdef"].get("color")) for rm in row_meta]
    grid_raw.insert(0, "Color", color_tags)  # first column

    # fill from events (first empty slot per shift/day)
    rows_for_key = {}
    for rm in row_meta:
        rows_for_key.setdefault(rm["skey"], []).append(rm["row_label"])

    for e in st.session_state.events:
        ext = (e.get("extendedProps") or {}); skey = ext.get("shift_key")
        if not skey: continue
        try:
            d = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        # Filter to selected month
        if d.year != year or d.month != month:
            continue
        prov = (ext.get("provider") or "").strip().upper() or "UNASSIGNED"
        col = str(d.day)
        for row_label in rows_for_key.get(skey, []):
            if grid_raw.at[row_label, col] == "":
                grid_raw.at[row_label, col] = prov
                break

    # height to avoid vertical scroll
    height_px = min(2200, 110 + len(row_meta) * 38)


    hi = (st.session_state.get("highlight_provider", "") or "").strip().upper()
    enable_highlight = hi != ""
    edit_mode = st.toggle("Edit grid (disables highlighting)", value=False, disabled=not enable_highlight)

    if enable_highlight and not edit_mode:
        # Styled, read-only grid with light background highlight
        day_only_cols = [c for c in grid_raw.columns if c.isdigit()]

        def _style_fn(val):
            try:
                return (
                    "background-color:#fff3bf; color:#111111; font-weight:700;"
                    if str(val).strip().upper() == hi else ""
                )
            except Exception:
                return ""

        styled = grid_raw.style.applymap(_style_fn, subset=day_only_cols)
        st.dataframe(styled, use_container_width=True, height=height_px)
        st.caption(f"Highlighting cells for **{hi}**. Toggle *Edit grid* to make changes.")
    else:
        # Editable grid - separate providers by type
        physician_provs = sorted(
            st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()
        ) if not st.session_state.providers_df.empty else []
        app_provs = sorted(APP_PROVIDER_INITIALS)
        
        col_config = {"Color": st.column_config.TextColumn(disabled=True, help="Shift color tag")}
        try:
            for c in day_cols:
                # Determine which providers can be assigned to this column based on row metadata
                # Check what shift types are available in this column based on the grid rows
                shift_types_in_col = set()
                for rm in row_meta:
                    if rm["row_label"] in grid_raw.index:
                        shift_types_in_col.add(rm["skey"])
                
                # Set options based on shift types in this column
                if "APP" in shift_types_in_col and len(shift_types_in_col) == 1:
                    # If ONLY APP shifts are available, only APP providers can be assigned
                    options = [""] + app_provs
                    help_text = f"Assignments for day {c} (APP providers only)"
                elif "APP" not in shift_types_in_col:
                    # If NO APP shifts are available, only physician providers
                    options = [""] + physician_provs
                    help_text = f"Assignments for day {c} (Physicians only)"
                else:
                    # Mixed shift types - allow both provider types
                    all_providers = physician_provs + app_provs
                    options = [""] + sorted(all_providers)
                    help_text = f"Assignments for day {c} (All providers)"
                
                col_config[c] = st.column_config.SelectboxColumn(
                    options=options,
                    help=help_text
                )
        except Exception:
            pass

        edited_grid = st.data_editor(
            grid_raw,
            num_rows="fixed",
            use_container_width=True,
            height=height_px,
            column_config=col_config,
            key="grid_editor",
        )

        # Auto-apply when grid changes
        if edited_grid is not None and not edited_grid.equals(grid_raw):
            # Validate for double assignments first
            conflicts = validate_grid_for_double_assignments(edited_grid)
            if conflicts:
                st.error("❌ Double assignments detected:\n" + "\n".join(conflicts))
            else:
                # Apply changes automatically
                apply_grid_to_calendar(edited_grid, year, month, row_meta)
                st.success("✅ Grid changes applied automatically!")
                # Remove automatic rerun to improve performance
                # st.rerun()

        # Manual apply button (for backup)
        if st.button("Apply grid to calendar (Manual)"):
            apply_grid_to_calendar(edited_grid, year, month, row_meta)
            st.success("✅ Grid changes applied manually!")
            # Only rerun on manual apply if needed
            # st.rerun()
        
        # Save functionality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Month"):
                save_month_to_file(year, month)
        with col2:
            if st.button("📁 Load Saved Month"):
                load_month_from_file(year, month)
        
        # Shift Swapping Functionality
        st.subheader("🔄 Shift Swapping")
        st.caption("Swap shifts between providers for the selected month")
        
        # Get all providers for swapping
        all_providers = sorted(physician_provs + app_provs)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            provider1 = st.selectbox("Provider 1", options=all_providers, key="swap_provider1")
        with col2:
            day1 = st.selectbox("Day 1", options=day_cols, key="swap_day1")
        with col3:
            provider2 = st.selectbox("Provider 2", options=all_providers, key="swap_provider2")
        with col4:
            day2 = st.selectbox("Day 2", options=day_cols, key="swap_day2")
        
        if st.button("🔄 Execute Shift Swap", help="Swap shifts between the selected providers and days"):
            if provider1 == provider2 and day1 == day2:
                st.warning("⚠️ Please select different providers or days for swapping")
            else:
                success = execute_shift_swap(provider1, int(day1), provider2, int(day2), year, month)
                if success:
                    st.success("✅ Shift swap completed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Shift swap failed. Check the console for details.")

def apply_grid_to_calendar(edited_grid, target_year, target_month, row_meta=None):
    """Apply grid changes to calendar events"""
    # Always normalize existing events before processing
    st.session_state.events = events_for_calendar(st.session_state.get("events", []))

    sdefs = {s["key"]: s for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())}
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps = st.session_state.get("provider_caps", {})
    prov_rules = st.session_state.get("provider_rules", {})
    global_rules = get_global_rules()
    base_max = recommended_max_shifts_for_month()
    mbx = getattr(global_rules, "max_block_size", None)
    
    # If row_meta is not provided, we need to build it
    if row_meta is None:
        stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
        row_meta = []
        for s in stypes:
            skey = s["key"]
            cap = int(cap_map.get(skey, 1))
            for slot in range(1, cap + 1):
                row_label = f"{skey} — {s['label']} (slot {slot})"
                row_meta.append({
                    "row_label": row_label, "skey": skey, "sdef": s,
                    "slot": slot
                })

    # keep comments by (date, shift_key, provider)
    comments_by_key = {}
    for e in st.session_state.events:
        ext = (e.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey or skey not in sdefs:
            continue
        try:
            d0 = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        if d0.year == target_year and d0.month == target_month:
            prov0 = (ext.get("provider") or "").strip().upper()
            comments_by_key[(d0, skey, prov0)] = list(st.session_state.comments.get(e["id"], []))

    # identify grid-controlled events for this month
    def is_grid_event(E: dict) -> bool:
        ext = (E.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey or skey not in sdefs:
            return False
        try:
            d0 = pd.to_datetime(E["start"]).date()
        except Exception:
            return False
        return d0.year == target_year and d0.month == target_month

    preserved = [E for E in st.session_state.events if not is_grid_event(E)]

    new_events = []
    seen_day_provider = set()  # {(date, provider)}
    conflicts = []

    # helpers that look at what's been added so far (new_events)
    def day_shift_count(dy, key):
        return sum(1 for E in new_events
                   if pd.to_datetime(E["start"]).date() == dy and
                      (E.get("extendedProps") or {}).get("shift_key") == key)

    def provider_has_shift_on_day(provider, dy):
        return any(
            (E.get("extendedProps") or {}).get("provider", "").upper() == provider and
            pd.to_datetime(E["start"]).date() == dy
            for E in new_events
        )

    def provider_days(provider):
        return {pd.to_datetime(E["start"]).date()
                for E in new_events
                if (E.get("extendedProps") or {}).get("provider", "").upper() == provider}

    def left_run_len(days_set, d0):
        run = 0; cur = d0 - timedelta(days=1)
        while cur in days_set:
            run += 1; cur -= timedelta(days=1)
        return run

    def right_run_len(days_set, d0):
        run = 0; cur = d0 + timedelta(days=1)
        while cur in days_set:
            run += 1; cur += timedelta(days=1)
        return run

    def total_block_len_if_assigned(provider, d0):
        ds = provider_days(provider)
        return left_run_len(ds, d0) + 1 + right_run_len(ds, d0)

    # live counters for per-provider totals/nights in this month build
    counts = {}
    nights = {}

    row_to_key = {rm["row_label"]: rm["skey"] for rm in row_meta}
    day_only_cols = [c for c in edited_grid.columns if c.isdigit()]

    for row_label in edited_grid.index:
        skey = row_to_key.get(row_label)
        if not skey:
            continue
        sdef = sdefs.get(skey)
        if not sdef:
            continue

        for col in day_only_cols:
            prov = edited_grid.at[row_label, col]
            prov = ("" if prov is None else str(prov)).strip().upper()
            if not prov:
                continue

            day_date = date(target_year, target_month, int(col))

            # one shift per provider per day
            key_dp = (day_date, prov)
            if key_dp in seen_day_provider or provider_has_shift_on_day(prov, day_date):
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (duplicate same-day assignment; skipped)")
                continue

            # per-shift daily capacity (in case capacity < number of rows filled)
            if day_shift_count(day_date, skey) >= int(cap_map.get(skey, 1)):
                conflicts.append(f"{day_date:%Y-%m-%d} {skey} over capacity; skipped")
                continue

            # hard block: unavailable (vacation or specific date)
            if is_provider_unavailable_on_date(prov, day_date):
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (on vacation/unavailable; skipped)")
                continue

            # eligibility (allowed shift types)
            allowed = prov_caps.get(prov, [])
            if allowed and skey not in allowed:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} not eligible for {skey}; skipped")
                continue

            # effective max shifts (month default, minus 3 if any vacation in month)
            pr = prov_rules.get(prov, {}) or {}
            eff_max = pr.get("max_shifts", base_max)
            if _provider_has_vacation_in_month(pr):
                eff_max = max(0, (eff_max or 0) - 3)

            counts.setdefault(prov, 0)
            nights.setdefault(prov, 0)

            if eff_max is not None and counts[prov] + 1 > eff_max:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} exceeds max shifts {eff_max}; skipped")
                continue

            # max nights
            max_nights = pr.get("max_nights", global_rules.max_nights_per_provider)
            if skey == "N12" and max_nights is not None and nights[prov] + 1 > max_nights:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} exceeds max nights {max_nights}; skipped")
                continue

            # max block size (if set)
            if mbx and mbx > 0 and total_block_len_if_assigned(prov, day_date) > mbx:
                conflicts.append(f"{day_date:%Y-%m-%d} — {prov} would exceed max block {mbx}; skipped")
                continue

            # build event
            def _parse(hhmm: str):
                hh, mm = hhmm.split(":")
                return time(int(hh), int(mm))
            start_dt = datetime.combine(day_date, _parse(sdef["start"]))
            end_dt   = datetime.combine(day_date, _parse(sdef["end"]))
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            eid = str(uuid.uuid4())
            ev = {
                "id": eid,
                "title": f"{sdef['label']} — {prov}",
                "start": start_dt.isoformat(),
                "end":   end_dt.isoformat(),
                "allDay": False,
                "backgroundColor": sdef.get("color"),
                "extendedProps": {"provider": prov, "shift_key": skey, "label": sdef["label"]},
            }
            new_events.append(ev)
            seen_day_provider.add(key_dp)

            # carry comments forward if any mapping existed
            k = (day_date, skey, prov)
            if k in st.session_state.comments:
                st.session_state.comments[eid] = st.session_state.comments[k]
            elif k in comments_by_key:
                st.session_state.comments[eid] = comments_by_key[k]

            # update counters
            counts[prov] += 1
            if skey == "N12":
                nights[prov] += 1

    st.session_state.events = events_for_calendar(preserved + new_events)

    if conflicts:
        st.warning("Some cells were skipped:\n- " + "\n- ".join(conflicts))
    else:
        st.success("Applied grid to calendar.")

def save_month_to_file(year: int, month: int):
    """Save the current month's events to a file"""
    try:
        # Get events for the specific month
        month_events = []
        for e in st.session_state.get("events", []):
            try:
                d = pd.to_datetime(e["start"]).date()
                if d.year == year and d.month == month:
                    month_events.append(e)
            except Exception:
                continue
        
        # Create filename
        filename = f"saved_month_{year}_{month:02d}.json"
        
        # Save to session state and file
        month_key = f"{year}_{month:02d}"
        st.session_state["saved_months"][month_key] = {
            "events": month_events,
            "comments": {eid: comments for eid, comments in st.session_state.get("comments", {}).items() 
                        if any(pd.to_datetime(e["start"]).date().year == year and 
                               pd.to_datetime(e["start"]).date().month == month 
                               for e in month_events if e["id"] == eid)},
            "saved_at": datetime.now().isoformat()
        }
        
        # Also save to disk
        import os
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(st.session_state["saved_months"][month_key], f, indent=2)
        
        st.success(f"✅ Saved {len(month_events)} events for {date(year, month, 1).strftime('%B %Y')} to {filename}")
        
    except Exception as e:
        st.error(f"❌ Failed to save month: {e}")

def load_month_from_file(year: int, month: int):
    """Load a saved month's events"""
    try:
        month_key = f"{year}_{month:02d}"
        saved_data = st.session_state.get("saved_months", {}).get(month_key)
        
        if not saved_data:
            # Try to load from disk
            import os
            data_dir = os.path.join(os.getcwd(), "data")
            filename = f"saved_month_{year}_{month:02d}.json"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    saved_data = json.load(f)
                st.session_state["saved_months"][month_key] = saved_data
            else:
                st.warning(f"❌ No saved data found for {date(year, month, 1).strftime('%B %Y')}")
                return
        
        # Remove existing events for this month
        existing_events = st.session_state.get("events", [])
        filtered_events = []
        for e in existing_events:
            try:
                d = pd.to_datetime(e["start"]).date()
                if not (d.year == year and d.month == month):
                    filtered_events.append(e)
            except Exception:
                filtered_events.append(e)
        
        # Add saved events
        saved_events = saved_data.get("events", [])
        st.session_state["events"] = filtered_events + saved_events
        
        # Restore comments
        saved_comments = saved_data.get("comments", {})
        st.session_state["comments"].update(saved_comments)
        
        st.success(f"✅ Loaded {len(saved_events)} events for {date(year, month, 1).strftime('%B %Y')}")
        
    except Exception as e:
        st.error(f"❌ Failed to load month: {e}")



# -------------------------
# App entry
# -------------------------
def main():
    init_session_state()
    

    
    # Main header
    st.title("🏥 Hospitalist Monthly Scheduler")
    
    # Provider status indicator
    if st.session_state.get("providers_loaded", False) and not st.session_state.providers_df.empty:
        provider_count = len(st.session_state.providers_df)
        st.success(f"✅ {provider_count} providers loaded and ready")
    else:
        st.error("❌ No providers loaded. Please go to the Providers tab to load providers.")
    
    st.markdown("---")
    
    # Navigation tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Calendar", "⚙️ Settings", "👥 Providers", "📊 Grid View", "📅 Google Sync", "📝 Requests"])
    
    with tab1:
        # Calendar tab - main scheduling interface
        st.header("Monthly Calendar")
        
        # Top controls in a clean layout
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            # Ensure providers are loaded and get the list
            if not st.session_state.providers_df.empty:
                physician_provs = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist())
                app_provs = sorted(APP_PROVIDER_INITIALS)
                
                # Filter out APP providers from the physician list
                physician_providers = [p for p in physician_provs if p not in app_provs]
                
                # Create provider options with separators
                provider_options = ["(Select Provider)"]
                if physician_providers:
                    provider_options.append("--- Physicians ---")
                    provider_options.extend(physician_providers)
                if app_providers:
                    provider_options.append("--- APPs ---")
                    provider_options.extend(app_provs)
                
                default = st.session_state.highlight_provider if st.session_state.highlight_provider in provider_options else "(All providers)"
                idx = provider_options.index(default) if default in provider_options else 0
                sel = st.selectbox("Highlight provider", options=provider_options, index=idx)
                st.session_state.highlight_provider = "" if sel == "(All providers)" else sel
            else:
                st.warning("No providers loaded. Please check the Providers tab.")
                st.session_state.highlight_provider = ""
        with col2:
            st.caption(f"📅 Currently viewing: {st.session_state.month.strftime('%B %Y')}")
        with col3:
            st.caption("💡 Use navigation buttons above calendar to change month")
        with col4:
            st.caption("🔄 Generate button creates schedules for the displayed month")
        
        # Generation info
        if st.session_state.get("generation_count", 0) > 0:
            st.caption(f"📊 Generated {st.session_state.generation_count} schedule(s) so far. Each generation creates a different schedule!")
        
        # Action buttons
        g1, g2, g3 = st.columns(3)
        with g1:
            if st.button("🔄 Generate Draft", help="Generate schedule for the displayed month"):
                if st.session_state.providers_df.empty:
                    st.error("❌ No providers loaded! Please go to the Providers tab and load providers first.")
                else:
                    providers = st.session_state.providers_df["initials"].tolist()
                    # Add APP providers to the list (avoiding duplicates)
                    csv_providers_upper = [p.upper() for p in providers]
                    for app_provider in APP_PROVIDER_INITIALS:
                        if app_provider.upper() not in csv_providers_upper:
                            providers.append(app_provider)
                    if not providers:
                        st.error("❌ Provider list is empty! Please add providers in the Providers tab.")
                    else:
                        # Count how many APP providers were actually added
                        app_providers_added = sum(1 for app_provider in APP_PROVIDER_INITIALS 
                                                if app_provider.upper() not in csv_providers_upper)
                        st.info(f"🔄 Generating schedule for {len(providers)} providers (including {app_providers_added} additional APP providers)...")
                        rules = RuleConfig(**st.session_state.rules)
                        # Generate days for the current month only
                        days = make_month_days(st.session_state.month.year, st.session_state.month.month)
                        # Add randomness seed for different schedules on each generation
                        import random
                        random_seed = random.randint(1, 10000)
                        random.seed(random_seed)
                        # Generate new events using the greedy algorithm
                        new_events = assign_greedy(providers, days, st.session_state.shift_types, rules)
                        # Convert SEvent objects to dictionary format for calendar
                        st.session_state.events = [_event_to_dict(e) for e in new_events]
                        st.session_state.comments = {}
                        st.session_state.generation_count += 1
                        st.success(f"✅ Draft schedule generated for {st.session_state.month.strftime('%B %Y')} (use navigation buttons above calendar to change month) with {len(new_events)} events! (Generation #{st.session_state.generation_count}, Seed: {random_seed})")
        with g2:
            if st.button("✅ Validate Schedule", help="Check for rule violations"):
                rules = RuleConfig(**st.session_state.rules)
                evs = [SEvent(**{**e, "start": datetime.fromisoformat(e["start"]), "end": datetime.fromisoformat(e["end"])}) for e in st.session_state.events]
                
                # Show shift counts for debugging
                provider_counts = {}
                for ev in evs:
                    provider = (ev.extendedProps.get("provider") or "").strip().upper()
                    if provider:
                        if provider not in provider_counts:
                            provider_counts[provider] = 0
                        provider_counts[provider] += 1
                
                viols = validate_rules(evs, rules)
                
                # Create a comprehensive validation results interface
                st.markdown("## 📋 Schedule Validation Results")
                
                # Summary metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", len(evs))
                with col2:
                    st.metric("Providers", len(provider_counts))
                with col3:
                    st.metric("Violations", len(viols) if viols else 0)
                with col4:
                    status = "✅ Valid" if not viols else "❌ Issues Found"
                    st.metric("Status", status)
                
                # Create tabs for different views
                if viols:
                    tab_summary, tab_details, tab_stats = st.tabs(["📊 Summary", "🔍 Details", "📈 Statistics"])
                    
                    with tab_summary:
                        st.subheader("🚨 Violation Summary")
                        
                        # Quick violation overview
                        violation_types = {}
                        for provider, violations in viols.items():
                            for violation in violations:
                                # Extract violation type from message
                                if "exceeds max" in violation.lower():
                                    violation_types["Max Shifts"] = violation_types.get("Max Shifts", 0) + 1
                                elif "below minimum" in violation.lower():
                                    violation_types["Min Shifts"] = violation_types.get("Min Shifts", 0) + 1
                                elif "insufficient rest" in violation.lower():
                                    violation_types["Rest Periods"] = violation_types.get("Rest Periods", 0) + 1
                                elif "unavailable" in violation.lower():
                                    violation_types["Unavailable Dates"] = violation_types.get("Unavailable Dates", 0) + 1
                                else:
                                    violation_types["Other"] = violation_types.get("Other", 0) + 1
                        
                        # Display violation types
                        if violation_types:
                            cols = st.columns(len(violation_types))
                            for i, (violation_type, count) in enumerate(violation_types.items()):
                                with cols[i]:
                                    st.metric(violation_type, count)
                        
                        # Providers with most violations
                        provider_violation_counts = {p: len(v) for p, v in viols.items()}
                        if provider_violation_counts:
                            st.subheader("⚠️ Providers with Issues")
                            sorted_providers = sorted(provider_violation_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            for provider, count in sorted_providers[:5]:  # Show top 5
                                st.warning(f"**{provider}**: {count} violation(s)")
                    
                    with tab_details:
                        st.subheader("🔍 Detailed Violations")
                        
                        # Group violations by provider in expandable sections
                        for provider, violations in viols.items():
                            with st.expander(f"**{provider}** ({len(violations)} violations)", expanded=False):
                                for i, violation in enumerate(violations, 1):
                                    # Color code violations
                                    if "exceeds max" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    elif "below minimum" in violation.lower():
                                        st.warning(f"{i}. {violation}")
                                    elif "insufficient rest" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    elif "unavailable" in violation.lower():
                                        st.error(f"{i}. {violation}")
                                    else:
                                        st.info(f"{i}. {violation}")
                    
                    with tab_stats:
                        st.subheader("📈 Provider Statistics")
                        
                        # Calculate statistics
                        stats_data = calculate_provider_statistics(evs)
                        provider_stats = stats_data["provider_stats"]
                        
                        if provider_stats:
                            # 1. Total shifts per provider
                            st.subheader("📊 Total Shifts per Provider")
                            shifts_data = {provider: data["total_shifts"] for provider, data in provider_stats.items()}
                            shifts_df = pd.DataFrame(list(shifts_data.items()), columns=["Provider", "Total Shifts"])
                            shifts_df = shifts_df.sort_values("Total Shifts", ascending=False)
                            
                            st.dataframe(shifts_df, use_container_width=True)
                            
                            # 2. Weekend shifts per provider
                            st.subheader("🌅 Weekend Shifts per Provider")
                            weekend_data = {provider: data["weekend_shifts"] for provider, data in provider_stats.items()}
                            weekend_df = pd.DataFrame(list(weekend_data.items()), columns=["Provider", "Weekend Shifts"])
                            weekend_df = weekend_df.sort_values("Weekend Shifts", ascending=False)
                            
                            st.dataframe(weekend_df, use_container_width=True)
                            
                            # 3. Coverage gaps
                            st.subheader("⚠️ Coverage Gaps")
                            gaps = identify_coverage_gaps(evs, st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES), 
                                                        st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
                            
                            # Debug information
                            st.caption(f"📊 **Debug Info**: Analyzing {len(evs)} events across {len(set(event.start.date() for event in evs))} unique days")
                            
                            if gaps:
                                # Group gaps by date for better display
                                gaps_by_date = {}
                                for gap in gaps:
                                    date_str = gap["date"].strftime("%Y-%m-%d")
                                    if date_str not in gaps_by_date:
                                        gaps_by_date[date_str] = []
                                    gaps_by_date[date_str].append(gap)
                                
                                # Create a summary table
                                summary_data = []
                                for date_str, date_gaps in gaps_by_date.items():
                                    # Get the date object from the first gap in this date group
                                    date_obj = date_gaps[0]["date"]
                                    day_name = date_obj.strftime("%A")
                                    total_shortage = sum(gap["shortage"] for gap in date_gaps)
                                    missing_shifts = [f"{gap['shift_type']} (-{gap['shortage']})" for gap in date_gaps]
                                    summary_data.append({
                                        "Date": date_str,
                                        "Day": day_name,
                                        "Total Shortage": total_shortage,
                                        "Missing Shifts": ", ".join(missing_shifts)
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                summary_df = summary_df.sort_values("Date")
                                
                                st.dataframe(summary_df, use_container_width=True)
                                
                                # Summary of gaps
                                total_gaps = len(gaps)
                                total_shortage = sum(gap["shortage"] for gap in gaps)
                                st.info(f"📋 **Coverage Summary**: {len(gaps_by_date)} days have insufficient coverage ({total_shortage} total missing providers)")
                                
                                # Show detailed breakdown
                                with st.expander("🔍 Detailed Coverage Analysis"):
                                    st.write("**Shift Type Breakdown:**")
                                    shift_breakdown = pd.DataFrame(gaps).groupby("shift_type").agg({
                                        "expected": "sum",
                                        "actual": "sum", 
                                        "shortage": "sum"
                                    }).reset_index()
                                    st.dataframe(shift_breakdown, use_container_width=True)
                                    
                                    # Show detailed gaps table
                                    st.write("**Detailed Gaps:**")
                                    gaps_df = pd.DataFrame(gaps)
                                    gaps_df["Date"] = gaps_df["date"].dt.strftime("%Y-%m-%d")
                                    gaps_df["Day"] = gaps_df["date"].dt.strftime("%A")
                                    gaps_df = gaps_df[["Date", "Day", "shift_type", "expected", "actual", "shortage"]]
                                    gaps_df.columns = ["Date", "Day", "Shift Type", "Expected", "Actual", "Shortage"]
                                    gaps_df = gaps_df.sort_values("Date")
                                    st.dataframe(gaps_df, use_container_width=True)
                            else:
                                st.success("✅ **No coverage gaps detected!** All shifts are properly staffed.")
                        else:
                            st.warning("⚠️ No provider data available for statistics.")
                else:
                    # No violations - show success with details
                    st.success("🎉 **Schedule is Valid!** No rule violations detected.")
                    
                    # Show statistics even when no violations
                    st.subheader("📈 Provider Statistics")
                    
                    # Calculate statistics
                    stats_data = calculate_provider_statistics(evs)
                    provider_stats = stats_data["provider_stats"]
                    
                    if provider_stats:
                        # 1. Total shifts per provider
                        st.subheader("📊 Total Shifts per Provider")
                        shifts_data = {provider: data["total_shifts"] for provider, data in provider_stats.items()}
                        shifts_df = pd.DataFrame(list(shifts_data.items()), columns=["Provider", "Total Shifts"])
                        shifts_df = shifts_df.sort_values("Total Shifts", ascending=False)
                        
                        st.dataframe(shifts_df, use_container_width=True)
                        
                        # 2. Weekend shifts per provider
                        st.subheader("🌅 Weekend Shifts per Provider")
                        weekend_data = {provider: data["weekend_shifts"] for provider, data in provider_stats.items()}
                        weekend_df = pd.DataFrame(list(weekend_data.items()), columns=["Provider", "Weekend Shifts"])
                        weekend_df = weekend_df.sort_values("Weekend Shifts", ascending=False)
                        
                        st.dataframe(weekend_df, use_container_width=True)
                        
                        # 3. Coverage gaps
                        st.subheader("⚠️ Coverage Gaps")
                        gaps = identify_coverage_gaps(evs, st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES), 
                                                    st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
                        
                        # Debug information
                        st.caption(f"📊 **Debug Info**: Analyzing {len(evs)} events across {len(set(event.start.date() for event in evs))} unique days")
                        
                        if gaps:
                            # Group gaps by date for better display
                            gaps_by_date = {}
                            for gap in gaps:
                                date_str = gap["date"].strftime("%Y-%m-%d")
                                if date_str not in gaps_by_date:
                                    gaps_by_date[date_str] = []
                                gaps_by_date[date_str].append(gap)
                            
                            # Create a summary table
                            summary_data = []
                            for date_str, date_gaps in gaps_by_date.items():
                                # Get the date object from the first gap in this date group
                                date_obj = date_gaps[0]["date"]
                                day_name = date_obj.strftime("%A")
                                total_shortage = sum(gap["shortage"] for gap in date_gaps)
                                missing_shifts = [f"{gap['shift_type']} (-{gap['shortage']})" for gap in date_gaps]
                                summary_data.append({
                                    "Date": date_str,
                                    "Day": day_name,
                                    "Total Shortage": total_shortage,
                                    "Missing Shifts": ", ".join(missing_shifts)
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            summary_df = summary_df.sort_values("Date")
                            
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Summary of gaps
                            total_gaps = len(gaps)
                            total_shortage = sum(gap["shortage"] for gap in gaps)
                            st.info(f"📋 **Coverage Summary**: {len(gaps_by_date)} days have insufficient coverage ({total_shortage} total missing providers)")
                            
                            # Show detailed breakdown
                            with st.expander("🔍 Detailed Coverage Analysis"):
                                st.write("**Shift Type Breakdown:**")
                                shift_breakdown = pd.DataFrame(gaps).groupby("shift_type").agg({
                                    "expected": "sum",
                                    "actual": "sum", 
                                    "shortage": "sum"
                                }).reset_index()
                                st.dataframe(shift_breakdown, use_container_width=True)
                                
                                # Show detailed gaps table
                                st.write("**Detailed Gaps:**")
                                gaps_df = pd.DataFrame(gaps)
                                gaps_df["Date"] = gaps_df["date"].dt.strftime("%Y-%m-%d")
                                gaps_df["Day"] = gaps_df["date"].dt.strftime("%A")
                                gaps_df = gaps_df[["Date", "Day", "shift_type", "expected", "actual", "shortage"]]
                                gaps_df.columns = ["Date", "Day", "Shift Type", "Expected", "Actual", "Shortage"]
                                gaps_df = gaps_df.sort_values("Date")
                                st.dataframe(gaps_df, use_container_width=True)
                        else:
                            st.success("✅ **No coverage gaps detected!** All shifts are properly staffed.")
                    else:
                        st.warning("⚠️ No provider data available for statistics.")
        with g3:
            if st.button("🗑️ Clear All", help="Clear all events"):
                st.session_state.events = []
                st.session_state.comments = {}
                st.success("All events cleared!")
        
        # Calendar display
        render_calendar()
    
    with tab2:
        # Settings tab - global rules and shift types
        st.header("Global Settings")
        
        # Global rules section
        st.subheader("📋 Scheduling Rules")
        
        # Info about dynamic minimum shifts and enhanced features
        st.info("💡 **Enhanced Features**:\n"
                "• **Dynamic Minimum Shifts**: Automatically enforced based on month length\n"
                "• **Shift Consistency**: Providers stay on same shift type within blocks\n"
                "• **Random Generation**: Each generate creates a different schedule\n"
                "• **Smart Month Generation**: Generates for the month currently displayed in calendar")
        
        rc = RuleConfig(**st.session_state.get("rules", RuleConfig().model_dump()))
        
        col1, col2 = st.columns(2)
        with col1:
            rc.max_shifts_per_provider = st.number_input("Max shifts/provider", 1, 31, value=int(rc.max_shifts_per_provider))
            rc.min_rest_days_between_shifts = st.number_input("Min rest (days) between shifts", min_value=0.0, max_value=14.0, step=0.5, value=float(getattr(rc, "min_rest_days_between_shifts", 1.0)))
            rc.min_block_size = st.number_input("Preferred block size (days)", 1, 7, value=int(rc.min_block_size))
        with col2:
            rc.require_at_least_one_weekend = st.checkbox("Require at least one weekend shift", value=bool(rc.require_at_least_one_weekend))
            limit_nights = st.checkbox("Limit 7pm–7am (N12) nights per provider", value=st.session_state.rules.get("max_nights_per_provider", 6) is not None)
            if limit_nights:
                default_nights = int(st.session_state.rules.get("max_nights_per_provider", 6) or 0)
                rc.max_nights_per_provider = st.number_input("Max nights/provider", 0, 31, value=default_nights)
            else:
                rc.max_nights_per_provider = None
        
        st.session_state.rules = rc.model_dump()
        
        # Shift types section
        st.subheader("🕐 Shift Types")
        st.caption("Edit labels, times, and colors for each shift type.")
        for i, s in enumerate(st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())):
            with st.expander(f"{s['label']} ({s['key']})", expanded=False):
                s["label"] = st.text_input("Label", value=s["label"], key=f"s_lbl_{i}")
                s["start"] = st.text_input("Start (HH:MM)", value=s["start"], key=f"s_st_{i}")
                s["end"]   = st.text_input("End (HH:MM)",   value=s["end"],   key=f"s_en_{i}")
                s["color"] = st.color_picker("Color", value=s.get("color", "#3388ff"), key=f"s_co_{i}")
        
        # Daily capacities section
        st.subheader("📊 Daily Shift Capacities")
        if st.button("Reset to default capacities"):
            st.session_state["shift_capacity"] = DEFAULT_SHIFT_CAPACITY.copy()
            st.toast("Capacities reset to defaults.", icon="♻️")
        
        cap_map = dict(st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
        for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy()):
            key = s["key"]; label = s["label"]
            default_cap = int(cap_map.get(key, DEFAULT_SHIFT_CAPACITY.get(key, 1)))
            cap_map[key] = int(st.number_input(f"{label} ({key}) capacity/day", min_value=0, max_value=50, value=default_cap, key=f"cap_{key}"))
        st.session_state["shift_capacity"] = cap_map
    
    with tab3:
        # Providers tab - manage provider roster and individual rules
        st.header("Provider Management")
        
        # Provider roster management
        st.subheader("👥 Provider Roster")
        current_list = st.session_state.providers_df["initials"].astype(str).tolist()
        st.caption(f"Currently loaded: {len(current_list)} providers")
        
        # Add a button to load default providers if none are loaded
        if len(current_list) == 0:
            st.warning("No providers loaded. Please load default providers or add providers manually.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Default Providers"):
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
                    st.success(f"Loaded {len(PROVIDER_INITIALS_DEFAULT)} default providers!")
                    st.rerun()
            with col2:
                if st.button("Load from CSV"):
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
                                st.success(f"Loaded {len(providers_df)} providers from CSV!")
                                st.rerun()
                            else:
                                st.error("CSV file is empty or has no valid data.")
                        else:
                            st.error("IMIS_initials.csv file not found.")
                    except Exception as e:
                        st.error(f"Failed to load CSV: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("➕ Add Providers", expanded=False):
                new_one = st.text_input("Add single provider (initials)", key="add_single_init")
                if st.button("Add", key="btn_add_single"):
                    cand = _normalize_initials_list([new_one])
                    if not cand:
                        st.warning("Enter initials to add.")
                    else:
                        initial = list(cand)[0]
                        if initial in current_list:
                            st.info(f"{initial} is already in the list.")
                        else:
                            st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(current_list + [initial])})
                            st.toast(f"Added {initial}", icon="✅")
                
                st.markdown("---")
                batch = st.text_area("Add multiple (comma/space/newline separated)", key="add_batch_area")
                if st.button("Add batch", key="btn_add_batch"):
                    tokens = _normalize_initials_list(batch.replace(",", "\n").split())
                    if not tokens:
                        st.warning("Nothing to add.")
                    else:
                        merged = _normalize_initials_list(current_list + list(tokens))
                        st.session_state.providers_df = pd.DataFrame({"initials": merged})
                        st.toast(f"Added {len(merged) - len(current_list)} new provider(s).", icon="✅")
        
        with col2:
            with st.expander("➖ Remove Providers", expanded=False):
                to_remove = st.multiselect("Select providers to remove", options=current_list, key="rm_multi")
                if st.button("Remove selected", key="btn_rm"):
                    if not to_remove:
                        st.info("No providers selected.")
                    else:
                        remaining = [p for p in current_list if p not in set(to_remove)]
                        st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(remaining)})
                        st.session_state["provider_caps"] = {k: v for k, v in st.session_state.provider_caps.items() if k in remaining}
                        st.toast(f"Removed {len(to_remove)} provider(s).", icon="🗑️")
        
        # Provider-specific rules
        st.subheader("⚙️ Provider-Specific Rules")
        provider_selector()
        provider_rules_panel()

    with tab4:
        # Grid view tab
        st.header("📊 Schedule Grid View")
        st.caption("Edit assignments directly in the grid below")
        schedule_grid_view()

    with tab5:
        # Google Calendar Sync tab
        provider_google_calendar_sync()

    with tab6:
        # Provider Requests tab
        st.header("📝 Provider Requests")
        st.caption("Providers can submit vacation requests, blackout dates, and shift swap requests")
        provider_requests_panel()

def execute_shift_swap(provider1: str, day1: int, provider2: str, day2: int, year: int, month: int) -> bool:
    """Execute a shift swap between two providers on different days"""
    try:
        # Normalize events
        st.session_state.events = events_for_calendar(st.session_state.get("events", []))
        
        # Find events for the specified days and providers
        events_to_swap = []
        target_events = []
        
        for event in st.session_state.events:
            ext = event.get("extendedProps", {})
            event_provider = ext.get("provider", "").strip().upper()
            try:
                event_date = pd.to_datetime(event["start"]).date()
            except Exception:
                continue
                
            if event_date.year == year and event_date.month == month:
                if event_date.day == day1 and event_provider == provider1:
                    events_to_swap.append(event)
                elif event_date.day == day2 and event_provider == provider2:
                    target_events.append(event)
        
        if not events_to_swap or not target_events:
            st.error(f"No shifts found for the specified providers and days")
            return False
        
        # Swap the providers
        for event in events_to_swap:
            event["extendedProps"]["provider"] = provider2
            event["title"] = event["title"].replace(provider1, provider2)
        
        for event in target_events:
            event["extendedProps"]["provider"] = provider1
            event["title"] = event["title"].replace(provider2, provider1)
        
        return True
        
    except Exception as e:
        st.error(f"Error during shift swap: {e}")
        return False

def provider_google_calendar_sync():
    """Allow each provider to sync their shifts to their own Google Calendar."""
    st.subheader("👤 Provider Google Calendar Sync")
    st.caption("Each provider can connect to their own Google Calendar and sync their shifts.")
    
    # Get all providers
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded. Please load providers first.")
        return
    
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
        "Select Provider to Sync",
        options=provider_options,
        key="provider_sync_select"
    )
    
    if selected_provider == "(Select Provider)" or selected_provider.startswith("---"):
        st.info("Please select a provider to sync their shifts to Google Calendar.")
        return
    
    # Initialize provider-specific session state
    provider_key = f"gcal_provider_{selected_provider}"
    if provider_key not in st.session_state:
        st.session_state[provider_key] = {
            "connected": False,
            "calendar_id": "primary",
            "calendar_name": "Primary Calendar"
        }
    
    provider_state = st.session_state[provider_key]
    
    # Connect to Google Calendar for this provider
    svc = None
    if st.button(f"Connect {selected_provider}'s Google Calendar", key=f"connect_{selected_provider}"):
        svc = get_gcal_service()
        if svc:
            provider_state["connected"] = True
            st.success(f"Connected {selected_provider} to Google Calendar.")
        else:
            st.error("Failed to connect to Google Calendar.")
    
    # Try to reuse previous connection
    if provider_state.get("connected"):
        svc = get_gcal_service()
    
    if not svc:
        st.caption(f"Click **Connect {selected_provider}'s Google Calendar** to authenticate.")
        return
    
    # Choose calendar for this provider
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    
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
        key=f"calendar_{selected_provider}"
    )
    sel_id = cal_ids[cal_labels.index(sel_label)]
    provider_state["calendar_id"] = sel_id
    provider_state["calendar_name"] = sel_label
    
    st.caption(f"Target: **{sel_label}** for {selected_provider}")
    
    # Filter events for this provider
    provider_events = []
    for event in st.session_state.get("events", []):
        ext = event.get("extendedProps", {})
        event_provider = (ext.get("provider") or "").strip().upper()
        if event_provider == selected_provider:
            provider_events.append(event)
    
    if not provider_events:
        st.info(f"No shifts found for {selected_provider}.")
        return
    
    # Show provider's shifts summary
    st.subheader(f"📊 {selected_provider}'s Shifts Summary")
    shifts_by_month = {}
    for event in provider_events:
        try:
            event_date = pd.to_datetime(event["start"]).date()
            month_key = (event_date.year, event_date.month)
            if month_key not in shifts_by_month:
                shifts_by_month[month_key] = []
            shifts_by_month[month_key].append(event)
        except Exception:
            continue
    
    # Display shifts by month
    for (year, month), month_events in sorted(shifts_by_month.items()):
        month_name = date(year, month, 1).strftime('%B %Y')
        st.write(f"**{month_name}**: {len(month_events)} shifts")
    
    # Sync options
    st.subheader(f"🔄 Sync Options for {selected_provider}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Sync {selected_provider}'s Shifts → Google", key=f"sync_{selected_provider}"):
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
            st.success(f"Synced {selected_provider}: created {created}, updated {updated} events")
    
    with col2:
        if st.button(f"Remove {selected_provider}'s Events from Google", key=f"remove_{selected_provider}"):
            removed = 0
            # Get all local event IDs for this provider
            local_ids = {event["id"] for event in provider_events}
            
            # Find and remove events from Google Calendar
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id,
                    singleEvents=True,
                    showDeleted=False,
                    pageToken=page_token
                ).execute()
                
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            
            st.success(f"Removed {removed} events for {selected_provider} from Google Calendar")
    
    # Show sync status
    st.subheader(f"📋 Sync Status for {selected_provider}")
    st.info(f"**Connected**: {provider_state['connected']}")
    st.info(f"**Calendar**: {provider_state['calendar_name']}")
    st.info(f"**Total Shifts**: {len(provider_events)}")
    
    # Quick sync for current month only
    st.subheader(f"📅 Quick Sync - Current Month")
    current_month_events = []
    current_year = st.session_state.month.year
    current_month = st.session_state.month.month
    
    for event in provider_events:
        try:
            event_date = pd.to_datetime(event["start"]).date()
            if event_date.year == current_year and event_date.month == current_month:
                current_month_events.append(event)
        except Exception:
            continue
    
    if current_month_events:
        st.write(f"**{current_month_events[0]['start'][:7]}**: {len(current_month_events)} shifts")
        if st.button(f"Sync {selected_provider}'s Current Month Only", key=f"quick_sync_{selected_provider}"):
            created, updated = 0, 0
            for event in current_month_events:
                body = local_event_to_gcal_body(event)
                g_ev = gcal_find_by_app_id(svc, sel_id, event.get("id", ""))
                if g_ev is None:
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
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
            st.success(f"Quick sync for {selected_provider}: created {created}, updated {updated} events")
    else:
        st.info(f"No shifts for {selected_provider} in the current month.")


def google_calendar_panel():
    st.subheader("Google Calendar Sync")

    # Connect / Authenticate
    svc = None
    if st.button("Connect Google Calendar"):
        svc = get_gcal_service()
        st.session_state["gcal_connected"] = bool(svc)
        if svc:
            st.success("Connected to Google Calendar.")
    else:
        # Try to reuse previous session silently
        if st.session_state.get("gcal_connected"):
            svc = get_gcal_service()

    if not svc:
        st.caption("Click **Connect Google Calendar** to authenticate.")
        return

    # Choose calendar
    calendars = gcal_list_calendars(svc)
    if not calendars:
        st.warning("No calendars available for this account.")
        return
    cal_ids = [c[0] for c in calendars]
    cal_labels = [c[1] for c in calendars]

    default_cal = st.session_state.get("gcal_calendar_id", "primary")
    if default_cal not in cal_ids:
        default_cal = cal_ids[0]

    sel_idx = cal_ids.index(default_cal)
    sel_label = st.selectbox("Calendar", options=cal_labels, index=sel_idx)
    sel_id = cal_ids[cal_labels.index(sel_label)]
    st.session_state["gcal_calendar_id"] = sel_id

    st.caption(f"Target: **{sel_label}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Push current month → Google"):
            to_push = filter_events_for_current_month()
            created, updated = 0, 0
            for E in to_push:
                body = local_event_to_gcal_body(E)
                # Try to find existing GCal event by our app_event_id
                g_ev = gcal_find_by_app_id(svc, sel_id, E.get("id",""))
                if g_ev is None:
                    # Create
                    svc.events().insert(calendarId=sel_id, body=body).execute()
                    created += 1
                else:
                    # Update if changed
                    if (g_ev.get("summary") != body["summary"]) or (not _is_same_event_times(g_ev, body)):
                        g_ev["summary"] = body["summary"]
                        g_ev["start"]   = body["start"]
                        g_ev["end"]     = body["end"]
                        g_ev["description"] = body["description"]
                        g_ev.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                            body["extendedProperties"]["private"]
                        )
                        svc.events().update(calendarId=sel_id, eventId=g_ev["id"], body=g_ev).execute()
                        updated += 1
            st.success(f"Pushed month: created {created}, updated {updated}")

    with c2:
        if st.button("Remove this month's pushed events from Google"):
            # We'll look for events in this month that have our app_event_id private property and delete them
            year = st.session_state.month.year
            month = st.session_state.month.month
            start = datetime(year, month, 1)
            end = (start + relativedelta(months=1))
            time_min = start.isoformat() + "Z"
            time_max = end.isoformat() + "Z"

            removed = 0
            # Fetch all events in window and filter by privateExtendedProperty via app_event_id of local events
            local_ids = {e["id"] for e in filter_events_for_current_month()}
            page_token = None
            while True:
                resp = svc.events().list(
                    calendarId=sel_id, timeMin=time_min, timeMax=time_max,
                    singleEvents=True, showDeleted=False, pageToken=page_token
                ).execute()
                for g_ev in resp.get("items", []):
                    priv = (g_ev.get("extendedProperties") or {}).get("private", {}) or {}
                    app_id = priv.get("app_event_id")
                    if app_id and app_id in local_ids:
                        svc.events().delete(calendarId=sel_id, eventId=g_ev["id"]).execute()
                        removed += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            st.success(f"Removed {removed} events from Google for this month.")

def provider_requests_panel():
    """Panel for managing provider requests (vacations, blackout dates, shift swaps)."""
    st.subheader("📝 Provider Requests Management")
    
    # Initialize requests in session state
    if "provider_requests" not in st.session_state:
        st.session_state.provider_requests = {
            "vacations": [],
            "blackout_dates": [],
            "shift_swaps": []
        }
    
    # Get all providers
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded. Please load providers first.")
        return
    
    all_providers = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
    app_providers = sorted(APP_PROVIDER_INITIALS)
    
    # Create provider options with separators
    provider_options = ["(Select Provider)"]
    if all_providers:
        provider_options.append("--- Physicians ---")
        provider_options.extend(all_providers)
    if app_providers:
        provider_options.append("--- APPs ---")
        provider_options.extend(app_providers)
    
    # Request type selection
    request_type = st.selectbox(
        "Request Type",
        options=["Vacation Request", "Blackout Date Request", "Shift Swap Request"],
        key="request_type_select"
    )
    
    if request_type == "Vacation Request":
        vacation_request_form(provider_options)
    elif request_type == "Blackout Date Request":
        blackout_date_request_form(provider_options)
    elif request_type == "Shift Swap Request":
        shift_swap_request_form(provider_options)
    
    # Display existing requests
    display_existing_requests()


def vacation_request_form(provider_options):
    """Form for submitting vacation requests."""
    st.subheader("🏖️ Vacation Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="vacation_provider")
        start_date = st.date_input("Start Date", key="vacation_start")
    with col2:
        end_date = st.date_input("End Date", key="vacation_end")
        reason = st.text_area("Reason (optional)", key="vacation_reason", height=100)
    
    if st.button("Submit Vacation Request", key="submit_vacation"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        elif start_date > end_date:
            st.error("Start date must be before or equal to end date.")
        else:
            request = {
                "id": f"vacation_{len(st.session_state.provider_requests['vacations'])}",
                "type": "vacation",
                "provider": provider,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat()
            }
            st.session_state.provider_requests["vacations"].append(request)
            st.success(f"Vacation request submitted for {provider} from {start_date} to {end_date}")


def blackout_date_request_form(provider_options):
    """Form for submitting blackout date requests."""
    st.subheader("🚫 Blackout Date Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="blackout_provider")
        blackout_date = st.date_input("Blackout Date", key="blackout_date")
    with col2:
        reason = st.text_area("Reason (optional)", key="blackout_reason", height=100)
    
    if st.button("Submit Blackout Date Request", key="submit_blackout"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        else:
            request = {
                "id": f"blackout_{len(st.session_state.provider_requests['blackout_dates'])}",
                "type": "blackout_date",
                "provider": provider,
                "date": blackout_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat()
            }
            st.session_state.provider_requests["blackout_dates"].append(request)
            st.success(f"Blackout date request submitted for {provider} on {blackout_date}")


def shift_swap_request_form(provider_options):
    """Form for submitting shift swap requests."""
    st.subheader("🔄 Shift Swap Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider1 = st.selectbox("Provider 1", options=provider_options, key="swap_provider1")
        provider1_date = st.date_input("Provider 1 Date", key="swap_date1")
    with col2:
        provider2 = st.selectbox("Provider 2", options=provider_options, key="swap_provider2")
        provider2_date = st.date_input("Provider 2 Date", key="swap_date2")
    
    reason = st.text_area("Reason for swap (optional)", key="swap_reason", height=100)
    
    # Check for violations if both providers are selected
    if (provider1 != "(Select Provider)" and not provider1.startswith("---") and 
        provider2 != "(Select Provider)" and not provider2.startswith("---") and
        provider1 != provider2):
        
        violations = check_shift_swap_violations(
            provider1, 
            provider1_date.isoformat(), 
            provider2, 
            provider2_date.isoformat()
        )
        
        if violations:
            st.warning("⚠️ **Potential Violations Detected:**")
            for violation in violations:
                st.caption(violation)
            st.info("💡 You can still submit the request, but it will be flagged for review.")
        else:
            st.success("✅ No violations detected for this swap")
    
    if st.button("Submit Shift Swap Request", key="submit_swap"):
        if (provider1 == "(Select Provider)" or provider1.startswith("---") or 
            provider2 == "(Select Provider)" or provider2.startswith("---")):
            st.error("Please select both providers.")
        elif provider1 == provider2:
            st.error("Please select different providers for the swap.")
        else:
            # Check violations again before submitting
            violations = check_shift_swap_violations(
                provider1, 
                provider1_date.isoformat(), 
                provider2, 
                provider2_date.isoformat()
            )
            
            request = {
                "id": f"swap_{len(st.session_state.provider_requests['shift_swaps'])}",
                "type": "shift_swap",
                "provider1": provider1,
                "provider1_date": provider1_date.isoformat(),
                "provider2": provider2,
                "provider2_date": provider2_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_date": date.today().isoformat(),
                "violations": violations
            }
            st.session_state.provider_requests["shift_swaps"].append(request)
            
            if violations:
                st.warning(f"⚠️ Shift swap request submitted with {len(violations)} potential violation(s). Request will be reviewed.")
            else:
                st.success(f"✅ Shift swap request submitted between {provider1} ({provider1_date}) and {provider2} ({provider2_date})")


def display_existing_requests():
    """Display all existing requests with management options."""
    st.subheader("📋 Existing Requests")
    
    if not any(st.session_state.provider_requests.values()):
        st.info("No requests submitted yet.")
        return
    
    # Vacation requests
    if st.session_state.provider_requests["vacations"]:
        st.write("**🏖️ Vacation Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["vacations"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider']}**: {request['start_date']} to {request['end_date']}")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_vacation_{i}"):
                        st.session_state.provider_requests["vacations"][i]["status"] = "approved"
                        st.session_state.provider_requests["vacations"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_vacation_{i}"):
                        st.session_state.provider_requests["vacations"][i]["status"] = "rejected"
                        st.session_state.provider_requests["vacations"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")
    
    # Blackout date requests
    if st.session_state.provider_requests["blackout_dates"]:
        st.write("**🚫 Blackout Date Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["blackout_dates"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider']}**: {request['date']}")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_blackout_{i}"):
                        st.session_state.provider_requests["blackout_dates"][i]["status"] = "approved"
                        st.session_state.provider_requests["blackout_dates"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_blackout_{i}"):
                        st.session_state.provider_requests["blackout_dates"][i]["status"] = "rejected"
                        st.session_state.provider_requests["blackout_dates"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")
    
    # Shift swap requests
    if st.session_state.provider_requests["shift_swaps"]:
        st.write("**🔄 Shift Swap Requests:**")
        for i, request in enumerate(st.session_state.provider_requests["shift_swaps"]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status_color = {
                    "pending": "🟡",
                    "approved": "🟢",
                    "rejected": "🔴"
                }.get(request['status'], "⚪")
                
                st.write(f"{status_color} **{request['provider1']}** ({request['provider1_date']}) ↔ **{request['provider2']}** ({request['provider2_date']})")
                if request['reason']:
                    st.caption(f"Reason: {request['reason']}")
                st.caption(f"Status: {request['status'].title()} | Submitted: {request['submitted_date']}")
                
                # Check for violations if pending
                if request['status'] == "pending":
                    violations = check_shift_swap_violations(
                        request['provider1'], 
                        request['provider1_date'], 
                        request['provider2'], 
                        request['provider2_date']
                    )
                    if violations:
                        st.warning("⚠️ **Potential Violations:**")
                        for violation in violations:
                            st.caption(violation)
                    else:
                        st.success("✅ No violations detected")
                
                # Show decision info if processed
                if request['status'] != "pending":
                    decision_date = request.get('decision_date', 'Unknown')
                    st.success(f"✅ Decision: {request['status'].title()} on {decision_date}")
                    if request['status'] == "approved":
                        st.info("🔄 Shift swap was executed")
            
            # Only show buttons if pending
            if request['status'] == "pending":
                with col2:
                    if st.button("Approve", key=f"approve_swap_{i}"):
                        st.session_state.provider_requests["shift_swaps"][i]["status"] = "approved"
                        st.session_state.provider_requests["shift_swaps"][i]["decision_date"] = date.today().isoformat()
                        # Execute the swap
                        success = execute_shift_swap(
                            request['provider1'], 
                            int(request['provider1_date'].split('-')[2]), 
                            request['provider2'], 
                            int(request['provider2_date'].split('-')[2]),
                            int(request['provider1_date'].split('-')[0]),
                            int(request['provider1_date'].split('-')[1])
                        )
                        if success:
                            st.session_state.provider_requests["shift_swaps"][i]["executed"] = True
                            st.success("Shift swap executed successfully!")
                        else:
                            st.session_state.provider_requests["shift_swaps"][i]["executed"] = False
                            st.error("Shift swap failed to execute!")
                        st.rerun()
                with col3:
                    if st.button("Reject", key=f"reject_swap_{i}"):
                        st.session_state.provider_requests["shift_swaps"][i]["status"] = "rejected"
                        st.session_state.provider_requests["shift_swaps"][i]["decision_date"] = date.today().isoformat()
                        st.rerun()
            else:
                # Show processed status
                with col2:
                    st.info("✅ Processed")
                with col3:
                    st.info("✅ Processed")

def check_shift_swap_violations(provider1: str, date1: str, provider2: str, date2: str) -> list:
    """Check for violations in a proposed shift swap."""
    violations = []
    
    try:
        # Parse dates
        date1_obj = pd.to_datetime(date1).date()
        date2_obj = pd.to_datetime(date2).date()
        
        # Get current events
        current_events = st.session_state.get("events", [])
        
        # Check for double assignments
        for event in current_events:
            ext = event.get("extendedProps", {})
            event_provider = ext.get("provider", "").strip().upper()
            try:
                event_date = pd.to_datetime(event["start"]).date()
            except Exception:
                continue
            
            # Check if provider1 is already assigned on date2
            if event_date == date2_obj and event_provider == provider1:
                violations.append(f"⚠️ {provider1} is already assigned on {date2_obj}")
            
            # Check if provider2 is already assigned on date1
            if event_date == date1_obj and event_provider == provider2:
                violations.append(f"⚠️ {provider2} is already assigned on {date1_obj}")
        
        # Check for 7+ shifts in a block
        def count_shifts_in_block(provider: str, target_date: date) -> int:
            """Count shifts for a provider in a 7-day block around the target date."""
            start_date = target_date - timedelta(days=3)
            end_date = target_date + timedelta(days=3)
            
            shift_count = 0
            for event in current_events:
                ext = event.get("extendedProps", {})
                event_provider = ext.get("provider", "").strip().upper()
                try:
                    event_date = pd.to_datetime(event["start"]).date()
                except Exception:
                    continue
                
                if event_provider == provider and start_date <= event_date <= end_date:
                    shift_count += 1
            
            return shift_count
        
        # Check provider1's block after swap
        provider1_shifts_after = count_shifts_in_block(provider1, date2_obj)
        if provider1_shifts_after > 7:
            violations.append(f"⚠️ {provider1} would have {provider1_shifts_after} shifts in 7-day block around {date2_obj}")
        
        # Check provider2's block after swap
        provider2_shifts_after = count_shifts_in_block(provider2, date1_obj)
        if provider2_shifts_after > 7:
            violations.append(f"⚠️ {provider2} would have {provider2_shifts_after} shifts in 7-day block around {date1_obj}")
        
        return violations
        
    except Exception as e:
        violations.append(f"⚠️ Error checking violations: {e}")
        return violations

if __name__ == "__main__":
    main()
