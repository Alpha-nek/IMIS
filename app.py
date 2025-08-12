# app.py — Interactive Monthly Scheduler for Multi-Doctor Shifts (Streamlit)
# ---------------------------------------------------------------
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
from datetime import datetime, date, timedelta, time
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, validator

try:
    from streamlit_calendar import calendar
except Exception:
    calendar = None

# -------------------------
# Utilities & Data Models
# -------------------------

GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
GCAL_TOKEN_FILE = 'token.json'          # created on first successful auth
GCAL_CREDENTIALS_FILE = 'credentials.json'  # download from Google Cloud
APP_TIMEZONE = 'America/New_York'       # your timezone

DEFAULT_SHIFT_TYPES = [
    {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "07:00", "color": "#06b6d4"},
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ----- Default provider roster -----
PROVIDER_INITIALS_DEFAULT = [
    "AA","AD","AM","FS","JM","JT","KA","LN","SM","OI","NP","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","EB","RS","RG","CJ","MS","AT",
    "YH","XL","MA","LM","MQ","CM","AI"
]

DEFAULT_SHIFT_CAPACITY = {"N12": 4, "NB": 1, "R12": 13, "A12": 1, "A10": 2}


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

    @validator("initials")
    def normalize(cls, v: str) -> str:
        return v.strip().upper()

# Internal event schema aligned with FullCalendar
class SEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    allDay: bool = False
    backgroundColor: Optional[str] = None
    borderColor: Optional[str] = None
    extendedProps: Dict[str, Any] = {}

    def to_fc(self) -> Dict[str, Any]:
        d = self.dict()
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d

import json
from datetime import datetime

def _ensure_iso(v):
    if isinstance(v, str):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime().isoformat()
    return str(v) if v is not None else None

def events_for_calendar(raw_events):
    out = []
    for e in (raw_events or []):
        d = _event_to_dict(e)
        if d is not None:
            out.append(d)
    return out
# --- Month-aware defaults ---
def _month_days_count() -> int:
    m = st.session_state.month
    import calendar
    return calendar.monthrange(m.year, m.month)[1]

def recommended_max_shifts_for_month() -> int:
    import calendar
    m = st.session_state.month
    days = calendar.monthrange(m.year, m.month)[1]
    if days == 31:
        return 16
    if days == 30:
        return 15
    return get_global_rules().max_shifts_per_provider


# --- Vacation helpers ---
def _expand_vacation_dates(vacations: list) -> set:
    """Expand [{'start':'YYYY-MM-DD','end':'YYYY-MM-DD'}, ...] to a set of date objects."""
    import pandas as pd
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

def get_shift_label_maps():
    stypes = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())
    label_for_key = {s["key"]: s["label"] for s in stypes}
    key_for_label = {v: k for k, v in label_for_key.items()}
    return label_for_key, key_for_label

def provider_weekend_count(p: str) -> int:
        return sum(1 for e in events
               if e.extendedProps.get("provider") == p and e.start.weekday() >= 5)

def get_global_rules():
    return RuleConfig(**st.session_state.get("rules", RuleConfig().dict()))

def is_provider_unavailable_on_date(provider: str, day: date) -> bool:
    """Returns True if provider is unavailable (specific date or any vacation range) on 'day'."""
    import pandas as pd
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
     # Provider roster (preloaded with your list)
    if "providers_df" not in st.session_state or st.session_state.get("providers_df") is None:
        st.session_state["providers_df"] = pd.DataFrame({"initials": _normalize_initials_list(PROVIDER_INITIALS_DEFAULT)})
    else:
        # Ensure normalization if present
        df = st.session_state["providers_df"]
        if df.empty:
            st.session_state["providers_df"] = pd.DataFrame({"initials": _normalize_initials_list(PROVIDER_INITIALS_DEFAULT)})
        else:
            st.session_state["providers_df"] = pd.DataFrame({
                "initials": _normalize_initials_list(df["initials"].tolist())
            })
    st.session_state.setdefault("rules", RuleConfig().dict())
# migrate if missing/None
    if st.session_state["rules"].get("max_block_size") is None:
        st.session_state["rules"]["max_block_size"] = 7
    # Migrate old hours-based field to days-based if present
    _rules = st.session_state.get("rules", {})
    if "min_rest_hours_between_shifts" in _rules and "min_rest_days_between_shifts" not in _rules:
        try:
            _rules["min_rest_days_between_shifts"] = float(_rules.pop("min_rest_hours_between_shifts")) / 24.0
        except Exception:
            _rules["min_rest_days_between_shifts"] = 1.0
    st.session_state["rules"] = _rules


    st.session_state.setdefault("shift_types", DEFAULT_SHIFT_TYPES.copy())
    st.session_state.setdefault("shift_capacity", DEFAULT_SHIFT_CAPACITY.copy())
    st.session_state.setdefault(
        "providers_df",
        pd.DataFrame({"initials": sorted(set(PROVIDER_INITIALS_DEFAULT))})
    )
    st.session_state.setdefault("rules", RuleConfig().dict())
    st.session_state.setdefault("provider_rules", {})     # per-provider overrides & vacations
    st.session_state.setdefault("provider_caps", {})      # per-provider allowed shift keys
    st.session_state.setdefault("events", [])             # calendar events (JSON-safe dicts)
    st.session_state.setdefault("comments", {})           # id -> list[str]
    st.session_state.setdefault("month", date.today().replace(day=1))
    st.session_state.setdefault("highlight_provider", "") # global selected provider

def get_gcal_service():
    """Return an authenticated Google Calendar API service."""
    creds = None
    if os.path.exists(GCAL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GCAL_TOKEN_FILE, GCAL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GCAL_CREDENTIALS_FILE):
                st.error("Missing credentials.json. Create an OAuth Client (Desktop) in Google Cloud and download it next to app.py.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(GCAL_CREDENTIALS_FILE, GCAL_SCOPES)
            # Local server flow (best for local testing)
            creds = flow.run_local_server(port=0)
        with open(GCAL_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('calendar', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Google API init failed: {e}")
        return None


def gcal_list_calendars(service):
    """Return list of (id, summary) tuples for available calendars."""
    items = []
    page_token = None
    while True:
        feed = service.calendarList().list(pageToken=page_token).execute()
        for c in feed.get('items', []):
            items.append((c['id'], c.get('summary', c['id'])))
        page_token = feed.get('nextPageToken')
        if not page_token:
            break
    return items


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


######################################################################################################## Validation Rules #############################################################################
def validate_rules(events: list[SEvent], rules: RuleConfig) -> dict[str, list[str]]:
    import pandas as pd
    violations: dict[str, list[str]] = {}


    cap_map: dict[str, int]   = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict]      = st.session_state.get("provider_rules", {})

    # --- helpers ---
    def _expand_vacation_dates(vacations: list) -> set[date]:
        out: set[date] = set()
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

    def _contiguous_blocks(dates_sorted: list[date]) -> list[tuple[date, date, int]]:
        blocks: list[tuple[date, date, int]] = []
        if not dates_sorted:
            return blocks
        start = prev = dates_sorted[0]
        for d in dates_sorted[1:]:
            if (d - prev).days == 1:
                prev = d
            else:
                blocks.append((start, prev, (prev - start).days + 1))
                start = prev = d
        blocks.append((start, prev, (prev - start).days + 1))
        return blocks

    # --- group events ---
    per_p: dict[str, list[SEvent]] = {}
    day_prov_counts: dict[tuple[date, str], int] = {}
    day_shift_counts: dict[tuple[date, str], int] = {}

    for ev in events:
        p = (ev.extendedProps.get("provider") or "").strip().upper()
        skey = ev.extendedProps.get("shift_key")
        d = ev.start.date()
        if p:
            per_p.setdefault(p, []).append(ev)
            day_prov_counts[(d, p)] = day_prov_counts.get((d, p), 0) + 1
        if skey:
            day_shift_counts[(d, skey)] = day_shift_counts.get((d, skey), 0) + 1

    # month-aware defaults
    base_default = recommended_max_shifts_for_month()
    min_required = int(getattr(rules, "min_shifts_per_provider", 15))
    mbs = int(getattr(rules, "min_block_size", 1) or 1)
    mbx = getattr(rules, "max_block_size", None)
    min_rest_days_global = float(getattr(rules, "min_rest_days_between_shifts", 1.0))


    for p_upper, evs in per_p.items():
        evs.sort(key=lambda e: e.start)
        pr = prov_rules.get(p_upper, {})

        # effective max shifts (override or month default), minus 3 if any vacation in month
        eff_max = pr.get("max_shifts", base_default)
        if _provider_has_vacation_in_month(pr):
            eff_max = max(0, (eff_max or 0) - 3)

        max_nights = pr.get("max_nights", rules.max_nights_per_provider)
        min_rest_days = float(pr.get("min_rest_days", min_rest_days_global))
    
        weekend_required = pr.get("require_weekend", rules.require_at_least_one_weekend)
        if weekend_required and not any(ev.start.weekday() >= 5 for ev in evs):
            violations.setdefault(p_upper, []).append("No weekend shifts")

        # 0) Min shifts
        if min_required and len(evs) < min_required:
            violations.setdefault(p_upper, []).append(f"Has {len(evs)} shifts < min {min_required}")

        # 1) Max shifts
        if eff_max is not None and len(evs) > eff_max:
            violations.setdefault(p_upper, []).append(f"Has {len(evs)} shifts > max {eff_max}")

        # 2) Rest hours between consecutive assignments
        for a, b in zip(evs, evs[1:]):
            rest_days = (b.start - a.end).total_seconds() / 86400.0
            if rest_days < (min_rest_days or 0.0):
                violations.setdefault(p_upper, []).append(
                    f"Rest {rest_days:.2f} days < min {min_rest_days:.2f} days "
                    f"between {a.start:%m-%d} and {b.start:%m-%d}"
                )

        # 3) Max nights
        if max_nights is not None:
            nights = sum(1 for ev in evs if ev.extendedProps.get("shift_key") == "N12")
            if nights > max_nights:
                violations.setdefault(p_upper, []).append(f"Nights {nights} > max {max_nights}")

        # 4) Weekend requirement
        if rules.require_at_least_one_weekend and not any(ev.start.weekday() >= 5 for ev in evs):
            violations.setdefault(p_upper, []).append("No weekend shifts")

        # 5) One shift per day
        for (d, pp), cnt in day_prov_counts.items():
            if pp == p_upper and cnt > 1:
                violations.setdefault(p_upper, []).append(f"{d:%Y-%m-%d}: {cnt} shifts in one day (limit 1)")

        # 6) Eligibility (allowed shift keys)
        allowed = prov_caps.get(p_upper, [])
        if allowed:
            bad = [ev for ev in evs if ev.extendedProps.get("shift_key") not in allowed]
            if bad:
                bad_keys = sorted(set(ev.extendedProps.get("shift_key") for ev in bad))
                violations.setdefault(p_upper, []).append(f"Not eligible for: {', '.join(bad_keys)}")

        # 7) Unavailability (specific dates + vacations) — HARD CHECK
        bad_dates = sorted({ev.start.date() for ev in evs if _is_unavailable(p_upper, ev.start.date())})
        for d in bad_dates:
            violations.setdefault(p_upper, []).append(f"{d:%Y-%m-%d}: provider unavailable (vacation/unavailable)")

        # 8) Block analysis (min/ max block sizes)
        dates_sorted = sorted([ev.start.date() for ev in evs])
        blocks = _contiguous_blocks(dates_sorted)

        if mbs > 1:
            for s, e, L in blocks:
                if L < mbs:
                    msg = f"{s:%Y-%m-%d}: 1-day block (pref {mbs})" if L == 1 else f"{s:%Y-%m-%d}–{e:%Y-%m-%d}: {L}-day block (pref {mbs})"
                    violations.setdefault(p_upper, []).append(msg)

        if mbx and mbx > 0:
            for s, e, L in blocks:
                if L > mbx:
                    violations.setdefault(p_upper, []).append(f"{s:%Y-%m-%d}–{e:%Y-%m-%d}: {L}-day block exceeds max {mbx}")

    # Global per-day / per-shift capacity
    for (d, skey), cnt in day_shift_counts.items():
        cap = cap_map.get(skey, 1)
        if cnt > cap:
            violations.setdefault("GLOBAL", []).append(f"{d:%Y-%m-%d} {skey}: {cnt} assigned > capacity {cap}")

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

    # Counters and accumulator
    counts: Dict[str, int] = {p: 0 for p in providers}
    nights: Dict[str, int] = {p: 0 for p in providers}
    events: List[SEvent] = []

    # Month-aware/global knobs
    base_max = recommended_max_shifts_for_month()
    min_required = int(getattr(rules, "min_shifts_per_provider", 15))
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

    # ---------- feasibility + scoring ----------
    def ok(p: str, d: date, skey: str) -> bool:
        p_upper = (p or "").upper()

        # 1) Eligibility
        allowed = prov_caps.get(p_upper, [])
        if allowed and skey not in allowed:
            return False

        # 2) Provider overrides / month defaults
        pr = prov_rules.get(p_upper, {}) or {}
        eff_max = pr.get("max_shifts", base_max)
        if _provider_has_vacation_in_month(pr):
            eff_max = max(0, (eff_max or 0) - 3)
        max_nights = pr.get("max_nights", rules.max_nights_per_provider)
        min_rest_days = float(pr.get("min_rest_days", min_rest_days_global))

        # 3) Hard unavailability
        if is_provider_unavailable_on_date(p_upper, d):
            return False

        # 4) Per-day caps & one-shift-per-day
        if day_shift_count(d, skey) >= cap_map.get(skey, 1):
            return False
        if provider_has_shift_on_day(p, d):
            return False

        # 5) Max totals & nights
        if eff_max is not None and counts[p] + 1 > eff_max:
            return False
        if skey == "N12" and max_nights is not None and nights[p] + 1 > max_nights:
            return False

        # 6) Block cap
        if mbx and mbx > 0 and total_block_len_if_assigned(p, d) > mbx:
            return False

        # 7) Rest window (in DAYS) against already-assigned events
        sdef = sdefs[skey]
        start_dt = datetime.combine(d, parse_time(sdef["start"]))
        end_dt   = datetime.combine(d, parse_time(sdef["end"]))
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        pu = p_upper
        for e in (ev for ev in events if (ev.extendedProps.get("provider") or "").upper() == pu):
            rest_after_prev_days  = (start_dt - e.end).total_seconds() / 86400.0
            rest_before_next_days = (e.start  - end_dt).total_seconds() / 86400.0
            if -(min_rest_days or 0.0) < rest_after_prev_days < (min_rest_days or 0.0):
                return False
            if -(min_rest_days or 0.0) < rest_before_next_days < (min_rest_days or 0.0):
                return False

        return True

    def score(provider_id: str, day: date, shift_key: str) -> float:
        sc = 0.0
        # toward minimum
        if counts[provider_id] < min_required: sc += 4.0
        # contiguous blocks up to preferred min size
        ds = provider_days(provider_id)
        L = left_run_len(ds, day)
        if L > 0: sc += 2.0
        if L < mbs: sc += 4.0
        # gentle load balance
        sc += max(0, 20 - counts[provider_id]) * 0.01
        # soft penalty if this hits the max block size
        if mbx and mbx > 0 and total_block_len_if_assigned(provider_id, day) == mbx: sc -= 0.2
        # weekend incentive if required & none yet
        weekend_required = prov_rules.get(provider_id, {}).get("require_weekend", rules.require_at_least_one_weekend)
        if day.weekday() >= 5 and weekend_required and provider_weekend_count(provider_id) == 0:
            sc += 3.0
        return sc

    # ---------- build schedule ----------
    for current_day in days:
        for shift_key in stypes:
            capacity = cap_map.get(shift_key, 1)
            for _ in range(capacity):
                candidates = [prov for prov in providers if ok(prov, current_day, shift_key)]
                if not candidates:
                    continue
                best = max(candidates, key=lambda prov: score(prov, current_day, shift_key))

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

# -------------------------
# UI
# -------------------------

def _event_to_dict(e):
    # Convert SEvent -> dict, and coerce datetimes to ISO strings
    from datetime import datetime
    import pandas as pd

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


def sidebar_inputs():
    st.sidebar.header("Providers & Rules")

    # ---- Safety bootstraps (avoid missing-key errors) ----
    st.session_state.setdefault("shift_types", DEFAULT_SHIFT_TYPES.copy())
    st.session_state.setdefault("shift_capacity", DEFAULT_SHIFT_CAPACITY.copy())
    st.session_state.setdefault("provider_caps", {})
    # Ensure providers_df exists with your preloaded roster
    base_roster = _normalize_initials_list(PROVIDER_INITIALS_DEFAULT)
    if "providers_df" not in st.session_state or st.session_state.get("providers_df") is None:
        st.session_state["providers_df"] = pd.DataFrame({"initials": base_roster})
    elif st.session_state.providers_df.empty:
        st.session_state["providers_df"] = pd.DataFrame({"initials": base_roster})
    else:
        st.session_state["providers_df"] = pd.DataFrame({
            "initials": _normalize_initials_list(st.session_state.providers_df["initials"].tolist())
        })

    # ===================== Providers (manage in-app) =====================
    st.sidebar.subheader("Providers")
    current_list = st.session_state.providers_df["initials"].astype(str).tolist()
    st.sidebar.caption(f"{len(current_list)} providers loaded.")

    with st.sidebar.expander("Add providers", expanded=False):
        new_one = st.text_input("Add single provider (initials)", key="add_single_init")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Add", key="btn_add_single"):
                cand = _normalize_initials_list([new_one])
                if not cand:
                    st.warning("Enter initials to add.")
                else:
                    initial = list(cand)[0]
                    if initial in current_list:
                        st.info(f"{initial} is already in the list.")
                    else:
                        st.session_state.providers_df = pd.DataFrame(
                            {"initials": _normalize_initials_list(current_list + [initial])}
                        )
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

    with st.sidebar.expander("Remove providers", expanded=False):
        to_remove = st.multiselect("Select providers to remove", options=current_list, key="rm_multi")
        if st.button("Remove selected", key="btn_rm"):
            if not to_remove:
                st.info("No providers selected.")
            else:
                remaining = [p for p in current_list if p not in set(to_remove)]
                st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(remaining)})
                st.session_state["provider_caps"] = {k: v for k, v in st.session_state.provider_caps.items() if k in remaining}
                st.toast(f"Removed {len(to_remove)} provider(s).", icon="🗑️")


    # ===================== Rules =====================
    st.sidebar.subheader("Rules")
    rc = RuleConfig(**st.session_state.get("rules", RuleConfig().dict()))
    rc.max_shifts_per_provider = st.sidebar.number_input("Max shifts/provider", 1, 31, value=int(rc.max_shifts_per_provider))
    rc.min_rest_days_between_shifts = st.number_input(
    "Min rest (days) between shifts",
    min_value=0.0, max_value=14.0, step=0.5,
    value=float(getattr(rc, "min_rest_days_between_shifts", 1.0)),
    key="rule_min_rest_days",)
    rc.min_block_size = st.sidebar.number_input("Preferred block size (days)", 1, 7, value=int(rc.min_block_size))
    rc.require_at_least_one_weekend = st.sidebar.checkbox("Require at least one weekend shift", value=bool(rc.require_at_least_one_weekend))
    limit_nights = st.sidebar.checkbox(
        "Limit 7pm–7am (N12) nights per provider",
        value=st.session_state.rules.get("max_nights_per_provider", 6) is not None
    )
    if limit_nights:
        default_nights = int(st.session_state.rules.get("max_nights_per_provider", 6) or 0)
        rc.max_nights_per_provider = st.sidebar.number_input("Max nights/provider", 0, 31, value=default_nights)
    else:
        rc.max_nights_per_provider = None
    st.session_state.rules = rc.dict()

    # ===================== Shift Types editor =====================
    st.sidebar.subheader("Shift Types")
    st.sidebar.caption("Edit labels/times; colors only affect calendar display.")
    for i, s in enumerate(st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())):
        with st.sidebar.expander(f"{s['label']} ({s['key']})", expanded=False):
            s["label"] = st.text_input("Label", value=s["label"], key=f"s_lbl_{i}")
            s["start"] = st.text_input("Start (HH:MM)", value=s["start"], key=f"s_st_{i}")
            s["end"]   = st.text_input("End (HH:MM)",   value=s["end"],   key=f"s_en_{i}")
            s["color"] = st.color_picker("Color", value=s.get("color", "#3388ff"), key=f"s_co_{i}")
    # write back edited shifts
    st.session_state["shift_types"] = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())

    # ===================== Daily shift capacities =====================
    st.sidebar.subheader("Daily shift capacities")
    if st.sidebar.button("Reset to default capacities"):
        st.session_state["shift_capacity"] = DEFAULT_SHIFT_CAPACITY.copy()
        st.toast("Capacities reset to defaults.", icon="♻️")

    cap_map = dict(st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
    for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy()):
        key = s["key"]; label = s["label"]
        default_cap = int(cap_map.get(key, DEFAULT_SHIFT_CAPACITY.get(key, 1)))
        cap_map[key] = int(
            st.sidebar.number_input(
                f"{label} ({key}) capacity/day",
                min_value=0, max_value=50, value=default_cap, key=f"cap_{key}"
            )
        )
    st.session_state["shift_capacity"] = cap_map

   
              
@st.cache_data
def make_month_days(year: int, month: int) -> List[date]:
    start, end = month_start_end(year, month)
    return list(date_range(start, end))


def top_controls():
    st.title("Hospitalist Monthly Scheduler — MVP")
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        year = st.number_input("Year", min_value=2020, max_value=2100, value=st.session_state.month.year)
    with c2:
        month = st.number_input("Month", min_value=1, max_value=12, value=st.session_state.month.month)
    with c3:
        if st.button("Go to Month"):
            st.session_state.month = date(int(year), int(month), 1)
    with c4:
        provs = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()) if not st.session_state.providers_df.empty else []
        options = ["(All providers)"] + provs
        default = st.session_state.highlight_provider if st.session_state.highlight_provider in provs else "(All providers)"
        idx = options.index(default) if default in options else 0
        sel = st.selectbox("Highlight provider (initials)", options=options, index=idx)
        st.session_state.highlight_provider = "" if sel == "(All providers)" else sel
    


    # Generate & Validate buttons
    g1, g2, g3 = st.columns(3)
    with g1:
        if st.button("Generate Draft from Rules"):
            providers = st.session_state.providers_df["initials"].tolist()
            if not providers:
                st.warning("Add providers first.")
            else:
                rules = RuleConfig(**st.session_state.rules)
                days = make_month_days(st.session_state.month.year, st.session_state.month.month)
                st.session_state.events = [_event_to_dict(e) for e in st.session_state.events]
                st.session_state.events = [e.to_fc() for e in assign_greedy(providers, days, st.session_state.shift_types, rules)]
                st.session_state.comments = {}
    with g2:
        if st.button("Validate Schedule"):
            rules = RuleConfig(**st.session_state.rules)
            evs = [SEvent(**{**e, "start": datetime.fromisoformat(e["start"]), "end": datetime.fromisoformat(e["end"])}) for e in st.session_state.events]
            viols = validate_rules(evs, rules)
            if not viols:
                st.success("No violations detected.")
            else:
                for p, arr in viols.items():
                    st.error(f"{p}:\n - " + "\n - ".join(arr))
    with g3:
        if st.button("Clear Month"):
            st.session_state.events = []
            st.session_state.comments = {}

def engine_panel():
    import pandas as pd
    st.header("Engine")

    # --- ONE global provider selector ---
    provider_selector()

    # ===== Providers (manage roster) =====
    st.subheader("Providers")
    current_list = st.session_state.providers_df["initials"].astype(str).tolist()
    st.caption(f"{len(current_list)} providers loaded.")

    with st.expander("Add providers", expanded=False):
        new_one = st.text_input("Add single provider (initials)", key="add_single_init")
        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            if st.button("Add", key="btn_add_single"):
                cand = _normalize_initials_list([new_one])
                if cand:
                    initial = list(cand)[0]
                    if initial not in current_list:
                        st.session_state.providers_df = pd.DataFrame(
                            {"initials": _normalize_initials_list(current_list + [initial])}
                        )
                        st.toast(f"Added {initial}", icon="✅")
                else:
                    st.warning("Enter initials to add.")
        st.markdown("---")
        batch = st.text_area("Add multiple (comma/space/newline separated)", key="add_batch_area")
        if st.button("Add batch", key="btn_add_batch"):
            tokens = _normalize_initials_list(batch.replace(",", "\n").split())
            if tokens:
                merged = _normalize_initials_list(current_list + list(tokens))
                st.session_state.providers_df = pd.DataFrame({"initials": merged})
                st.toast(f"Added {len(merged) - len(current_list)} new provider(s).", icon="✅")
            else:
                st.warning("Nothing to add.")

    with st.expander("Remove providers", expanded=False):
        to_remove = st.multiselect("Select providers to remove", options=current_list, key="rm_multi")
        if st.button("Remove selected", key="btn_rm"):
            if to_remove:
                remaining = [p for p in current_list if p not in set(to_remove)]
                st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(remaining)})
                st.session_state["provider_caps"] = {
                    k: v for k, v in st.session_state.provider_caps.items() if k in remaining
                }
                st.toast(f"Removed {len(to_remove)} provider(s).", icon="🗑️")
            else:
                st.info("No providers selected.")


    # ===== Global rules =====
   # ===== Global rules =====
    st.subheader("Rules (global)")
    
    # Load current rules safely
    rc_data = st.session_state.get("rules", RuleConfig().dict())
    rc = RuleConfig(**rc_data)
    
    # Month-aware recommendation (31-day→16, 30-day→15)
    rec_max = recommended_max_shifts_for_month()
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        rc.max_shifts_per_provider = st.number_input(
            "Max shifts/provider",
            min_value=1, max_value=50,
            value=int(rc.max_shifts_per_provider or rec_max),
            key="rule_max_shifts",
            help=f"Recommended this month: {rec_max}",
        )
    
        rc.min_shifts_per_provider = st.number_input(
            "Min shifts/provider",
            min_value=0, max_value=50,
            value=int(getattr(rc, "min_shifts_per_provider", 15)),
            key="rule_min_shifts",
        )
    
        rc.min_block_size = st.number_input(
            "Preferred block size (days)",
            min_value=1, max_value=7,
            value=int(getattr(rc, "min_block_size", 1)),
            key="rule_min_block",
        )
    
    with col2:
        # Max shifts per block (0 = no max). Default to 7 when unset/None.
        mbx_init = rc.max_block_size if rc.max_block_size is not None else 7
        mbx_val = st.number_input(
            "Max shifts per block (0 = no max)",
            min_value=0, max_value=31,
            value=int(mbx_init),
            key="rule_max_block",
        )
        rc.max_block_size = None if mbx_val == 0 else int(mbx_val)
    
        rc.require_at_least_one_weekend = st.checkbox(
            "Require at least one weekend shift",
            value=bool(getattr(rc, "require_at_least_one_weekend", False)),
            key="rule_req_weekend",
        )
    
    # Nights limit toggle + value
    limit_nights = st.checkbox(
        "Limit 7pm–7am (N12) nights per provider",
        value=(getattr(rc, "max_nights_per_provider", None) is not None),
        key="rule_limit_nights",
    )
    if limit_nights:
        default_nights = int(getattr(rc, "max_nights_per_provider", 6) or 6)
        rc.max_nights_per_provider = st.number_input(
            "Max nights/provider",
            min_value=0, max_value=50,
            value=default_nights,
            key="rule_max_nights",
        )
    else:
        rc.max_nights_per_provider = None
    
    # Persist back to session
    st.session_state["rules"] = rc.dict()

    # ===== Shift Types =====
    st.subheader("Shift Types")
    st.caption("Edit labels/times; colors only affect calendar display.")
    for i, s in enumerate(st.session_state.shift_types):
        with st.expander(f"{s['label']} ({s['key']})", expanded=False):
            s["label"] = st.text_input("Label", value=s["label"], key=f"s_lbl_{i}")
            s["start"] = st.text_input("Start (HH:MM)", value=s["start"], key=f"s_st_{i}")
            s["end"]   = st.text_input("End (HH:MM)",   value=s["end"],   key=f"s_en_{i}")
            s["color"] = st.color_picker("Color", value=s.get("color", "#3388ff"), key=f"s_co_{i}")

    # ===== Daily capacities (with default reset) =====
    st.subheader("Daily shift capacities")
    if st.button("Reset to default capacities"):
        st.session_state["shift_capacity"] = DEFAULT_SHIFT_CAPACITY.copy()
        st.toast("Capacities reset to defaults.", icon="♻️")

    cap_map = dict(st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY))
    for s in st.session_state.shift_types:
        key = s["key"]; label = s["label"]
        default_cap = int(cap_map.get(key, DEFAULT_SHIFT_CAPACITY.get(key, 1)))
        cap_map[key] = int(
            st.number_input(f"{label} ({key}) capacity/day", min_value=0, max_value=50, value=default_cap, key=f"cap_{key}")
        )
    st.session_state["shift_capacity"] = cap_map



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
        if st.button("Remove this month’s pushed events from Google"):
            # We’ll look for events in this month that have our app_event_id private property and delete them
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
    """One provider dropdown that updates global selection."""
    roster = (
        st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
        if not st.session_state.providers_df.empty else []
    )
    roster = sorted(roster)
    options = ["(All providers)"] + roster
    cur = st.session_state.get("highlight_provider", "") or ""
    idx = options.index(cur) if cur and cur in options else 0

    sel = st.selectbox("Provider", options=options, index=idx, key="provider_selector")
    st.session_state.highlight_provider = "" if sel == "(All providers)" else sel


def render_calendar():
    st.subheader(f"Calendar — {st.session_state.month:%B %Y}")
    if calendar is None:
        st.warning("streamlit-calendar is not installed or failed to import. Please install and restart.")
        return

    # Prepare events for FullCalendar
    all_events = st.session_state.events
    hi = (st.session_state.highlight_provider or "").strip().upper()
    if hi:
        # Show only the selected provider's shifts
        events = [e for e in all_events if ((e.get("extendedProps") or {}).get("provider", "").strip().upper() == hi)]
    else:
        events = list(all_events)

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
    
    # (Optional) filter calendar by the global provider selection
    hi = (st.session_state.get("highlight_provider", "") or "").strip().upper()
    if hi:
        events = [
            e for e in events
            if (e.get("extendedProps", {}).get("provider", "") or "").upper() == hi
        ]
    
    # Render the calendar
    state = calendar(
        events=events,
        # add your existing options/custom_css/etc. here if needed
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

def middle_actions_panel():
    """Month nav + generate/validate/clear, shown under the calendar (middle column)."""
    import pandas as pd

    st.subheader("Actions")

    # Month navigation row
    nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("◀ Prev month", key="mid_prev_month"):
            m = st.session_state.month
            y = m.year - (1 if m.month == 1 else 0)
            mm = 12 if m.month == 1 else m.month - 1
            st.session_state.month = date(y, mm, 1)
    with nav_label:
        st.markdown(
            f"<div style='text-align:center;font-weight:600'>{st.session_state.month:%B %Y}</div>",
            unsafe_allow_html=True
        )
    with nav_next:
        if st.button("Next month ▶", key="mid_next_month"):
            m = st.session_state.month
            y = m.year + (1 if m.month == 12 else 0)
            mm = 1 if m.month == 12 else m.month + 1
            st.session_state.month = date(y, mm, 1)

    # Action buttons row
    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("Generate Draft from Rules", key="mid_generate"):
            providers = st.session_state.providers_df["initials"].astype(str).tolist()
            if not providers:
                st.warning("Add providers first.")
            else:
                rules = RuleConfig(**st.session_state.rules)
                days = make_month_days(st.session_state.month.year, st.session_state.month.month)
                evs = assign_greedy(providers, days, st.session_state.shift_types, rules)

                # preserve events outside this month
                def is_this_month(e):
                    try:
                        d = pd.to_datetime(e["start"]).date()
                        return d.year == st.session_state.month.year and d.month == st.session_state.month.month
                    except Exception:
                        return False

                keep_others = [E for E in st.session_state.events if not is_this_month(E)]
                new_json = [e.to_fc() if hasattr(e, "to_fc") else e for e in evs]
                st.session_state.events = events_for_calendar(keep_others + new_json)
                st.success("Draft generated.")

    with act2:
        if st.button("Validate schedule", key="mid_validate"):
            # Convert JSON events to SEvent if needed
            def _to_sevent(E):
                if isinstance(E, dict):
                    ext = E.get("extendedProps") or {}
                    return SEvent(
                        id=E.get("id", ""),
                        title=E.get("title", ""),
                        start=pd.to_datetime(E["start"]).to_pydatetime(),
                        end=pd.to_datetime(E["end"]).to_pydatetime(),
                        backgroundColor=E.get("backgroundColor"),
                        extendedProps={
                            "provider": ext.get("provider"),
                            "shift_key": ext.get("shift_key"),
                            "label": ext.get("label"),
                        },
                    )
                return E

            events_obj = [_to_sevent(E) for E in st.session_state.events]
            viol = validate_rules(events_obj, RuleConfig(**st.session_state.rules))
            if not viol:
                st.success("No violations found.")
            else:
                for who, msgs in viol.items():
                    st.warning(f"**{who}**:\n- " + "\n- ".join(msgs))

    with act3:
        if st.button("Clear month", key="mid_clear"):
            def is_this_month(e):
                try:
                    d = pd.to_datetime(e["start"]).date()
                    return d.year == st.session_state.month.year and d.month == st.session_state.month.month
                except Exception:
                    return False
            st.session_state.events = [E for E in st.session_state.events if not is_this_month(E)]
            st.toast("Cleared this month.", icon="🧹")

# provider rules section
# make sure this version is in your codebase
def provider_rules_panel():
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

    # ----- Overrides & availability
    st.markdown("---")
    st.subheader("Overrides (optional)")

    base_default = recommended_max_shifts_for_month()
    curr = rules_map.get(sel, {}).copy()  # work on a copy

    c1, c2 = st.columns(2)
    with c1:
        use_max = st.checkbox(
            "Override max shifts / month",
            value=("max_shifts" in curr),
            key=f"pr_use_max_{sel}",
        )
        st.caption(f"Recommended default this month: **{base_default}**")
        max_sh = st.number_input(
            "Max shifts (this month)",
            1, 50,
            value=int(curr.get("max_shifts", base_default)),
            key=f"pr_max_{sel}",
        )
    with c2:
        use_nights = st.checkbox(
            "Override max nights / month",
            value=("max_nights" in curr),
            key=f"pr_use_nights_{sel}",
        )
        global_rules = get_global_rules()
        default_max_n = global_rules.max_nights_per_provider if global_rules.max_nights_per_provider is not None else 0
        max_n = st.number_input(
            "Max nights (this month)",
            0, 50,
            value=int(curr.get("max_nights", default_max_n)),
            key=f"pr_max_n_{sel}",
        )

    # NEW: Weekend requirement override (no min_rest editor anywhere)
    use_weekend = st.checkbox(
        "Override weekend requirement",
        value=("require_weekend" in curr),
        key=f"pr_use_weekend_{sel}",
    )
    # --- replace the min-rest override UI in provider_rules_panel() ---
    use_rest = st.checkbox(
        "Override min rest (days)",
        value=("min_rest_days" in curr or "min_rest_hours" in curr),  # allow old hours key for migration
        key=f"pr_use_rest_{sel}",
    )
    
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

    wk_idx = 0 if curr.get("require_weekend", True) else 1
    wk_choice = st.radio(
        "Weekend requirement",
        options=["Require at least one", "No weekend required"],
        index=wk_idx,
        key=f"pr_weekend_choice_{sel}",
        horizontal=True,
    )

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

        # merge toggles
        if use_max:    new_entry["max_shifts"] = int(max_sh)
        else:          new_entry.pop("max_shifts", None)

        if use_nights: new_entry["max_nights"] = int(max_n)
        else:          new_entry.pop("max_nights", None)

        if use_weekend:
            new_entry["require_weekend"] = (wk_choice == "Require at least one")
        else:
            new_entry.pop("require_weekend", None)

                # in the "Save provider rules" handler:
        if use_rest:
            new_entry["min_rest_days"] = float(min_rest_days)


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

        st.success("Saved provider rules.")


def schedule_grid_view():
    st.subheader("Monthly Grid — Shifts × Days (one provider per cell)")

    if not st.session_state.shift_types:
        st.info("No shift types configured.")
        return

    def tod_group_and_order(skey: str, sdef: Dict[str, Any]):
        start = parse_time(sdef["start"])
        if skey in ("R12", "A12"): return "Day (07:00–19:00)", 1
        if skey == "A10":          return "Evening (10:00–22:00)", 2
        if skey == "N12":          return "Night (19:00–07:00)", 3
        if skey == "NB":           return "Late Night (23:00–03:00)", 4
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

    # month context
    year  = st.session_state.month.year
    month = st.session_state.month.month
    days  = make_month_days(year, month)
    day_cols = [str(d.day) for d in days]

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
        # Editable grid
        valid_provs = sorted(
            st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()
        ) if not st.session_state.providers_df.empty else []
        col_config = {"Color": st.column_config.TextColumn(disabled=True, help="Shift color tag")}
        try:
            for c in day_cols:
                col_config[c] = st.column_config.SelectboxColumn(options=[""] + valid_provs,
                                                                 help=f"Assignments for day {c}")
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

       # Apply back to events
if st.button("Apply grid to calendar"):
    # Always normalize existing events before processing
    st.session_state.events = events_for_calendar(st.session_state.get("events", []))

    sdefs = {s["key"]: s for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())}
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps = st.session_state.get("provider_caps", {})
    prov_rules = st.session_state.get("provider_rules", {})
    global_rules = get_global_rules()
    base_max = recommended_max_shifts_for_month()
    mbx = getattr(global_rules, "max_block_size", None)

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
        if d0.year == year and d0.month == month:
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
        return d0.year == year and d0.month == month

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
        L = left_run_len(ds, d0)
        R = right_run_len(ds, d0)
        return L + 1 + R

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

            day_date = date(year, month, int(col))

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



# -------------------------
# App entry
# -------------------------
def main():
    init_session_state()
    left_col, mid_col, right_col = st.columns([3,5,3], gap="large")
    with left_col:
        engine_panel()                 # (now without bulk eligibility + actions)
    with mid_col:
        render_calendar()
        middle_actions_panel()         # ← new spot for actions
        schedule_grid_view()
    with right_col:
        provider_rules_panel()

main()






