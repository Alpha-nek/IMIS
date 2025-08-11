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
from typing import List, Dict, Any, Optional

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
    "AA","AD","AM","DN","FS","JL","JM","JT","KA","LN","MO","SM","OI","NP","AR","JA","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","KP","EB","RS","RG","CJ","MS","AT",
    "YH","XL","NO","MA","LM","MQ","CM","AI"
]

DEFAULT_SHIFT_CAPACITY = {"N12": 4, "NB": 1, "R12": 13, "A12": 1, "A10": 2}


def _normalize_initials_list(items):
    return sorted({str(x).strip().upper() for x in items if str(x).strip()})


class RuleConfig(BaseModel):
    max_shifts_per_provider: int = Field(15, ge=1, le=31)
    min_rest_hours_between_shifts: int = Field(12, ge=0, le=48)
    min_block_size: int = Field(3, ge=1, le=7, description="Minimum consecutive days in a block when possible")
    require_at_least_one_weekend: bool = True
    # Optional caps per shift type
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

def _event_to_dict(e):
    # Normalize any event-like object to a plain dict with ISO datetimes
    if isinstance(e, dict):
        d = dict(e)
    elif hasattr(e, "to_json_event"):
        d = dict(e.to_json_event())
    else:
        d = {
            "id": getattr(e, "id", None),
            "title": getattr(e, "title", ""),
            "start": getattr(getattr(e, "start", None), "isoformat", lambda: None)(),
            "end": getattr(getattr(e, "end", None), "isoformat", lambda: None)(),
            "backgroundColor": getattr(e, "backgroundColor", None),
            "extendedProps": getattr(e, "extendedProps", {}) or {},
        }

    d["title"] = str(d.get("title", ""))
    d["start"] = _ensure_iso(d.get("start"))
    d["end"]   = _ensure_iso(d.get("end"))
    d["allDay"] = bool(d.get("allDay", False))
    if d.get("backgroundColor") is not None:
        d["backgroundColor"] = str(d["backgroundColor"])

    # extendedProps must be a JSON-safe dict
    ext = d.get("extendedProps") or {}
    if not isinstance(ext, dict):
        ext = {}
    ext2 = {}
    for k, v in ext.items():
        try:
            json.dumps(v)
            ext2[str(k)] = v
        except Exception:
            ext2[str(k)] = str(v)
    d["extendedProps"] = ext2

    # Drop events missing required fields
    if not d.get("start") or not d.get("end"):
        return None
    return d

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

def get_global_rules():
    return RuleConfig(**st.session_state.get("rules", RuleConfig().dict()))


# -------------------------
# State helpers
# -------------------------



    
# --- Session bootstrap: make sure all keys exist before anything touches them ---
def init_session_state():
    st.set_page_config(page_title="Scheduling", layout="wide", initial_sidebar_state="collapsed")
    st.session_state.setdefault("shift_capacity", DEFAULT_SHIFT_CAPACITY.copy())
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

    # Eligibility & capacities defaults
    st.session_state.setdefault("provider_caps", {})  # initials -> allowed shift keys
    st.session_state.setdefault("shift_capacity", DEFAULT_SHIFT_CAPACITY.copy())
    
def _bootstrap_session_state():
    from datetime import date

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

_bootstrap_session_state()





   



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


def validate_rules(events: List[SEvent], rules: RuleConfig) -> Dict[str, List[str]]:
    violations: Dict[str, List[str]] = {}
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: Dict[str, List[str]] = st.session_state.get("provider_caps", {})
    prov_rules: Dict[str, Dict[str, Any]] = st.session_state.get("provider_rules", {})

    per_p: Dict[str, List[SEvent]] = {}
    for ev in events:
        p = ev.extendedProps.get("provider")
        if p: per_p.setdefault(p, []).append(ev)

    day_prov_counts: Dict[tuple, int] = {}
    day_shift_counts: Dict[tuple, int] = {}
    for ev in events:
        p = ev.extendedProps.get("provider")
        skey = ev.extendedProps.get("shift_key")
        day = ev.start.date()
        if p:    day_prov_counts[(day, p)] = day_prov_counts.get((day, p), 0) + 1
        if skey: day_shift_counts[(day, skey)] = day_shift_counts.get((day, skey), 0) + 1

    # Month-aware base default
    base_default = recommended_max_shifts_for_month()

    for p, evs in per_p.items():
        evs.sort(key=lambda e: e.start)
        pr = prov_rules.get(p, {})
        # Effective max shifts = provider override OR month default
        eff_max = pr.get("max_shifts", base_default)

        # If provider has any vacation in this month → auto reduce by 3
        if _provider_has_vacation_in_month(pr):
            eff_max = max(0, (eff_max or 0) - 3)

        max_nights = pr.get("max_nights", rules.max_nights_per_provider)
        min_rest   = pr.get("min_rest_hours", rules.min_rest_hours_between_shifts)

        # Unavailable = specific dates + expanded vacation days
        import pandas as pd
        unavail_set = set()
        for tok in pr.get("unavailable_dates", []):
            try: unavail_set.add(pd.to_datetime(tok).date())
            except Exception: pass
        unavail_set |= _expand_vacation_dates(pr.get("vacations", []))

        # 1) Max shifts
        if eff_max is not None and len(evs) > eff_max:
            violations.setdefault(p, []).append(f"Has {len(evs)} shifts > max {eff_max}")

        # 2) Rest hours
        for a, b in zip(evs, evs[1:]):
            rest = (b.start - a.end).total_seconds()/3600
            if rest < (min_rest or 0):
                violations.setdefault(p, []).append(
                    f"Rest {rest:.1f}h < min {min_rest}h between {a.start:%m-%d} and {b.start:%m-%d}"
                )

        # 3) Max nights
        if max_nights is not None:
            nights = sum(1 for ev in evs if ev.extendedProps.get("shift_key") == "N12")
            if nights > max_nights:
                violations.setdefault(p, []).append(f"Nights {nights} > max {max_nights}")

        # 4) Weekend requirement
        if rules.require_at_least_one_weekend:
            if not any(ev.start.weekday() >= 5 for ev in evs):
                violations.setdefault(p, []).append("No weekend shifts")

        # 5) One shift per day
        for (d, pp), cnt in day_prov_counts.items():
            if pp == p and cnt > 1:
                violations.setdefault(p, []).append(f"{d:%Y-%m-%d}: {cnt} shifts in one day (limit 1)")

        # 6) Eligibility
        allowed = prov_caps.get(p, [])
        if allowed:
            bad = [ev for ev in evs if ev.extendedProps.get("shift_key") not in allowed]
            if bad:
                bad_keys = sorted(set(ev.extendedProps.get("shift_key") for ev in bad))
                violations.setdefault(p, []).append(f"Not eligible for: {', '.join(bad_keys)}")

        # 7) Unavailability (dates + vacations)
        if unavail_set:
            bad_dates = sorted({ev.start.date() for ev in evs if ev.start.date() in unavail_set})
            for d in bad_dates:
                violations.setdefault(p, []).append(f"{d:%Y-%m-%d}: provider unavailable")

    # Day/shift capacity checks (GLOBAL)
    for (d, skey), cnt in day_shift_counts.items():
        cap = cap_map.get(skey, 1)
        if cnt > cap:
            violations.setdefault("GLOBAL", []).append(f"{d:%Y-%m-%d} {skey}: {cnt} assigned > capacity {cap}")

    return violations



def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    sdefs = {s["key"]: s for s in shift_types}
    stypes = [s["key"] for s in shift_types]
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: Dict[str, List[str]] = st.session_state.get("provider_caps", {})
    prov_rules: Dict[str, Dict[str, Any]] = st.session_state.get("provider_rules", {})

    counts = {p: 0 for p in providers}
    nights = {p: 0 for p in providers}
    events: List[SEvent] = []

    # Month-aware base default
    base_default = recommended_max_shifts_for_month()

    def day_shift_count(d: date, skey: str) -> int:
        return sum(1 for e in events if e.extendedProps.get("shift_key") == skey and e.start.date() == d)

    def provider_has_shift_on_day(p: str, d: date) -> bool:
        return any(e.extendedProps.get("provider") == p and e.start.date() == d for e in events)

    def ok(p: str, d: date, skey: str) -> bool:
        allowed = prov_caps.get(p, [])
        if allowed and skey not in allowed:
            return False

        pr = prov_rules.get(p, {})
        eff_max = pr.get("max_shifts", base_default)
        if _provider_has_vacation_in_month(pr):
            eff_max = max(0, (eff_max or 0) - 3)

        max_nights = pr.get("max_nights", rules.max_nights_per_provider)
        min_rest   = pr.get("min_rest_hours", rules.min_rest_hours_between_shifts)

        # Unavailability set (dates + vacations)
        unavail_set = _expand_vacation_dates(pr.get("vacations", []))
        # also add specific dates
        import pandas as pd
        for tok in pr.get("unavailable_dates", []):
            try: unavail_set.add(pd.to_datetime(tok).date())
            except Exception: pass
        if d in unavail_set:
            return False

        if day_shift_count(d, skey) >= cap_map.get(skey, 1):
            return False
        if provider_has_shift_on_day(p, d):
            return False
        if eff_max is not None and counts[p] + 1 > eff_max:
            return False
        if skey == "N12" and max_nights is not None and nights[p] + 1 > max_nights:
            return False

        sdef = sdefs[skey]
        start_dt = datetime.combine(d, parse_time(sdef["start"]))
        end_dt   = datetime.combine(d, parse_time(sdef["end"]))
        if end_dt <= start_dt: end_dt += timedelta(days=1)
        for e in [e for e in events if e.extendedProps.get("provider") == p]:
            rest1 = (start_dt - e.end).total_seconds()/3600
            rest2 = (e.start - end_dt).total_seconds()/3600
            if - (min_rest or 0) < rest1 < (min_rest or 0): return False
            if - (min_rest or 0) < rest2 < (min_rest or 0): return False
        return True

    p_idx = 0
    for d in days:
        for skey in stypes:
            capacity = cap_map.get(skey, 1)
            for _ in range(capacity):
                assigned = False; tries = 0
                while not assigned and tries < len(providers):
                    p = providers[p_idx % len(providers)]
                    if ok(p, d, skey):
                        sdef = sdefs[skey]
                        start_dt = datetime.combine(d, parse_time(sdef["start"]))
                        end_dt   = datetime.combine(d, parse_time(sdef["end"]))
                        if end_dt <= start_dt: end_dt += timedelta(days=1)
                        ev = SEvent(
                            id=str(uuid.uuid4()),
                            title=f"{sdef['label']} — {p}",
                            start=start_dt,
                            end=end_dt,
                            backgroundColor=sdef.get("color"),
                            extendedProps={"provider": p, "shift_key": skey, "label": sdef["label"]},
                        )
                        events.append(ev)
                        counts[p] += 1
                        if skey == "N12": nights[p] += 1
                        assigned = True
                    else:
                        p_idx += 1; tries += 1
                p_idx += 1
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

    with st.sidebar.expander("Replace entire list", expanded=False):
        repl = st.text_area("Paste full roster (will replace all)", value="\n".join(current_list), key="replace_all_area")
        if st.button("Replace list", key="btn_replace_all"):
            new_roster = _normalize_initials_list(repl.replace(",", "\n").split())
            if not new_roster:
                st.warning("Replacement list is empty — keeping current roster.")
            else:
                st.session_state.providers_df = pd.DataFrame({"initials": new_roster})
                st.session_state["provider_caps"] = {k: v for k, v in st.session_state.provider_caps.items() if k in new_roster}
                st.toast("Provider roster replaced.", icon="♻️")

    # ===================== Rules =====================
    st.sidebar.subheader("Rules")
    rc = RuleConfig(**st.session_state.get("rules", RuleConfig().dict()))
    rc.max_shifts_per_provider = st.sidebar.number_input("Max shifts/provider", 1, 31, value=int(rc.max_shifts_per_provider))
    rc.min_rest_hours_between_shifts = st.sidebar.number_input("Min rest hours between shifts", 0, 48, value=int(rc.min_rest_hours_between_shifts))
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

    # ===================== Provider shift eligibility =====================
    st.sidebar.subheader("Provider shift eligibility")
    with st.sidebar.expander("Assign allowed shift types per provider", expanded=False):
        label_for_key = {s["key"]: s["label"] for s in st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES.copy())}
        key_for_label = {v: k for k, v in label_for_key.items()}
        all_shift_labels = list(label_for_key.values())

        roster = st.session_state.providers_df["initials"].astype(str).tolist() if not st.session_state.providers_df.empty else []
        if not roster:
            st.caption("Add providers to configure eligibility.")
        else:
            filt = st.text_input("Filter providers (by initials)", value="", key="elig_filter").strip().upper()
            view_roster = [p for p in roster if filt in p] if filt else roster
            st.caption("Tip: leave a provider empty to allow ALL shift types for that provider.")

            for init in view_roster:
                allowed_keys = st.session_state.provider_caps.get(init, None)
                default_labels = [label_for_key[k] for k in (allowed_keys or []) if k in label_for_key]

                c1, c2 = st.columns([3, 1])
                with c1:
                    selected_labels = st.multiselect(
                        init,
                        options=all_shift_labels,
                        default=default_labels,
                        key=f"elig_{init}"
                    )
                with c2:
                    if st.button("All shifts", key=f"elig_all_{init}", help="Remove restrictions for this provider"):
                        if init in st.session_state.provider_caps:
                            del st.session_state.provider_caps[init]
                        st.session_state[f"elig_{init}"] = []
                        st.experimental_rerun()

                # Persist: empty or all selected => no restriction
                if len(selected_labels) == 0 or len(selected_labels) == len(all_shift_labels):
                    if init in st.session_state.provider_caps:
                        del st.session_state.provider_caps[init]
                else:
                    st.session_state.provider_caps[init] = [key_for_label[lbl] for lbl in selected_labels]


              
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

    with st.expander("Replace entire list", expanded=False):
        repl = st.text_area("Paste full roster (will replace all)", value="\n".join(current_list), key="replace_all_area")
        if st.button("Replace list", key="btn_replace_all"):
            new_roster = _normalize_initials_list(repl.replace(",", "\n").split())
            if new_roster:
                st.session_state.providers_df = pd.DataFrame({"initials": new_roster})
                st.session_state["provider_caps"] = {
                    k: v for k, v in st.session_state.provider_caps.items() if k in new_roster
                }
                st.toast("Provider roster replaced.", icon="♻️")
            else:
                st.warning("Replacement list is empty — keeping current roster.")

    # ===== Global rules =====
    st.subheader("Rules (global)")
    rc = RuleConfig(**st.session_state.rules)
    rc.max_shifts_per_provider = st.number_input("Max shifts/provider", 1, 50, value=int(rc.max_shifts_per_provider))
    rc.min_rest_hours_between_shifts = st.number_input("Min rest hours between shifts", 0, 48, value=int(rc.min_rest_hours_between_shifts))
    rc.min_block_size = st.number_input("Preferred block size (days)", 1, 7, value=int(rc.min_block_size))
    rc.require_at_least_one_weekend = st.checkbox("Require at least one weekend shift", value=bool(rc.require_at_least_one_weekend))
    limit_nights = st.checkbox("Limit 7pm–7am (N12) nights per provider", value=st.session_state.rules.get("max_nights_per_provider", 6) is not None)
    if limit_nights:
        default_nights = int(st.session_state.rules.get("max_nights_per_provider", 6) or 0)
        rc.max_nights_per_provider = st.number_input("Max nights/provider", 0, 50, value=default_nights)
    else:
        rc.max_nights_per_provider = None
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

    # ===== Provider eligibility (bulk) =====
    st.subheader("Provider shift eligibility (bulk)")
    with st.expander("Assign allowed shift types per provider", expanded=False):
        label_for_key = {s["key"]: s["label"] for s in st.session_state.shift_types}
        key_for_label = {v: k for k, v in label_for_key.items()}
        roster = st.session_state.providers_df["initials"].astype(str).tolist()
        if not roster:
            st.caption("Add providers to configure eligibility.")
        else:
            filt = st.text_input("Filter providers (by initials)", value="", key="elig_filter").strip().upper()
            view_roster = [p for p in roster if filt in p] if filt else roster
            st.caption("Tip: leave empty or select all to allow ALL shifts.")
            for init in view_roster:
                allowed_keys = st.session_state.provider_caps.get(init, [])
                default_labels = [label_for_key[k] for k in allowed_keys if k in label_for_key]
                selected = st.multiselect(init, options=list(label_for_key.values()), default=default_labels, key=f"elig_{init}")
                if len(selected) == 0 or len(selected) == len(label_for_key):
                    st.session_state.provider_caps.pop(init, None)
                else:
                    st.session_state.provider_caps[init] = [key_for_label[lbl] for lbl in selected]

    # ===== Actions =====
    st.subheader("Actions")

    # Month nav row
    nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("◀ Prev month"):
            m = st.session_state.month
            y = m.year - (1 if m.month == 1 else 0)
            mm = 12 if m.month == 1 else m.month - 1
            st.session_state.month = date(y, mm, 1)
    with nav_label:
        st.markdown(f"<div style='text-align:center;font-weight:600'>{st.session_state.month:%B %Y}</div>", unsafe_allow_html=True)
    with nav_next:
        if st.button("Next month ▶"):
            m = st.session_state.month
            y = m.year + (1 if m.month == 12 else 0)
            mm = 1 if m.month == 12 else m.month + 1
            st.session_state.month = date(y, mm, 1)

    # Action buttons row (no JSON download)
    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("Generate Draft from Rules"):
            providers = st.session_state.providers_df["initials"].astype(str).tolist()
            days = make_month_days(st.session_state.month.year, st.session_state.month.month)
            evs = assign_greedy(providers, days, st.session_state.shift_types, RuleConfig(**st.session_state.rules))
            # preserve events outside this month
            def is_this_month(e):
                try:
                    d = pd.to_datetime(e["start"]).date()
                    return d.year == st.session_state.month.year and d.month == st.session_state.month.month
                except Exception:
                    return False
            keep_others = [E for E in st.session_state.events if not is_this_month(E)]
            new_json = [e.to_json_event() if hasattr(e, "to_json_event") else e for e in evs]
            st.session_state.events = events_for_calendar(keep_others + new_json)
            st.success("Draft generated.")
    with act2:
        if st.button("Validate schedule"):
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
                        extendedProps={"provider": ext.get("provider"), "shift_key": ext.get("shift_key"), "label": ext.get("label")},
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
        if st.button("Clear month"):
            def is_this_month(e):
                try:
                    d = pd.to_datetime(e["start"]).date()
                    return d.year == st.session_state.month.year and d.month == st.session_state.month.month
                except Exception:
                    return False
            st.session_state.events = [E for E in st.session_state.events if not is_this_month(E)]
            st.toast("Cleared this month.", icon="🧹")



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


# provider rules section
# make sure this version is in your codebase
def provider_rules_panel():
    st.header("Provider-specific rules")

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

    # ... (rest of the rules editor as previously provided)

    # Backing stores
    # ... (keep your header and early `sel` checks)

rules_map = st.session_state.setdefault("provider_rules", {})
global_rules = get_global_rules()




# Allowed shifts (unchanged)
label_for_key, key_for_label = get_shift_label_maps()

current_allowed = st.session_state.get("provider_caps", {}).get(sel, [])
default_labels = [label_for_key[k] for k in current_allowed if k in label_for_key]

st.subheader(f"Allowed shift types — {sel}")
picked_labels = st.multiselect(
    "Assign only these shift types (leave empty to allow ALL)",
    options=list(label_for_key.values()),
    default=default_labels,
    key=f"pr_allowed_{sel}"
)
if len(picked_labels) == 0:
    st.session_state["provider_caps"].pop(sel, None)
else:
    st.session_state["provider_caps"][sel] = [key_for_label[lbl] for lbl in picked_labels]

st.markdown("---")
st.subheader("Overrides (optional)")
curr = rules_map.get(sel, {})
c1, c2 = st.columns(2)
with c1:
    use_max = st.checkbox("Override max shifts / month", value=("max_shifts" in curr))
    # Show month-aware recommended default for context
    st.caption(f"Recommended default this month: **{recommended_max_shifts_for_month()}**")
    max_sh = st.number_input("Max shifts (this month)", 1, 50,
                             value=int(curr.get("max_shifts", recommended_max_shifts_for_month())))
with c2:
    use_nights = st.checkbox("Override max nights / month", value=("max_nights" in curr))
    default_max_n = global_rules.max_nights_per_provider if global_rules.max_nights_per_provider is not None else 0
    max_n  = st.number_input("Max nights (this month)", 0, 50,
                             value=int(curr.get("max_nights", default_max_n)))

use_rest = st.checkbox("Override min rest hours", value=("min_rest_hours" in curr))
min_rest = st.number_input("Min rest hours between shifts", 0, 48,
                           value=int(curr.get("min_rest_hours", global_rules.min_rest_hours_between_shifts)))

st.markdown("---")
st.subheader("Unavailable specific dates")
dates_txt = st.text_input(
    "YYYY-MM-DD, comma-separated",
    value=",".join(curr.get("unavailable_dates", [])),
    key=f"pr_unavail_{sel}"
)

# >>> NEW: Vacations (date ranges) <<<
st.markdown("---")
st.subheader("Vacations (date ranges)")
vac_list = curr.get("vacations", [])
if not isinstance(vac_list, list):
    vac_list = []

# add input row
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

# show existing vacations with remove buttons
if vac_list:
    for i, rng in enumerate(vac_list):
        rr1, rr2, rr3 = st.columns([2, 2, 1])
        rr1.markdown(f"**Start:** {rng.get('start','')}")
        rr2.markdown(f"**End:** {rng.get('end','')}")
        if rr3.button("Remove", key=f"pr_vac_del_{sel}_{i}"):
            vac_list.pop(i)
            curr["vacations"] = vac_list
            rules_map[sel] = curr
            st.experimental_rerun()

st.text_area("Notes (optional)", value=curr.get("notes", ""), key=f"pr_notes_{sel}")

if st.button("Save provider rules", key=f"pr_save_{sel}"):
    new_entry = {}
    if use_max:    new_entry["max_shifts"] = int(max_sh)
    if use_nights: new_entry["max_nights"] = int(max_n)
    if use_rest:   new_entry["min_rest_hours"] = int(min_rest)

    # normalize dates
    import pandas as pd
    unavail = []
    for tok in [t.strip() for t in dates_txt.split(",") if t.strip()]:
        try:
            unavail.append(str(pd.to_datetime(tok).date()))
        except Exception:
            pass
    if unavail:
        new_entry["unavailable_dates"] = unavail

    if vac_list:
        new_entry["vacations"] = vac_list

    notes_val = st.session_state.get(f"pr_notes_{sel}", "")
    if notes_val:
        new_entry["notes"] = notes_val

    if new_entry:
        rules_map[sel] = new_entry
    else:
        rules_map.pop(sel, None)
    st.success("Saved.")



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
                return "background-color: #fff3bf;" if str(val).strip().upper() == hi else ""
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
            st.session_state.events = events_for_calendar(st.session_state.events)

            sdefs = {s["key"]: s for s in st.session_state.shift_types}

            # keep comments by (date, shift_key, provider)
            comments_by_key = {}
            for e in st.session_state.events:
                ext = (e.get("extendedProps") or {}); skey = ext.get("shift_key")
                if not skey or skey not in sdefs: continue
                try:
                    d = pd.to_datetime(e["start"]).date()
                except Exception:
                    continue
                if d.year == year and d.month == month:
                    prov0 = (ext.get("provider") or "").strip().upper()
                    comments_by_key[(d, skey, prov0)] = list(st.session_state.comments.get(e["id"], []))

            def is_grid_event(E: Dict[str, Any]) -> bool:
                ext = (E.get("extendedProps") or {}); skey = ext.get("shift_key")
                if not skey or skey not in sdefs: return False
                try:
                    d = pd.to_datetime(E["start"]).date()
                except Exception:
                    return False
                return d.year == year and d.month == month

            preserved = [E for E in st.session_state.events if not is_grid_event(E)]

            new_events = []
            seen_day_provider = set()
            conflicts = []

            row_to_key = {rm["row_label"]: rm["skey"] for rm in row_meta}
            day_only_cols = [c for c in edited_grid.columns if c.isdigit()]

            for row_label in edited_grid.index:
                skey = row_to_key.get(row_label)
                if not skey: continue
                sdef = sdefs.get(skey)
                if not sdef: continue
                for col in day_only_cols:
                    prov = edited_grid.at[row_label, col]
                    prov = ("" if prov is None else str(prov)).strip().upper()
                    if not prov: continue

                    day_date = date(year, month, int(col))
                    key_dp = (day_date, prov)
                    if key_dp in seen_day_provider:
                        conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (duplicate; skipped)")
                        continue
                    seen_day_provider.add(key_dp)

                    def _parse(hhmm: str):
                        hh, mm = hhmm.split(":"); return time(int(hh), int(mm))
                    start_dt = datetime.combine(day_date, _parse(sdef["start"]))
                    end_dt   = datetime.combine(day_date, _parse(sdef["end"]))
                    if end_dt <= start_dt: end_dt += timedelta(days=1)

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
                    k = (day_date, skey, prov)
                    if k in st.session_state.comments:
                        st.session_state.comments[eid] = st.session_state.comments[k]
                    elif k in comments_by_key:
                        st.session_state.comments[eid] = comments_by_key[k]

            st.session_state.events = preserved + new_events
            st.session_state.events = [_event_to_dict(e) for e in st.session_state.events]

            if conflicts:
                st.warning("Some duplicates were skipped:\n- " + "\n- ".join(conflicts))
            else:
                st.success("Applied grid to calendar.")




# -------------------------
# App entry
# -------------------------

def main():
    init_session_state()
    left_col, mid_col, right_col = st.columns([3,5,3], gap="large")
    with left_col:  engine_panel()
    with mid_col:   render_calendar(); schedule_grid_view()
    with right_col: provider_rules_panel()

main()





























