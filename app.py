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

import json
import uuid
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
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "03:00", "color": "#06b6d4"},
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

class RuleConfig(BaseModel):
    max_shifts_per_provider: int = Field(12, ge=1, le=31)
    min_rest_hours_between_shifts: int = Field(12, ge=0, le=48)
    min_block_size: int = Field(2, ge=1, le=7, description="Minimum consecutive days in a block when possible")
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

# -------------------------
# State helpers
# -------------------------

def init_session_state():
    st.session_state.setdefault("shift_types", DEFAULT_SHIFT_TYPES.copy())
    st.session_state.setdefault("providers_df", pd.DataFrame(columns=["initials"]))
    st.session_state.setdefault("events", [])  # list of dicts (FullCalendar JSON)
    st.session_state.setdefault("comments", {})  # event_id -> list[str]
    st.session_state.setdefault("month", date.today().replace(day=1))
    st.session_state.setdefault("rules", RuleConfig().dict())
    st.session_state.setdefault("highlight_provider", "")

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
    """Return {provider: [violation, ...]} and a GLOBAL bucket for day-level issues."""
    violations: Dict[str, List[str]] = {}
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", {}) or {}
    prov_caps: Dict[str, List[str]] = st.session_state.get("provider_caps", {}) or {}

    # Build per-provider schedules
    per_p: Dict[str, List[SEvent]] = {}
    for ev in events:
        p = ev.extendedProps.get("provider")
        if not p:
            continue
        per_p.setdefault(p, []).append(ev)

    # Per-day counts
    day_prov_counts: Dict[tuple, int] = {}
    day_shift_counts: Dict[tuple, int] = {}
    for ev in events:
        p = ev.extendedProps.get("provider")
        skey = ev.extendedProps.get("shift_key")
        day = ev.start.date()
        if p:
            day_prov_counts[(day, p)] = day_prov_counts.get((day, p), 0) + 1
        if skey:
            day_shift_counts[(day, skey)] = day_shift_counts.get((day, skey), 0) + 1

    # Provider-level checks
    for p, evs in per_p.items():
        evs.sort(key=lambda e: e.start)
        # 1) Max shifts
        if len(evs) > rules.max_shifts_per_provider:
            violations.setdefault(p, []).append(
                f"Has {len(evs)} shifts > max {rules.max_shifts_per_provider}")
        # 2) Rest hours between consecutive shifts
        for a, b in zip(evs, evs[1:]):
            rest = (b.start - a.end).total_seconds() / 3600
            if rest < rules.min_rest_hours_between_shifts:
                violations.setdefault(p, []).append(
                    f"Rest {rest:.1f}h < min {rules.min_rest_hours_between_shifts}h between {a.start:%m-%d} and {b.start:%m-%d}")
        # 3) Max nights
        if rules.max_nights_per_provider is not None:
            nights = sum(1 for ev in evs if ev.extendedProps.get("shift_key") == "N12")
            if nights > rules.max_nights_per_provider:
                violations.setdefault(p, []).append(
                    f"Nights {nights} > max {rules.max_nights_per_provider}")
        # 4) At least one weekend
        if rules.require_at_least_one_weekend:
            worked_weekend = any(ev.start.weekday() >= 5 for ev in evs)
            if not worked_weekend:
                violations.setdefault(p, []).append("No weekend shifts")
        # 5) One shift per day per provider
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

    # Day/shift capacity checks (GLOBAL)
    for (d, skey), cnt in day_shift_counts.items():
        cap = cap_map.get(skey, 1)
        if cnt > cap:
            violations.setdefault("GLOBAL", []).append(f"{d:%Y-%m-%d} {skey}: {cnt} assigned > capacity {cap}")

    return violations

def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    """Round-robin with constraints, capacity per shift/day, eligibility, and one-shift-per-day."""
    sdefs = {s["key"]: s for s in shift_types}
    stypes = [s["key"] for s in shift_types]
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", {}) or {}
    prov_caps: Dict[str, List[str]] = st.session_state.get("provider_caps", {}) or {}

    # Track counters
    counts = {p: 0 for p in providers}
    nights = {p: 0 for p in providers}

    events: List[SEvent] = []

    def day_shift_count(d: date, skey: str) -> int:
        return sum(1 for e in events if e.extendedProps.get("shift_key") == skey and e.start.date() == d)

    def provider_has_shift_on_day(p: str, d: date) -> bool:
        return any(e.extendedProps.get("provider") == p and e.start.date() == d for e in events)

    def ok(p: str, d: date, skey: str) -> bool:
        # Eligibility
        allowed = prov_caps.get(p, [])
        if allowed and skey not in allowed:
            return False
        # Capacity for this day/shift
        if day_shift_count(d, skey) >= cap_map.get(skey, 1):
            return False
        # One shift per day per provider
        if provider_has_shift_on_day(p, d):
            return False
        # Count caps
        if counts[p] + 1 > rules.max_shifts_per_provider:
            return False
        if skey == "N12" and rules.max_nights_per_provider is not None:
            if nights[p] + 1 > rules.max_nights_per_provider:
                return False
        # Rest constraint (against existing events)
        sdef = sdefs[skey]
        start_dt = datetime.combine(d, parse_time(sdef["start"]))
        end_dt = datetime.combine(d, parse_time(sdef["end"]))
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        for e in [e for e in events if e.extendedProps.get("provider") == p]:
            rest1 = (start_dt - e.end).total_seconds() / 3600
            rest2 = (e.start - end_dt).total_seconds() / 3600
            if -rules.min_rest_hours_between_shifts < rest1 < rules.min_rest_hours_between_shifts:
                return False
            if -rules.min_rest_hours_between_shifts < rest2 < rules.min_rest_hours_between_shifts:
                return False
        return True

    # Assignment with preferred block size
    p_idx = 0
    for d in days:
        for skey in stypes:
            capacity = cap_map.get(skey, 1)
            for _slot in range(capacity):
                assigned = False
                tries = 0
                while not assigned and tries < len(providers):
                    p = providers[p_idx % len(providers)]
                    if ok(p, d, skey):
                        # Assign block across consecutive days
                        for bday_i in range(rules.min_block_size):
                            bday = d + timedelta(days=bday_i)
                            if bday not in days:
                                break
                            if not ok(p, bday, skey):
                                break
                            sdef = sdefs[skey]
                            start_dt = datetime.combine(bday, parse_time(sdef["start"]))
                            end_dt = datetime.combine(bday, parse_time(sdef["end"]))
                            if end_dt <= start_dt:
                                end_dt += timedelta(days=1)
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
                            if skey == "N12":
                                nights[p] += 1
                        assigned = True
                    else:
                        p_idx += 1
                        tries += 1
                p_idx += 1

    return events

# -------------------------
# UI
# -------------------------

def sidebar_inputs():
    st.sidebar.header("Providers & Rules")

    # Provider input
    st.sidebar.subheader("Providers")
    prov_upload = st.sidebar.file_uploader("Upload CSV with 'initials' column", type=["csv"], key="prov_up")
    if prov_upload:
        df = pd.read_csv(prov_upload)
        if "initials" not in df.columns:
            st.sidebar.error("CSV must include an 'initials' column")
        else:
            st.session_state.providers_df = pd.DataFrame({"initials": df["initials"].astype(str).str.upper().str.strip()}).drop_duplicates()
    st.sidebar.caption("Or paste initials below (comma/space/newline separated)")
    pasted = st.sidebar.text_area("Initials", value=" ".join(st.session_state.providers_df["initials"].tolist()))
    if st.sidebar.button("Use pasted initials"):
        toks = [t.strip().upper() for t in pasted.replace(",", "
").splitlines() if t.strip()]
        st.session_state.providers_df = pd.DataFrame({"initials": sorted(set(toks))})

    # Rules
    st.sidebar.subheader("Rules")
    rc = RuleConfig(**st.session_state.rules)
    rc.max_shifts_per_provider = st.sidebar.number_input("Max shifts/provider", 1, 31, value=rc.max_shifts_per_provider)
    rc.min_rest_hours_between_shifts = st.sidebar.number_input("Min rest hours between shifts", 0, 48, value=rc.min_rest_hours_between_shifts)
    rc.min_block_size = st.sidebar.number_input("Preferred block size (days)", 1, 7, value=rc.min_block_size)
    rc.require_at_least_one_weekend = st.sidebar.checkbox("Require at least one weekend shift", value=rc.require_at_least_one_weekend)
    use_max_nights = st.sidebar.checkbox("Limit nights per provider", value=st.session_state.rules.get("max_nights_per_provider", 6) is not None)
    if use_max_nights:
        rc.max_nights_per_provider = st.sidebar.number_input("Max nights/provider", 0, 31, value=st.session_state.rules.get("max_nights_per_provider", 6) or 0)
    else:
        rc.max_nights_per_provider = None
    st.session_state.rules = rc.dict()

    # Shift Types
    st.sidebar.subheader("Shift Types")
    st.caption("Edit labels or times as needed. Colors are for on-screen clarity.")
    for i, s in enumerate(st.session_state.shift_types):
        with st.sidebar.expander(f"{s['label']}"):
            s["label"] = st.text_input("Label", value=s["label"], key=f"s_lbl_{i}")
            s["start"] = st.text_input("Start (HH:MM)", value=s["start"], key=f"s_st_{i}")
            s["end"] = st.text_input("End (HH:MM)", value=s["end"], key=f"s_en_{i}")
            s["color"] = st.color_picker("Color", value=s.get("color", "#3388ff"), key=f"s_co_{i}")

    # Daily shift capacities (e.g., A10 has capacity 2, NB has 1)
    st.sidebar.subheader("Daily shift capacities")
    cap_map = st.session_state.get("shift_capacity", {})
    for s in st.session_state.shift_types:
        key = s["key"]
        default_cap = cap_map.get(key, 1)
        cap_map[key] = st.sidebar.number_input(f"{s['label']} ({key}) capacity per day", 0, 10, value=int(default_cap))
    st.session_state.shift_capacity = cap_map

    # Provider shift eligibility
    st.sidebar.subheader("Provider shift eligibility")
    with st.sidebar.expander("Assign allowed shift types per provider"):
        if st.session_state.providers_df.empty:
            st.caption("Upload providers to configure eligibility.")
        else:
            label_for_key = {s["key"]: s["label"] for s in st.session_state.shift_types}
            key_for_label = {v: k for k, v in label_for_key.items()}
            for init in st.session_state.providers_df["initials"].tolist():
                allowed_keys = st.session_state.provider_caps.get(init, [])
                default_labels = [label_for_key[k] for k in allowed_keys if k in label_for_key]
                chosen = st.multiselect(init, options=list(key_for_label.keys()), default=default_labels, key=f"cap_{init}")
                st.session_state.provider_caps[init] = [key_for_label[lbl] for lbl in chosen]


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
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        if st.button("Generate Draft from Rules"):
            providers = st.session_state.providers_df["initials"].tolist()
            if not providers:
                st.warning("Add providers first.")
            else:
                rules = RuleConfig(**st.session_state.rules)
                days = make_month_days(st.session_state.month.year, st.session_state.month.month)
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
    with g4:
        st.download_button("Download JSON", data=json.dumps(st.session_state.events, indent=2), file_name=f"schedule_{st.session_state.month:%Y_%m}.json")

    # Upload/Load
    up = st.file_uploader("Load events JSON/CSV", type=["json", "csv"], key="events_up")
    if up is not None:
        if up.name.endswith(".json"):
            try:
                data = json.load(up)
                if isinstance(data, list):
                    st.session_state.events = data
                    st.success("Loaded JSON events.")
                else:
                    st.error("Invalid JSON format (expected a list).")
            except Exception as e:
                st.error(f"JSON load error: {e}")
        else:  # CSV
            try:
                df = pd.read_csv(up)
                req_cols = {"title", "start", "end"}
                if not req_cols.issubset(set(df.columns)):
                    st.error(f"CSV must include columns: {sorted(req_cols)}")
                else:
                    events = []
                    for _, r in df.iterrows():
                        ev = {
                            "id": str(uuid.uuid4()),
                            "title": r["title"],
                            "start": pd.to_datetime(r["start"]).to_pydatetime().isoformat(),
                            "end": pd.to_datetime(r["end"]).to_pydatetime().isoformat(),
                            "allDay": False,
                            "extendedProps": json.loads(r.get("extendedProps", "{}")) if isinstance(r.get("extendedProps"), str) else {},
                        }
                        events.append(ev)
                    st.session_state.events = events
                    st.success("Loaded CSV events.")
            except Exception as e:
                st.error(f"CSV load error: {e}")


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

    state = calendar(
        events=events,
        options=cal_options,
        custom_css=".fc {font-family: Inter, system-ui, sans-serif;} .fc-daygrid-event {border-radius: 8px;}",
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


def schedule_grid_view():
    st.subheader("Monthly Grid — Shifts × Days (Editable)")
    if not st.session_state.shift_types:
        st.info("No shift types configured.")
        return

    import re

    # Month context
    year = st.session_state.month.year
    month = st.session_state.month.month
    days = make_month_days(year, month)
    day_nums = [d.day for d in days]

    # Shift rows
    stypes = st.session_state.shift_types
    key_to_label = {s["key"]: f"{s['key']} — {s['label']}" for s in stypes}
    label_to_key = {v: k for k, v in key_to_label.items()}
    row_labels = [key_to_label[s["key"]] for s in stypes]

    # Build grid from existing events
    grid = pd.DataFrame("", index=row_labels, columns=day_nums)
    for e in st.session_state.events:
        ext = (e.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey or skey not in key_to_label:
            continue
        try:
            d = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        if d.year != year or d.month != month:
            continue
        prov = (ext.get("provider") or "").strip().upper() or "UNASSIGNED"
        r = key_to_label[skey]
        c = d.day
        prev = grid.at[r, c]
        grid.at[r, c] = (prev + ", " + prov) if prev else prov

    # Helper: show valid providers
    valid_provs = sorted(
        st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()
    ) if not st.session_state.providers_df.empty else []
    st.caption("Edit cells with provider initials (comma-separated for multiple).")
    if valid_provs:
        st.write("**Valid provider initials:**", ", ".join(valid_provs))

    # Make it editable
    edited_grid = st.data_editor(
        grid,
        num_rows="fixed",
        use_container_width=True,
        height=480,
        key="grid_editor",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Apply grid to calendar"):
            # Lookups
            sdefs = {s["key"]: s for s in st.session_state.shift_types}

            # Map old comments by (date, shift_key, provider) so we can reattach after rebuild
            comments_by_key = {}
            for e in st.session_state.events:
                ext = (e.get("extendedProps") or {})
                skey = ext.get("shift_key")
                if not skey or skey not in sdefs:
                    continue
                try:
                    d = pd.to_datetime(e["start"]).date()
                except Exception:
                    continue
                if d.year == year and d.month == month:
                    prov = (ext.get("provider") or "").strip().upper()
                    k = (d, skey, prov)
                    if e["id"] in st.session_state.comments:
                        comments_by_key[k] = list(st.session_state.comments.get(e["id"], []))

            # Keep events that are NOT part of this month's shift grid
            def is_grid_event(E: Dict[str, Any]) -> bool:
                ext = (E.get("extendedProps") or {})
                skey = ext.get("shift_key")
                if not skey or skey not in sdefs:
                    return False
                try:
                    d = pd.to_datetime(E["start"]).date()
                except Exception:
                    return False
                return d.year == year and d.month == month

            preserved = [E for E in st.session_state.events if not is_grid_event(E)]

            # Rebuild month events from edited grid
            new_events = []
            for row_label in edited_grid.index:
                skey = label_to_key.get(row_label)
                if not skey:
                    continue
                sdef = sdefs[skey]
                for col in edited_grid.columns:
                    cell = edited_grid.at[row_label, col]
                    if cell is None:
                        cell = ""
                    if not isinstance(cell, str):
                        cell = str(cell)

                    # Split tokens (commas, semicolons, or multiple spaces)
                    tokens = [t.strip().upper() for t in re.split(r"[,;/]+|\s{2,}", cell) if t.strip()]
                    if len(tokens) == 0 and cell.strip():
                        tokens = [cell.strip().upper()]  # single value without commas

                    day_date = date(year, month, int(col))

                    for prov in tokens:
                        # Build event datetimes (handle overnight)
                        def _parse(hhmm: str):
                            hh, mm = hhmm.split(":")
                            return time(int(hh), int(mm))
                        start_dt = datetime.combine(day_date, _parse(sdef["start"]))
                        end_dt = datetime.combine(day_date, _parse(sdef["end"]))
                        if end_dt <= start_dt:
                            end_dt += timedelta(days=1)

                        eid = str(uuid.uuid4())
                        ev = {
                            "id": eid,
                            "title": f"{sdef['label']} — {prov if prov else 'UNASSIGNED'}",
                            "start": start_dt.isoformat(),
                            "end": end_dt.isoformat(),
                            "allDay": False,
                            "backgroundColor": sdef.get("color"),
                            "extendedProps": {"provider": prov, "shift_key": skey, "label": sdef["label"]},
                        }
                        new_events.append(ev)

                        # Reattach comments if we had any for this (date, shift, provider)
                        k = (day_date, skey, prov)
                        if k in comments_by_key:
                            st.session_state.comments[eid] = comments_by_key[k]

            st.session_state.events = preserved + new_events
            st.success("Applied grid to calendar.")

    with c2:
        if st.button("Reload grid from calendar"):
            st.experimental_rerun()


# -------------------------
# App entry
# -------------------------

def main():
    init_session_state()
    sidebar_inputs()
    top_controls()
    render_calendar()
    schedule_grid_view()

if __name__ == "__main__":
    main()

