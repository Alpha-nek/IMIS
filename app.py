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
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("comments", {})
    st.session_state.setdefault("month", date.today().replace(day=1))
    st.session_state.setdefault("rules", RuleConfig().dict())
    st.session_state.setdefault("highlight_provider", "")
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
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
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
    cap_map: Dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)

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
            st.caption("Tip: leave a provider empty (or select all shifts) to allow ALL shift types for that provider.")

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
    st.subheader("Monthly Grid — Shifts × Days (one provider per cell)")

    if not st.session_state.shift_types:
        st.info("No shift types configured.")
        return

    # --- helpers
    def tod_group_and_order(skey: str, sdef: Dict[str, Any]):
        # group by time-of-day (known keys) else fallback by start hour
        start = parse_time(sdef["start"])
        if skey in ("R12", "A12"):
            return "Day (07:00–19:00)", 1
        if skey == "A10":
            return "Evening (10:00–22:00)", 2
        if skey == "N12":
            return "Night (19:00–07:00)", 3
        if skey == "NB":
            return "Late Night (23:00–03:00)", 4
        if 5 <= start.hour < 12:   return "Day", 1
        if 12 <= start.hour < 18:  return "Evening", 2
        return "Night", 3

    def start_minutes(sdef):
        t = parse_time(sdef["start"])
        return t.hour * 60 + t.minute

    # --- month context
    year  = st.session_state.month.year
    month = st.session_state.month.month
    days  = make_month_days(year, month)
    day_cols = [str(d.day) for d in days]

    # --- rows: one row per slot of each shift (capacity)
    stypes  = st.session_state.shift_types
    cap_map = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)


    row_meta = []  # each item: {"row_label","skey","sdef","slot","group","gorder"}
    for s in stypes:
        skey = s["key"]
        cap  = int(cap_map.get(skey, 1))
        group_label, gorder = tod_group_and_order(skey, s)
        for slot in range(1, cap + 1):
            # flat row label; easier/safer for data_editor
            row_label = f"[{group_label}] {skey} — {s['label']}  (slot {slot})"
            row_meta.append({
                "row_label": row_label, "skey": skey, "sdef": s,
                "slot": slot, "group": group_label, "gorder": gorder
            })

    # sort rows: group → start time → key → slot
    row_meta.sort(key=lambda r: (r["gorder"], start_minutes(r["sdef"]), r["skey"], r["slot"]))

    # --- build grid (strings only, one provider per cell)
    import pandas as pd
    grid = pd.DataFrame("", index=[rm["row_label"] for rm in row_meta], columns=day_cols, dtype="object")

    # index shift rows for filling
    rows_for_key = {}
    for rm in row_meta:
        rows_for_key.setdefault(rm["skey"], []).append(rm["row_label"])

    # fill from existing events -> first empty slot row for that shift/day
    for e in st.session_state.events:
        ext  = (e.get("extendedProps") or {})
        skey = ext.get("shift_key")
        if not skey:
            continue
        try:
            d = pd.to_datetime(e["start"]).date()
        except Exception:
            continue
        if d.year != year or d.month != month:
            continue
        prov = (ext.get("provider") or "").strip().upper() or "UNASSIGNED"
        col  = str(d.day)
        for row_label in rows_for_key.get(skey, []):
            if grid.at[row_label, col] == "":
                grid.at[row_label, col] = prov
                break
        # if all slots full: ignore (grid represents capacity)

    # dropdown options = known providers (plus blank)
    valid_provs = (st.session_state.providers_df["initials"].astype(str).str.upper().unique().tolist()
                   if not st.session_state.providers_df.empty else [])
    valid_provs = sorted(valid_provs)

    st.caption("Each cell holds a single provider (blank = unassigned). "
               "Use the sidebar ‘Daily shift capacities’ to add rows per shift type. "
               "Rows are grouped by time-of-day (Day → Evening → Night → Late Night).")

    # some Streamlit versions don’t have SelectboxColumn; keep it optional
    try:
        col_config = {c: st.column_config.SelectboxColumn(options=[""] + valid_provs,
                                                          help=f"Assignments for day {c}")
                      for c in day_cols}
    except Exception:
        col_config = None

    # render editor (flat index → fewer runtime issues)
    edited_grid = st.data_editor(
        grid,
        num_rows="fixed",
        use_container_width=True,
        height=560,
        column_config=col_config,
        key="grid_editor",
    )

    # map back to events
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply grid to calendar"):
            sdefs = {s["key"]: s for s in st.session_state.shift_types}

            # keep comments by (date, shift_key, provider)
            comments_by_key = {}
            for e in st.session_state.events:
                ext  = (e.get("extendedProps") or {})
                skey = ext.get("shift_key")
                if not skey or skey not in sdefs:
                    continue
                try:
                    d = pd.to_datetime(e["start"]).date()
                except Exception:
                    continue
                if d.year == year and d.month == month:
                    prov = (ext.get("provider") or "").strip().upper()
                    comments_by_key[(d, skey, prov)] = list(st.session_state.comments.get(e["id"], []))

            # keep non-grid events
            def is_grid_event(E: Dict[str, Any]) -> bool:
                ext  = (E.get("extendedProps") or {})
                skey = ext.get("shift_key")
                if not skey or skey not in sdefs:
                    return False
                try:
                    d = pd.to_datetime(E["start"]).date()
                except Exception:
                    return False
                return d.year == year and d.month == month

            preserved = [E for E in st.session_state.events if not is_grid_event(E)]

            # rebuild from edited grid, enforcing one shift/provider/day
            new_events = []
            seen_day_provider = set()
            conflicts = []

            # quick lookup from row_label → shift_key
            row_to_key = {rm["row_label"]: rm["skey"] for rm in row_meta}

            for row_label in edited_grid.index:
                skey = row_to_key.get(row_label)
                if not skey:
                    continue
                sdef = sdefs.get(skey)
                if not sdef:
                    continue
                for col in edited_grid.columns:
                    prov = edited_grid.at[row_label, col]
                    if prov is None:
                        prov = ""
                    prov = str(prov).strip().upper()
                    if not prov:
                        continue

                    day_date = date(year, month, int(col))
                    # enforce: one shift per provider per day
                    key_dp = (day_date, prov)
                    if key_dp in seen_day_provider:
                        conflicts.append(f"{day_date:%Y-%m-%d} — {prov} (duplicate; skipped)")
                        continue
                    seen_day_provider.add(key_dp)

                    # build start/end
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

                    # reattach comments
                    k = (day_date, skey, prov)
                    if k in comments_by_key:
                        st.session_state.comments[eid] = comments_by_key[k]

            st.session_state.events = preserved + new_events
            if conflicts:
                st.warning("Some duplicates were skipped:\n- " + "\n- ".join(conflicts))
            else:
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










