# =============================================================================
# IMIS Scheduler - Main Application
# =============================================================================
# 
# This is the main Streamlit application that imports from modular components.
# Run with: streamlit run main.py
# Vibe coded by Yazan Al-Fanek, MD

import streamlit as st
import sys
import os
import random

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all required libraries
import uuid
import json
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
from pydantic import BaseModel, Field, field_validator

try:
    from streamlit_calendar import calendar as st_calendar
except Exception:
    st_calendar = None

# Import our modular components
from models.constants import *
from models.data_models import *
from core.utils import *

# =============================================================================
# Additional Functions (from original app.py)
# =============================================================================

def _contiguous_blocks(dates: List[date]) -> List[Tuple[date, date, int]]:
    """Find contiguous blocks of dates."""
    if not dates:
        return []
    
    dates = sorted(dates)
    blocks = []
    start = dates[0]
    prev = dates[0]
    length = 1
    
    for d in dates[1:]:
        if (d - prev).days == 1:
            length += 1
            prev = d
        else:
            blocks.append((start, prev, length))
            start = d
            prev = d
            length = 1
    
    blocks.append((start, prev, length))
    return blocks

def get_app_shift_capacity(day: date) -> int:
    """Get APP shift capacity for a specific day."""
    if day.weekday() < 5:  # Weekday
        return 2
    else:  # Weekend
        return 1

def _event_to_dict(event: SEvent) -> Dict[str, Any]:
    """Convert SEvent to dictionary format for calendar."""
    return {
        "id": event.id,
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "backgroundColor": event.backgroundColor,
        "extendedProps": event.extendedProps,
    }

def events_for_calendar(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare events for calendar display."""
    calendar_events = []
    for e in events:
        # Add comment badge if there are comments
        title = e.get("title", "")
        event_id = e.get("id", "")
        comments = st.session_state.get("comments", {}).get(event_id, [])
        
        if comments:
            title += f' <span class="comment-badge">{len(comments)}</span>'
        
        calendar_event = {
            "id": e.get("id"),
            "title": title,
            "start": e.get("start"),
            "end": e.get("end"),
            "backgroundColor": e.get("backgroundColor"),
            "extendedProps": e.get("extendedProps", {}),
        }
        calendar_events.append(calendar_event)
    
    return calendar_events

def calculate_provider_statistics(events: List[SEvent]) -> Dict[str, Any]:
    """Calculate comprehensive provider statistics."""
    provider_stats = {}
    
    for event in events:
        provider = (event.extendedProps.get("provider") or "").strip().upper()
        if not provider:
            continue
            
        if provider not in provider_stats:
            provider_stats[provider] = {
                "total_shifts": 0,
                "weekend_shifts": 0,
                "night_shifts": 0,
                "day_shifts": 0,
                "shift_types": {},
                "dates": []
            }
        
        stats = provider_stats[provider]
        stats["total_shifts"] += 1
        stats["dates"].append(event.start.date())
        
        # Check if weekend
        if event.start.weekday() >= 5:
            stats["weekend_shifts"] += 1
        
        # Check shift type
        shift_key = event.extendedProps.get("shift_key", "")
        if shift_key in ["N12", "NB"]:
            stats["night_shifts"] += 1
        else:
            stats["day_shifts"] += 1
        
        # Count shift types
        if shift_key not in stats["shift_types"]:
            stats["shift_types"][shift_key] = 0
        stats["shift_types"][shift_key] += 1
    
    return {"provider_stats": provider_stats}

def identify_coverage_gaps(events: List[SEvent], shift_types: List[Dict], shift_capacity: Dict[str, int]) -> List[Dict[str, Any]]:
    """Identify coverage gaps in the schedule."""
    gaps = []
    
    # Group events by date
    events_by_date = {}
    for event in events:
        event_date = event.start.date()
        if event_date not in events_by_date:
            events_by_date[event_date] = []
        events_by_date[event_date].append(event)
    
    # Check each date for gaps
    for event_date, day_events in events_by_date.items():
        # Count shifts by type for this day
        shifts_by_type = {}
        for event in day_events:
            shift_key = event.extendedProps.get("shift_key", "")
            if shift_key not in shifts_by_type:
                shifts_by_type[shift_key] = 0
            shifts_by_type[shift_key] += 1
        
        # Check against expected capacity
        for shift_type in shift_types:
            shift_key = shift_type["key"]
            
            # Get expected capacity
            if shift_key == "APP":
                expected = get_app_shift_capacity(event_date)
            else:
                expected = shift_capacity.get(shift_key, 0)
            
            # Apply holiday adjustments
            expected = get_holiday_adjusted_capacity(expected, event_date)
            
            # Get actual count
            actual = shifts_by_type.get(shift_key, 0)
            
            # If there's a gap, record it
            if actual < expected:
                gaps.append({
                    "date": event_date,
                    "shift_type": shift_key,
                    "expected": expected,
                    "actual": actual,
                    "shortage": expected - actual
                })
    
    return gaps

# =============================================================================
# Core Scheduling Functions
# =============================================================================

def validate_rules(events: list[SEvent], rules: RuleConfig) -> dict[str, list[str]]:
    """Validate scheduling rules and return violations."""
    violations: dict[str, list[str]] = {}

    cap_map: dict[str, int] = st.session_state.get("shift_capacity", DEFAULT_SHIFT_CAPACITY)
    prov_caps: dict[str, list[str]] = st.session_state.get("provider_caps", {})
    prov_rules: dict[str, dict] = st.session_state.get("provider_rules", {})

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
                if "min_shifts" in pr:
                    eff_min = pr["min_shifts"]
                else:
                    eff_min = rules.min_shifts_per_provider
                
                if eff_min is not None and len(month_evs) < eff_min:
                    violations.setdefault(p_upper, []).append(
                        f"Month {year}-{month:02d}: {len(month_evs)} shifts below minimum {eff_min}"
                    )

    # Validate rest periods
    for p_upper in provider_month_events.keys():
        p_events = [e for e in events if (e.extendedProps.get("provider") or "").strip().upper() == p_upper]
        if not p_events:
            continue
            
        # Sort events by start time
        p_events.sort(key=lambda e: e.start)
        
        for i in range(len(p_events) - 1):
            ev1 = p_events[i]
            ev2 = p_events[i + 1]
            
            # Calculate rest period
            rest_days = (ev2.start.date() - ev1.start.date()).days
            
            # Get provider-specific rest requirement
            pr = prov_rules.get(p_upper, {}) or {}
            min_rest = pr.get("min_rest_days", rules.min_rest_days_between_shifts)
            
            if rest_days < min_rest:
                violations.setdefault(p_upper, []).append(
                    f"Insufficient rest: {rest_days} days between {ev1.start.date()} and {ev2.start.date()}"
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

def assign_greedy(providers: List[str], days: List[date], shift_types: List[Dict[str, Any]], rules: RuleConfig) -> List[SEvent]:
    """Greedy algorithm for assigning providers to shifts."""
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

    # ---------- helpers that read what we've already assigned in `events` ----------
    def day_shift_count(d: date, skey: str) -> int:
        return sum(1 for e in events if e.extendedProps.get("shift_key") == skey and e.start.date() == d)

    def provider_has_shift_on_day(p: str, d: date) -> bool:
        return any((e.extendedProps.get("provider") or "").upper() == p.upper() and e.start.date() == d for e in events)

    def provider_days(p: str) -> Set[date]:
        pu = (p or "").upper()
        return {e.start.date() for e in events if (e.extendedProps.get("provider") or "").upper() == pu}

    def left_run_len(days_set: Set[date], d: date) -> int:
        run = 0
        current = d
        while current in days_set:
            run += 1
            current += timedelta(days=1)
        return run

    def right_run_len(days_set: Set[date], d: date) -> int:
        run = 0
        current = d
        while current in days_set:
            run += 1
            current -= timedelta(days=1)
        return run

    def ok(prov: str, day: date, shift_key: str) -> bool:
        """Check if provider can be assigned to this shift."""
        pu = (prov or "").upper()
        
        # Check if provider is unavailable on this date
        if is_provider_unavailable_on_date(pu, day):
            return False
        
        # Check if provider already has a shift on this day
        if provider_has_shift_on_day(pu, day):
            return False
        
        # Check provider-specific shift eligibility
        if pu in prov_caps:
            allowed_shifts = prov_caps[pu]
            if shift_key not in allowed_shifts:
                return False
        
        # APP providers can only do APP shifts
        if pu in [ap.upper() for ap in APP_PROVIDER_INITIALS]:
            if shift_key != "APP":
                return False
        
        # Regular providers cannot do APP shifts
        if shift_key == "APP" and pu not in [ap.upper() for ap in APP_PROVIDER_INITIALS]:
            return False
        
        # Check rest period
        provider_days_set = provider_days(pu)
        if provider_days_set:
            # Find closest assigned day
            min_days_diff = min(abs((day - d).days) for d in provider_days_set)
            if min_days_diff < min_rest_days_global:
                return False
        
        return True

    def score(prov: str, day: date, shift_key: str) -> float:
        """Score a provider for this shift assignment."""
        pu = (prov or "").upper()
        
        # Base score starts with current shift count (lower is better)
        base_score = counts.get(pu, 0)
        
        # Prefer providers with fewer shifts
        score = -base_score * 10
        
        # Prefer weekend coverage for providers who need it
        if day.weekday() >= 5:  # Weekend
            pr = prov_rules.get(pu, {}) or {}
            if pr.get("require_weekend", rules.require_at_least_one_weekend):
                # Check if provider already has weekend shifts
                provider_days_set = provider_days(pu)
                weekend_days = sum(1 for d in provider_days_set if d.weekday() >= 5)
                if weekend_days == 0:
                    score += 50  # Big bonus for first weekend shift
        
        # Prefer shift consistency (same shift type in blocks)
        if shift_key in ["N12", "NB"]:  # Night shifts
            # Check if provider is already doing nights
            provider_days_set = provider_days(pu)
            if provider_days_set:
                # Check recent assignments
                recent_days = sorted(provider_days_set)[-3:]  # Last 3 days
                for recent_day in recent_days:
                    for event in events:
                        if (event.extendedProps.get("provider") or "").upper() == pu and event.start.date() == recent_day:
                            if event.extendedProps.get("shift_key") in ["N12", "NB"]:
                                score += 20  # Bonus for night shift consistency
                            break
        
        return score

    # Main assignment loop
    total_assignments = 0
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
                    title=f"{best} - {sdef['label']}",
                    start=start_dt,
                    end=end_dt,
                    backgroundColor=sdef["color"],
                    extendedProps={
                        "provider": best,
                        "shift_key": shift_key,
                        "label": sdef["label"]
                    }
                )
                
                events.append(ev)
                counts[best] = counts.get(best, 0) + 1
                
                # Track night shifts
                if shift_key in ["N12", "NB"]:
                    nights[best] = nights.get(best, 0) + 1

    return events

# =============================================================================
# UI Components
# =============================================================================

def render_calendar():
    """Render the interactive calendar."""
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
        st.caption("�� Navigate to change which month the Generate button will create schedules for")
    
    if not st_calendar:
        st.error("Calendar component not available. Please install streamlit-calendar.")
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
    changed = False
    
    if state.get("eventClick"):
        # Handle event click for editing
        clicked_event = state["eventClick"]["event"]
        event_id = clicked_event["id"]
        
        # Find the event in our session state
        for i, event in enumerate(st.session_state.events):
            if event["id"] == event_id:
                # Show edit form
                st.subheader("Edit Event")
                
                # Provider selection
                current_provider = event.get("extendedProps", {}).get("provider", "")
                all_providers = sorted(st.session_state.providers_df["initials"].tolist())
                new_provider = st.selectbox("Provider", ["(Select Provider)"] + all_providers, 
                                          index=all_providers.index(current_provider) + 1 if current_provider in all_providers else 0)
                
                if new_provider != "(Select Provider)":
                    st.session_state.events[i]["extendedProps"]["provider"] = new_provider
                    st.session_state.events[i]["title"] = f"{new_provider} - {event.get('extendedProps', {}).get('label', '')}"
                    changed = True
                
                # Comments
                if event_id not in st.session_state.comments:
                    st.session_state.comments[event_id] = []
                
                new_comment = st.text_input("Add comment")
                if st.button("Add Comment") and new_comment.strip():
                    st.session_state.comments[event_id].append({
                        "text": new_comment.strip(),
                        "timestamp": datetime.now().isoformat()
                    })
                    changed = True
                
                # Show existing comments
                if st.session_state.comments[event_id]:
                    st.subheader("Comments")
                    for comment in st.session_state.comments[event_id]:
                        st.write(f"💬 {comment['text']}")
                
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

def provider_rules_panel():
    """Provider-specific rules panel."""
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
        st.info("Select a provider in the Calendar tab to edit rules.")
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
    shift_count = sum(1 for e in all_events
                      if (e.get("extendedProps") or {}).get("provider") == sel)
    weekend_count = sum(1 for e in all_events
                        if (e.get("extendedProps") or {}).get("provider") == sel and 
                        pd.to_datetime(e.get("start")).weekday() >= 5)

    st.info(f"📊 **{sel}**: {shift_count} total shifts, {weekend_count} weekend shifts")

    # Max shifts
    max_sh = st.number_input("Max shifts per month", min_value=1, max_value=31, 
                            value=curr.get("max_shifts", base_default), key=f"pr_max_{sel}")
    
    # Min shifts
    min_sh = st.number_input("Min shifts per month", min_value=0, max_value=31, 
                            value=curr.get("min_shifts", 0), key=f"pr_min_{sel}")
    
    # Max nights
    max_n = st.number_input("Max night shifts per month", min_value=0, max_value=31, 
                           value=curr.get("max_nights", 6), key=f"pr_max_nights_{sel}")
    
    # Weekend requirement
    wk_choice = st.selectbox("Weekend requirement", 
                            ["No requirement", "Require at least one"], 
                            index=1 if curr.get("require_weekend", True) else 0,
                            key=f"pr_weekend_{sel}")
    
    # Min rest days
    min_rest_days = st.number_input("Min rest days between shifts", min_value=0.0, max_value=14.0, 
                                   value=curr.get("min_rest_days", 1.0), step=0.5, key=f"pr_rest_{sel}")
    
    # Day/night ratio preference
    ratio_val = st.slider("Day/Night ratio preference", min_value=0, max_value=100, 
                         value=curr.get("day_night_ratio", 50), key=f"pr_ratio_{sel}")
    
    # Half month preference
    half_month_val = st.slider("Half month preference", min_value=0, max_value=100, 
                              value=curr.get("half_month_preference", 50), key=f"pr_half_{sel}")
    
    # Shift consistency strength
    consistency_strength = st.slider("Shift consistency strength", min_value=0, max_value=100, 
                                   value=curr.get("shift_consistency_strength", 50), key=f"pr_consistency_{sel}")

    # Unavailable dates
    dates_txt = st.text_area("Unavailable dates (comma-separated, YYYY-MM-DD format)", 
                            value=", ".join(curr.get("unavailable_dates", [])), key=f"pr_dates_{sel}")

    # Vacations
    st.subheader("Vacations")
    vac_list = curr.get("vacations", [])
    
    if st.button("Add vacation", key=f"pr_add_vac_{sel}"):
        vac_list.append({"start": "", "end": ""})
    
    for i, vac in enumerate(vac_list):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            vac["start"] = st.date_input(f"Start {i+1}", 
                                        value=pd.to_datetime(vac.get("start")).date() if vac.get("start") else date.today(),
                                        key=f"pr_vac_start_{sel}_{i}")
        with col2:
            vac["end"] = st.date_input(f"End {i+1}", 
                                      value=pd.to_datetime(vac.get("end")).date() if vac.get("end") else date.today(),
                                      key=f"pr_vac_end_{sel}_{i}")
        with col3:
            if st.button("Remove", key=f"pr_remove_vac_{sel}_{i}"):
                vac_list.pop(i)
                st.rerun()

    # Notes
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

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state with defaults."""
    st.set_page_config(page_title="IMIS Scheduler", layout="wide", initial_sidebar_state="collapsed")
    
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

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
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
                if app_provs:
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
            st.caption(f"�� Currently viewing: {st.session_state.month.strftime('%B %Y')}")
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
                    providers = st.session_state.pr
