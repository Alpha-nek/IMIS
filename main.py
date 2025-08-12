# =============================================================================
# IMIS Scheduler - Main Application
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json
import os

# Import our modular components
from models.constants import (
    DEFAULT_SHIFT_TYPES, PROVIDER_INITIALS_DEFAULT, APP_PROVIDER_INITIALS,
    DEFAULT_SHIFT_CAPACITY, HOLIDAY_RULES
)
from models.data_models import RuleConfig, Provider, SEvent
from core.utils import (
    is_holiday, get_holiday_adjusted_capacity, parse_time, 
    date_range, month_start_end, make_month_days,
    _expand_vacation_dates, is_provider_unavailable_on_date,
    get_shift_label_maps, provider_weekend_count, get_global_rules
)
from core.scheduler import generate_schedule, validate_rules
from ui.calendar import render_calendar, render_month_navigation
from ui.grid import render_schedule_grid, apply_grid_changes_to_calendar
from ui.providers import providers_panel, load_providers_from_csv
from ui.requests import provider_requests_panel
from integrations.google_calendar import (
    get_gcal_service, gcal_list_calendars, sync_events_to_gcal,
    remove_events_from_gcal, provider_google_calendar_sync
)

# Page configuration
st.set_page_config(
    page_title="IMIS Scheduler",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "events" not in st.session_state:
        st.session_state.events = []
    
    if "current_month" not in st.session_state:
        st.session_state.current_month = date.today()
    
    if "shift_types" not in st.session_state:
        st.session_state.shift_types = DEFAULT_SHIFT_TYPES.copy()
    
    if "shift_capacity" not in st.session_state:
        st.session_state.shift_capacity = DEFAULT_SHIFT_CAPACITY.copy()
    
    if "providers" not in st.session_state:
        st.session_state.providers = PROVIDER_INITIALS_DEFAULT.copy()
    
    if "provider_rules" not in st.session_state:
        st.session_state.provider_rules = {}
    
    if "rules" not in st.session_state:
        st.session_state.rules = RuleConfig().model_dump()
    
    if "generation_count" not in st.session_state:
        st.session_state.generation_count = 0
    
    # Load providers from CSV if not already loaded
    if "providers_df" not in st.session_state or st.session_state.providers_df.empty:
        if load_providers_from_csv():
            st.session_state["providers_loaded"] = True
        else:
            # Fallback to default providers
            default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
            st.session_state["providers_df"] = default_providers
            st.session_state["providers_loaded"] = True

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("🏥 IMIS Scheduler")
    
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
                
                default = st.session_state.get("highlight_provider", "(All providers)")
                idx = provider_options.index(default) if default in provider_options else 0
                sel = st.selectbox("Highlight provider", options=provider_options, index=idx)
                st.session_state.highlight_provider = "" if sel == "(All providers)" else sel
            else:
                st.warning("No providers loaded. Please check the Providers tab.")
                st.session_state.highlight_provider = ""
        
        with col2:
            st.caption(f"�� Currently viewing: {st.session_state.current_month.strftime('%B %Y')}")
        
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
                    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                    global_rules = get_global_rules()
                    events = generate_schedule(
                        year=st.session_state.current_month.year,
                        month=st.session_state.current_month.month,
                        providers=providers,
                        shift_types=st.session_state.shift_types,
                        shift_capacity=st.session_state.shift_capacity,
                        provider_rules=st.session_state.provider_rules,
                        global_rules=global_rules
                    )
                    st.session_state.events = [event.to_json_event() for event in events]
                    st.session_state.generation_count += 1
                    st.success(f"✅ Draft schedule generated for {st.session_state.current_month.strftime('%B %Y')} with {len(events)} events! (Generation #{st.session_state.generation_count})")
        
        with g2:
            if st.button("✅ Validate Schedule", help="Check for rule violations"):
                if st.session_state.events:
                    global_rules = get_global_rules()
                    events = [SEvent(**{**e, "start": datetime.fromisoformat(e["start"]), "end": datetime.fromisoformat(e["end"])}) for e in st.session_state.events]
                    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                    
                    violations = validate_rules(events, providers, global_rules, st.session_state.provider_rules)
                    st.session_state.validation_results = violations
                    
                    if violations:
                        st.error("❌ Validation found issues!")
                    else:
                        st.success("✅ Schedule is valid!")
                else:
                    st.warning("No schedule to validate. Generate a schedule first.")
        
        with g3:
            if st.button("🗑️ Clear All", help="Clear all events"):
                st.session_state.events = []
                st.success("All events cleared!")
        
        # Calendar display
        if st.session_state.events:
            render_calendar(st.session_state.events)
        else:
            st.info("No schedule generated yet. Click 'Generate Draft' to create one.")
    
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
            rc.min_shifts_per_provider = st.number_input("Minimum shifts per provider", min_value=1, max_value=31, value=rc.min_shifts_per_provider)
            rc.max_shifts_per_provider = st.number_input("Maximum shifts per provider", min_value=1, max_value=31, value=rc.max_shifts_per_provider)
            rc.min_rest_days_between_shifts = st.number_input("Minimum rest days between shifts", min_value=0.0, max_value=14.0, value=rc.min_rest_days_between_shifts, step=0.5)
        
        with col2:
            rc.min_block_size = st.number_input("Minimum block size", min_value=1, max_value=7, value=rc.min_block_size)
            rc.max_block_size = st.number_input("Maximum block size", min_value=1, max_value=7, value=rc.max_block_size)
            rc.require_at_least_one_weekend = st.checkbox("Require at least one weekend", value=rc.require_at_least_one_weekend)
            rc.max_nights_per_provider = st.number_input("Maximum night shifts per provider", min_value=0, max_value=31, value=rc.max_nights_per_provider or 6)
        
        if st.button("Save Global Rules"):
            st.session_state.rules = rc.model_dump()
            st.success("Global rules saved!")
        
        # Shift types configuration
        st.subheader("🕐 Shift Types")
        for i, shift_type in enumerate(st.session_state.shift_types):
            with st.expander(f"Edit {shift_type['label']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.session_state.shift_types[i]["start"] = st.text_input(
                        "Start Time", shift_type["start"], key=f"start_{i}"
                    )
                with col2:
                    st.session_state.shift_types[i]["end"] = st.text_input(
                        "End Time", shift_type["end"], key=f"end_{i}"
                    )
                with col3:
                    st.session_state.shift_types[i]["color"] = st.color_picker(
                        "Color", shift_type["color"], key=f"color_{i}"
                    )
        
        # Shift capacity configuration
        st.subheader("📊 Shift Capacity")
        for shift_key, capacity in st.session_state.shift_capacity.items():
            st.session_state.shift_capacity[shift_key] = st.number_input(
                f"Capacity for {shift_key}", 
                min_value=1, 
                value=capacity, 
                key=f"capacity_{shift_key}"
            )
    
    with tab3:
        # Providers tab
        providers_panel()
    
    with tab4:
        # Grid view tab
        st.header("�� Schedule Grid View")
        st.caption("Edit assignments directly in the grid below")
        if st.session_state.events:
            grid_df = render_schedule_grid(st.session_state.events, 
                                         st.session_state.current_month.year, 
                                         st.session_state.current_month.month)
            
            # Grid editing controls
            if st.button("Apply Grid Changes to Calendar"):
                updated_events = apply_grid_changes_to_calendar(grid_df, st.session_state.events)
                st.session_state.events = updated_events
                st.success("Grid changes applied to calendar!")
                st.rerun()
        else:
            st.info("No schedule to display in grid view.")
    
    with tab5:
        # Google Calendar Sync tab
        provider_google_calendar_sync()
    
    with tab6:
        # Provider Requests tab
        provider_requests_panel()

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
        index=
