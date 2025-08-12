# =============================================================================
# IMIS Scheduler - Main Application
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json

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

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("🏥 IMIS Scheduler")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Month selection
        year, month = render_month_navigation()
        
        # Generate button
        if st.button("🎲 Generate Schedule", type="primary"):
            with st.spinner("Generating schedule..."):
                global_rules = get_global_rules()
                events = generate_schedule(
                    year=year,
                    month=month,
                    providers=st.session_state.providers,
                    shift_types=st.session_state.shift_types,
                    shift_capacity=st.session_state.shift_capacity,
                    provider_rules=st.session_state.provider_rules,
                    global_rules=global_rules
                )
                st.session_state.events = events
                st.session_state.generation_count += 1
                st.success("Schedule generated successfully!")
                st.rerun()
        
        # Validation button
        if st.button("✅ Validate Schedule"):
            if st.session_state.events:
                global_rules = get_global_rules()
                violations = validate_rules(
                    st.session_state.events,
                    st.session_state.providers,
                    global_rules,
                    st.session_state.provider_rules
                )
                st.session_state.validation_results = violations
                st.info("Schedule validated!")
            else:
                st.warning("No schedule to validate. Generate a schedule first.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📅 Calendar View", "📊 Grid View", "⚙️ Settings"])
    
    with tab1:
        st.header("Calendar View")
        if st.session_state.events:
            render_calendar(st.session_state.events)
        else:
            st.info("No schedule generated yet. Click 'Generate Schedule' to create one.")
    
    with tab2:
        st.header("Grid View")
        if st.session_state.events:
            grid_df = render_schedule_grid(st.session_state.events, year, month)
            
            # Grid editing controls
            if st.button("Apply Grid Changes to Calendar"):
                updated_events = apply_grid_changes_to_calendar(grid_df, st.session_state.events)
                st.session_state.events = updated_events
                st.success("Grid changes applied to calendar!")
                st.rerun()
        else:
            st.info("No schedule to display in grid view.")
    
    with tab3:
        st.header("Settings")
        
        # Shift types configuration
        st.subheader("Shift Types")
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
        st.subheader("Shift Capacity")
        for shift_key, capacity in st.session_state.shift_capacity.items():
            st.session_state.shift_capacity[shift_key] = st.number_input(
                f"Capacity for {shift_key}", 
                min_value=1, 
                value=capacity, 
                key=f"capacity_{shift_key}"
            )
        
        # Provider management
        st.subheader("Providers")
        new_provider = st.text_input("Add Provider (initials)")
        if st.button("Add Provider"):
            if new_provider.strip():
                st.session_state.providers.append(new_provider.strip().upper())
                st.success(f"Added provider: {new_provider.strip().upper()}")
                st.rerun()
        
        # Display current providers
        if st.session_state.providers:
            st.write("Current providers:")
            for provider in st.session_state.providers:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(provider)
                with col2:
                    if st.button("Remove", key=f"remove_{provider}"):
                        st.session_state.providers.remove(provider)
                        st.success(f"Removed provider: {provider}")
                        st.rerun()

if __name__ == "__main__":
    main()
