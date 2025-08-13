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
    DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS,
    PROVIDER_INITIALS_DEFAULT, HOLIDAY_RULES
)
from models.data_models import RuleConfig, Provider, SEvent
from core.utils import (
    is_holiday, get_holiday_adjusted_capacity, parse_time, 
    date_range, month_start_end, make_month_days,
    _expand_vacation_dates, is_provider_unavailable_on_date,
    get_global_rules
)
from core.scheduler import generate_schedule, validate_rules
from ui.calendar import render_calendar, render_month_navigation
from ui.grid import render_schedule_grid, apply_grid_changes_to_calendar
from ui.providers import providers_panel, load_providers_from_csv
from ui.requests import provider_requests_panel

# Page configuration
st.set_page_config(
    page_title="IMIS Scheduler",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "events" not in st.session_state:
        st.session_state.events = []
    
    if "current_year" not in st.session_state:
        st.session_state.current_year = datetime.now().year
    
    if "current_month" not in st.session_state:
        st.session_state.current_month = datetime.now().month
    
    if "providers_df" not in st.session_state:
        st.session_state.providers_df = pd.DataFrame()
    
    if "providers_loaded" not in st.session_state:
        st.session_state.providers_loaded = False
    
    if "global_rules" not in st.session_state:
        st.session_state.global_rules = RuleConfig()
    
    # Ensure all required attributes exist
    if not hasattr(st.session_state.global_rules, 'max_consecutive_shifts'):
        st.session_state.global_rules = RuleConfig()
    
    if "shift_types" not in st.session_state:
        st.session_state.shift_types = DEFAULT_SHIFT_TYPES.copy()
    
    # Ensure shift_types is properly structured
    if not isinstance(st.session_state.shift_types, list):
        st.session_state.shift_types = DEFAULT_SHIFT_TYPES.copy()
    else:
        # Validate each shift type
        for i, shift_type in enumerate(st.session_state.shift_types):
            if not isinstance(shift_type, dict) or 'name' not in shift_type:
                st.session_state.shift_types[i] = DEFAULT_SHIFT_TYPES[i] if i < len(DEFAULT_SHIFT_TYPES) else {
                    'name': f'Shift {i+1}',
                    'start_time': '08:00',
                    'end_time': '16:00',
                    'color': '#1f77b4'
                }
    
    if "shift_capacity" not in st.session_state:
        st.session_state.shift_capacity = DEFAULT_SHIFT_CAPACITY.copy()
    
    if "provider_rules" not in st.session_state:
        st.session_state.provider_rules = {}
    
    if "mobile_view" not in st.session_state:
        st.session_state.mobile_view = "home"

def render_mobile_interface():
    """Render mobile-optimized interface."""
    try:
        from ui.mobile_components import (
            mobile_quick_actions, mobile_my_shifts_view, 
            mobile_request_form, mobile_notifications_view,
            mobile_navigation
        )
        from ui.responsive import mobile_calendar
    except ImportError:
        # Fallback if mobile components don't exist yet
        st.error("Mobile components not available. Using desktop interface.")
        render_desktop_interface()
        return
    
    # Mobile header
    st.markdown("""
    <div class="main-header">
        <h1>📱 IMIS Mobile</h1>
        <p>Hospitalist Scheduler</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mobile navigation
    current_view = mobile_navigation()
    
    # Mobile view routing
    if current_view == "home" or current_view == "calendar":
        st.markdown("### 📅 Schedule")
        if st.session_state.events:
            mobile_calendar(st.session_state.events, height=400)
        else:
            st.info("No schedule available. Generate a schedule first.")
        
        mobile_quick_actions()
    
    elif current_view == "requests":
        st.markdown("### 📝 Requests")
        
        # Request form
        mobile_request_form()
        
        # Show existing requests
        if "mobile_requests" in st.session_state and st.session_state.mobile_requests:
            st.markdown("#### My Requests")
            for request in st.session_state.mobile_requests:
                st.json(request)
    
    elif current_view == "settings":
        st.markdown("### ⚙️ Settings")
        st.info("Settings will be available in the next update.")

def render_desktop_interface():
    """Render desktop interface."""
    # Custom CSS for professional styling
    st.markdown("""
    <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .main-header {
                padding: 0.5rem;
                margin-bottom: 1rem;
            }
            .main-header h1 {
                font-size: 1.5rem;
            }
            .main-header p {
                font-size: 0.9rem;
            }
            .metric-card {
                padding: 0.75rem;
                margin: 0.25rem 0;
            }
            .stButton > button {
                padding: 8px 16px;
                font-size: 14px;
            }
        }
        
        .main-header {
            background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        .status-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .status-success {
            border-left-color: #28a745;
        }
        
        .status-error {
            border-left-color: #dc3545;
        }
        
        .stButton > button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Professional header and provider status indicator
    st.markdown("""
    <div class="main-header">
        <h1>🏥 IMIS Hospitalist Scheduler</h1>
        <p>Intelligent Medical Inpatient Scheduling System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Provider status indicator
    if st.session_state.get("providers_loaded", False) and not st.session_state.providers_df.empty:
        provider_count = len(st.session_state.providers_df)
        st.markdown(f"""
        <div class="status-card status-success">
            <h4>✅ Providers Loaded Successfully</h4>
            <p><strong>{provider_count}</strong> providers available for scheduling</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-error">
            <h4>⚠️ No Providers Loaded</h4>
            <p>Please load providers from the Providers tab to start scheduling</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Calendar", "⚙️ Settings", "👥 Providers", "📊 Grid View", "📅 Google Sync", "📝 Requests"
    ])
    
    # Calendar Tab
    with tab1:
        st.header("📅 Schedule Calendar")
        
        # Month navigation
        year, month = render_month_navigation()
        st.session_state.current_year = year
        st.session_state.current_month = month
        
        # Generate schedule button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Generate Schedule", type="primary", use_container_width=True):
                if st.session_state.providers_df.empty:
                    st.error("Please load providers first!")
                else:
                    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                    
                    # Generate schedule
                    events = generate_schedule(
                        year=year,
                        month=month,
                        providers=providers,
                        shift_types=st.session_state.shift_types,
                        shift_capacity=st.session_state.shift_capacity,
                        provider_rules=st.session_state.provider_rules,
                        global_rules=st.session_state.global_rules
                    )
                    
                    # Convert SEvent objects to dictionaries for JSON compatibility
                    st.session_state.events = [event.to_json_event() for event in events]
                    
                    # Validate rules
                    validation_results = validate_rules(
                        events=events,
                        providers=providers,
                        global_rules=st.session_state.global_rules,
                        provider_rules=st.session_state.provider_rules
                    )
                    
                    st.session_state.validation_results = validation_results
                    
                    if validation_results["is_valid"]:
                        st.success("✅ Schedule generated successfully!")
                    else:
                        st.warning("⚠️ Schedule generated with violations. Check validation results.")
                    
                    st.rerun()
        
        with col2:
            if st.button("✅ Validate Rules", type="secondary", use_container_width=True):
                if st.session_state.events:
                    # Convert back to SEvent objects for validation
                    events = []
                    for event_dict in st.session_state.events:
                        event = SEvent(
                            id=event_dict["id"],
                            title=event_dict["title"],
                            start=datetime.fromisoformat(event_dict["start"]),
                            end=datetime.fromisoformat(event_dict["end"]),
                            extendedProps=event_dict["extendedProps"]
                        )
                        events.append(event)
                    
                    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                    
                    validation_results = validate_rules(
                        events=events,
                        providers=providers,
                        global_rules=st.session_state.global_rules,
                        provider_rules=st.session_state.provider_rules
                    )
                    
                    st.session_state.validation_results = validation_results
                    
                    if validation_results["is_valid"]:
                        st.success("✅ All rules validated successfully!")
                    else:
                        st.warning("⚠️ Rule violations found. Check details below.")
                    
                    st.rerun()
                else:
                    st.error("No schedule to validate!")
        
        with col3:
            if st.button("🗑️ Clear Schedule", type="secondary", use_container_width=True):
                st.session_state.events = []
                st.session_state.validation_results = None
                st.success("Schedule cleared!")
                st.rerun()
        
        # Display validation results if available
        if hasattr(st.session_state, 'validation_results') and st.session_state.validation_results:
            results = st.session_state.validation_results
            
            if not results["is_valid"]:
                st.error("❌ Rule Violations Found")
                
                with st.expander("View Violations", expanded=True):
                    if results["violations"]:
                        for violation in results["violations"]:
                            st.error(f"• {violation}")
                    
                    if results["provider_violations"]:
                        st.subheader("Provider-Specific Violations")
                        for provider, violations in results["provider_violations"].items():
                            st.error(f"**{provider}**:")
                            for violation in violations:
                                st.error(f"  - {violation}")
            else:
                st.success("✅ All rules validated successfully!")
        
        # Render calendar
        if st.session_state.events:
            render_calendar(st.session_state.events)
        else:
            st.info("No schedule available. Generate a schedule to view it here.")
    
    # Settings Tab
    with tab2:
        st.header("⚙️ Global Settings")
        
        # Global Rules Configuration
        st.subheader("📋 Global Scheduling Rules")
        
        # Safety check for global_rules
        if not hasattr(st.session_state.global_rules, 'max_consecutive_shifts'):
            st.session_state.global_rules = RuleConfig()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.global_rules.max_consecutive_shifts = st.number_input(
                "Max Consecutive Shifts", 
                min_value=1, max_value=14, value=getattr(st.session_state.global_rules, 'max_consecutive_shifts', 7)
            )
            
            st.session_state.global_rules.min_days_between_shifts = st.number_input(
                "Min Days Between Shifts", 
                min_value=0, max_value=7, value=getattr(st.session_state.global_rules, 'min_days_between_shifts', 1)
            )
            
            st.session_state.global_rules.max_shifts_per_month = st.number_input(
                "Max Shifts Per Month", 
                min_value=1, max_value=31, value=getattr(st.session_state.global_rules, 'max_shifts_per_month', 16)
            )
            
            st.session_state.global_rules.min_shifts_per_month = st.number_input(
                "Min Shifts Per Month", 
                min_value=0, max_value=31, value=getattr(st.session_state.global_rules, 'min_shifts_per_month', 8)
            )
        
        with col2:
            st.session_state.global_rules.max_weekend_shifts_per_month = st.number_input(
                "Max Weekend Shifts Per Month", 
                min_value=0, max_value=10, value=getattr(st.session_state.global_rules, 'max_weekend_shifts_per_month', 4)
            )
            
            st.session_state.global_rules.min_weekend_shifts_per_month = st.number_input(
                "Min Weekend Shifts Per Month", 
                min_value=0, max_value=10, value=getattr(st.session_state.global_rules, 'min_weekend_shifts_per_month', 1)
            )
            
            st.session_state.global_rules.max_night_shifts_per_month = st.number_input(
                "Max Night Shifts Per Month", 
                min_value=0, max_value=31, value=getattr(st.session_state.global_rules, 'max_night_shifts_per_month', 8)
            )
            
            st.session_state.global_rules.min_night_shifts_per_month = st.number_input(
                "Min Night Shifts Per Month", 
                min_value=0, max_value=31, value=getattr(st.session_state.global_rules, 'min_night_shifts_per_month', 2)
            )
        
        # Shift Types Configuration
        st.subheader("🔄 Shift Types")
        
        for i, shift_type in enumerate(st.session_state.shift_types):
            # Safety check for shift_type structure
            if not isinstance(shift_type, dict) or 'name' not in shift_type:
                # Reset to default if corrupted
                st.session_state.shift_types[i] = DEFAULT_SHIFT_TYPES[i] if i < len(DEFAULT_SHIFT_TYPES) else {
                    'name': f'Shift {i+1}',
                    'start_time': '08:00',
                    'end_time': '16:00',
                    'color': '#1f77b4'
                }
                shift_type = st.session_state.shift_types[i]
            
            with st.expander(f"Shift Type: {shift_type.get('name', f'Shift {i+1}')}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.session_state.shift_types[i]['name'] = st.text_input(
                        "Name", value=shift_type.get('name', f'Shift {i+1}'), key=f"shift_name_{i}"
                    )
                
                with col2:
                    st.session_state.shift_types[i]['start_time'] = st.text_input(
                        "Start Time", value=shift_type.get('start_time', '08:00'), key=f"shift_start_{i}"
                    )
                
                with col3:
                    st.session_state.shift_types[i]['end_time'] = st.text_input(
                        "End Time", value=shift_type.get('end_time', '16:00'), key=f"shift_end_{i}"
                    )
                
                # Color picker for shift type
                st.session_state.shift_types[i]['color'] = st.color_picker(
                    "Color", value=shift_type.get('color', '#1f77b4'), key=f"shift_color_{i}"
                )
        
        # Shift Capacity Configuration
        st.subheader("📊 Shift Capacity")
        
        for shift_type in st.session_state.shift_types:
            # Safety check for shift_type structure
            if not isinstance(shift_type, dict) or 'name' not in shift_type:
                continue  # Skip corrupted shift types
            
            shift_name = shift_type.get('name', 'Unknown')
            capacity = st.number_input(
                f"Capacity for {shift_name}", 
                min_value=1, max_value=10, 
                value=st.session_state.shift_capacity.get(shift_name, 1),
                key=f"capacity_{shift_name}"
            )
            st.session_state.shift_capacity[shift_name] = capacity
    
    # Providers Tab
    with tab3:
        providers_panel()
    
    # Grid View Tab
    with tab4:
        st.header("📊 Schedule Grid View")
        
        if st.session_state.events:
            # Render grid view
            grid_df = render_schedule_grid(st.session_state.events, year, month)
            
            # Apply changes button
            if st.button("🔄 Apply Grid Changes to Calendar", type="primary"):
                updated_events = apply_grid_changes_to_calendar(grid_df, st.session_state.events)
                st.session_state.events = updated_events
                st.success("Grid changes applied to calendar!")
                st.rerun()
        else:
            st.info("No schedule available. Generate a schedule to view it in grid format.")
    
    # Google Calendar Sync Tab
    with tab5:
        st.header("📅 Google Calendar Sync")
        st.info("Google Calendar integration will be implemented in the next phase.")
        st.write("This tab will allow providers to sync their shifts to their Google Calendar.")
    
    # Provider Requests Tab
    with tab6:
        provider_requests_panel()

def main():
    """Main application function."""
    initialize_session_state()
    
    # Mobile detection
    try:
        from ui.responsive import is_mobile
        if is_mobile():
            render_mobile_interface()
        else:
            render_desktop_interface()
    except ImportError:
        # Fallback to desktop interface if mobile components don't exist
        render_desktop_interface()

if __name__ == "__main__":
    main()
