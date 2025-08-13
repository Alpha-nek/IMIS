# =============================================================================
# IMIS Scheduler - Main Application
# =============================================================================
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json
import os
import traceback
import sys

# Import our modular components with error handling
try:
    from models.constants import (
        DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS,
        PROVIDER_INITIALS_DEFAULT, HOLIDAY_RULES
    )
except ImportError as e:
    st.error(f"Failed to import constants: {e}")
    st.stop()

try:
    from models.data_models import RuleConfig, Provider, SEvent
except ImportError as e:
    st.error(f"Failed to import data models: {e}")
    st.stop()

try:
    from core.utils import (
        is_holiday, get_holiday_adjusted_capacity, parse_time, 
        date_range, month_start_end, make_month_days,
        _expand_vacation_dates, is_provider_unavailable_on_date,
        get_global_rules
    )
except ImportError as e:
    st.error(f"Failed to import core utils: {e}")
    st.stop()

try:
    from core.scheduler import generate_schedule, validate_rules
except ImportError as e:
    st.error(f"Failed to import scheduler: {e}")
    st.stop()

try:
    from ui.calendar import render_calendar, render_month_navigation
except ImportError as e:
    st.error(f"Failed to import calendar UI: {e}")
    st.stop()

try:
    from ui.grid import render_schedule_grid, apply_grid_changes_to_calendar
except ImportError as e:
    st.error(f"Failed to import grid UI: {e}")
    st.stop()

try:
    from ui.providers import providers_panel, load_providers_from_csv
except ImportError as e:
    st.error(f"Failed to import providers UI: {e}")
    st.stop()

try:
    from ui.requests import provider_requests_panel
except ImportError as e:
    st.error(f"Failed to import requests UI: {e}")
    st.stop()

try:
    from ui.data_status import render_data_status
except ImportError as e:
    st.error(f"Failed to import data status UI: {e}")
    st.stop()

try:
    from core.data_manager import (
        initialize_default_data, auto_load_session_state, auto_save_session_state,
        save_providers, save_rules, save_schedule
    )
except ImportError as e:
    st.error(f"Failed to import data manager: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="IMIS Scheduler",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render_header_with_logo():
    """Render the main header with logo and branding."""
    st.markdown("""
    <style>
        /* Brown color palette */
        :root {
            --primary-brown: #8B4513;
            --secondary-brown: #A0522D;
            --light-brown: #DEB887;
            --warm-brown: #D2691E;
            --cream: #F5F5DC;
            --dark-brown: #654321;
            --accent-brown: #CD853F;
        }
        
        /* Header styling with logo */
        .main-header {
            background: linear-gradient(135deg, var(--primary-brown) 0%, var(--secondary-brown) 50%, var(--warm-brown) 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }
        
        .header-content {
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
        }
        
        .logo-container {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 0.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .logo-container img {
            height: 60px;
            width: auto;
            border-radius: 8px;
        }
        
        .header-text {
            text-align: left;
        }
        
        .header-text h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            color: var(--cream);
        }
        
        .header-text p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .header-subtitle {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.25rem;
        }
        
        /* Mobile responsive header */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .header-text {
                text-align: center;
            }
            
            .header-text h1 {
                font-size: 2rem;
            }
            
            .logo-container img {
                height: 50px;
            }
        }
        
        /* Global brown theme styling */
        .stButton > button {
            background-color: var(--primary-brown) !important;
            border-color: var(--primary-brown) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: var(--secondary-brown) !important;
            border-color: var(--secondary-brown) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(139, 69, 19, 0.3) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Secondary button styling */
        .stButton > button[kind="secondary"] {
            background-color: transparent !important;
            border-color: var(--primary-brown) !important;
            color: var(--primary-brown) !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--primary-brown) !important;
            color: white !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--cream);
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 500;
            color: var(--dark-brown);
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-brown) !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(139, 69, 19, 0.3);
        }
        
        .stTabs [aria-selected="false"]:hover {
            background-color: var(--light-brown) !important;
            color: var(--dark-brown) !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, var(--cream) 0%, white 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid var(--primary-brown);
            box-shadow: 0 4px 12px rgba(139, 69, 19, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(139, 69, 19, 0.2);
        }
        
        /* Status indicators */
        .status-success {
            border-left-color: #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #f8f9fa 100%);
        }
        
        .status-error {
            border-left-color: #dc3545;
            background: linear-gradient(135deg, #f8d7da 0%, #f8f9fa 100%);
        }
        
        .status-warning {
            border-left-color: #ffc107;
            background: linear-gradient(135deg, #fff3cd 0%, #f8f9fa 100%);
        }
        
        /* Data editor styling */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(139, 69, 19, 0.1);
        }
        
        /* Calendar styling */
        .calendar-container {
            background: var(--cream);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(139, 69, 19, 0.1);
        }
        
        /* Form styling */
        .stTextInput > div > div > input {
            border-color: var(--light-brown) !important;
            border-radius: 8px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-brown) !important;
            box-shadow: 0 0 0 2px rgba(139, 69, 19, 0.2) !important;
        }
        
        .stSelectbox > div > div > div {
            border-color: var(--light-brown) !important;
            border-radius: 8px !important;
        }
        
        .stSelectbox > div > div > div:focus-within {
            border-color: var(--primary-brown) !important;
            box-shadow: 0 0 0 2px rgba(139, 69, 19, 0.2) !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--cream) !important;
            border-radius: 8px !important;
            color: var(--dark-brown) !important;
            font-weight: 500 !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: var(--light-brown) !important;
        }
        
        /* Success/Error messages */
        .stAlert {
            border-radius: 8px !important;
            border-left: 4px solid !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            border-left-color: var(--primary-brown) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <div class="logo-container">
                <img src="data:image/png;base64,{}" alt="IMIS Logo">
            </div>
            <div class="header-text">
                <h1>🏥 IMIS Scheduler</h1>
                <p>Hospitalist Management & Scheduling System</p>
                <div class="header-subtitle">Professional • Efficient • Reliable</div>
            </div>
        </div>
    </div>
    """.format(get_base64_logo()), unsafe_allow_html=True)

def get_base64_logo():
    """Convert the logo to base64 for embedding in HTML."""
    import base64
    
    try:
        with open("brown logo.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        # Fallback if logo not found
        return ""
    except Exception as e:
        st.warning(f"Could not load logo: {e}")
        return ""

def initialize_session_state():
    """Initialize Streamlit session state variables with automatic data loading."""
    try:
        # Initialize default data files if they don't exist
        initialize_default_data()
    except Exception as e:
        st.error(f"Failed to initialize default data: {e}")
        st.stop()
    
    # Basic session state initialization with error handling
    try:
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
        
        if "shift_types" not in st.session_state:
            st.session_state.shift_types = DEFAULT_SHIFT_TYPES.copy()
        
        if "shift_capacity" not in st.session_state:
            st.session_state.shift_capacity = DEFAULT_SHIFT_CAPACITY.copy()
        
        if "provider_rules" not in st.session_state:
            st.session_state.provider_rules = {}
        
        if "mobile_view" not in st.session_state:
            st.session_state.mobile_view = "home"
        
        if "validation_results" not in st.session_state:
            st.session_state.validation_results = None
            
    except Exception as e:
        st.error(f"Failed to initialize session state: {e}")
        st.stop()
    
    # Auto-load data from saved files
    try:
        auto_load_session_state()
    except Exception as e:
        st.warning(f"Failed to auto-load session state: {e}")
    
    # Ensure all required attributes exist after loading
    try:
        if not hasattr(st.session_state.global_rules, 'max_consecutive_shifts'):
            st.session_state.global_rules = RuleConfig()
        
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
    except Exception as e:
        st.error(f"Failed to validate session state structure: {e}")
        st.stop()

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
    # Render the new header with logo
    render_header_with_logo()
    
    # Get current date info
    year = st.session_state.current_year
    month = st.session_state.current_month
    
    # Main content area with brown theme
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F5F5DC 0%, #FFFFFF 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
    """, unsafe_allow_html=True)
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
            total_providers = len(st.session_state.providers_df)
            st.metric("👥 Total Providers", total_providers)
        else:
            st.metric("👥 Total Providers", 0)
    
    with col2:
        if "events" in st.session_state:
            total_events = len(st.session_state.events)
            st.metric("📅 Scheduled Events", total_events)
        else:
            st.metric("📅 Scheduled Events", 0)
    
    with col3:
        if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
            physician_count = len(st.session_state.providers_df[st.session_state.providers_df["type"] == "Physician"])
            st.metric("👨‍⚕️ Physicians", physician_count)
        else:
            st.metric("👨‍⚕️ Physicians", 0)
    
    with col4:
        if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
            app_count = len(st.session_state.providers_df[st.session_state.providers_df["type"] == "APP"])
            st.metric("👩‍⚕️ APPs", app_count)
        else:
            st.metric("👩‍⚕️ APPs", 0)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main tabs with brown theme
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Calendar", "⚙️ Settings", "👥 Providers", "📊 Grid View", 
        "🔄 Sync", "📝 Requests", "💾 Data"
    ])
    
    # Calendar Tab
    with tab1:
        st.markdown("### 📅 Schedule Calendar")
        
        # Month navigation
        render_month_navigation(year, month)
        
        # Calendar display
        if st.session_state.events:
            st.markdown('<div class="calendar-container">', unsafe_allow_html=True)
            render_calendar(st.session_state.events, year, month)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No schedule available. Generate a schedule first.")
        
        # Schedule generation
        st.markdown("### 🚀 Generate Schedule")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Generate New Schedule", type="primary", use_container_width=True):
                try:
                    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
                        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                        
                        # Generate schedule
                        events = generate_schedule(
                            year, month, providers, 
                            st.session_state.shift_types, 
                            st.session_state.shift_capacity,
                            st.session_state.provider_rules, 
                            st.session_state.global_rules
                        )
                        
                        st.session_state.events = events
                        
                        # Validate rules
                        validation_results = validate_rules(
                            events, providers, 
                            st.session_state.global_rules, 
                            st.session_state.provider_rules
                        )
                        st.session_state.validation_results = validation_results
                        
                        # Auto-save
                        auto_save_session_state()
                        
                        st.success("✅ Schedule generated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No providers loaded. Please load providers first.")
                except Exception as e:
                    st.error(f"❌ Failed to generate schedule: {e}")
                    st.error(f"Error details: {traceback.format_exc()}")
        
        with col2:
            if st.button("🔄 Regenerate Schedule", type="secondary", use_container_width=True):
                try:
                    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
                        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                        
                        # Clear existing events
                        st.session_state.events = []
                        
                        # Generate new schedule
                        events = generate_schedule(
                            year, month, providers, 
                            st.session_state.shift_types, 
                            st.session_state.shift_capacity,
                            st.session_state.provider_rules, 
                            st.session_state.global_rules
                        )
                        
                        st.session_state.events = events
                        
                        # Validate rules
                        validation_results = validate_rules(
                            events, providers, 
                            st.session_state.global_rules, 
                            st.session_state.provider_rules
                        )
                        st.session_state.validation_results = validation_results
                        
                        # Auto-save
                        auto_save_session_state()
                        
                        st.success("✅ Schedule regenerated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No providers loaded. Please load providers first.")
                except Exception as e:
                    st.error(f"❌ Failed to regenerate schedule: {e}")
                    st.error(f"Error details: {traceback.format_exc()}")
        
        # Validation results display
        if hasattr(st.session_state, 'validation_results') and st.session_state.validation_results:
            validation = st.session_state.validation_results
            
            st.markdown("### 📊 Schedule Validation")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if validation["is_valid"]:
                    st.metric("✅ Status", "Valid", delta_color="normal")
                else:
                    st.metric("❌ Status", "Issues Found", delta_color="inverse")
            
            with col2:
                violation_count = len(validation["violations"])
                st.metric("⚠️ Violations", violation_count)
            
            with col3:
                if "events" in st.session_state:
                    total_events = len(st.session_state.events)
                    st.metric("📅 Total Events", total_events)
                else:
                    st.metric("📅 Total Events", 0)
            
            with col4:
                if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
                    providers_used = len(set(
                        event.extendedProps.get("provider", "") 
                        for event in st.session_state.events 
                        if hasattr(event, 'extendedProps')
                    ))
                    st.metric("👥 Providers Used", providers_used)
                else:
                    st.metric("👥 Providers Used", 0)
            
            # Detailed violations
            if not validation["is_valid"]:
                with st.expander("🔍 View Detailed Violations", expanded=False):
                    for violation in validation["violations"]:
                        st.markdown(f"• {violation}")
                    
                    # Provider-specific violations
                    if validation["provider_violations"]:
                        st.markdown("#### Provider-Specific Issues:")
                        for provider, violations in validation["provider_violations"].items():
                            if violations:
                                st.markdown(f"**{provider}:**")
                                for violation in violations:
                                    st.markdown(f"  - {violation}")
            
            # Quick action buttons
            if not validation["is_valid"]:
                st.markdown("### 🔧 Quick Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Regenerate to Fix Issues", type="primary"):
                        st.rerun()
                
                with col2:
                    if st.button("📊 View Grid for Manual Fixes", type="secondary"):
                        st.rerun()
        
        # Provider statistics section
        if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
            st.markdown("### 📈 Provider Statistics")
            
            providers_df = st.session_state.providers_df
            
            # Count by type
            physician_count = len(providers_df[providers_df["type"] == "Physician"])
            app_count = len(providers_df[providers_df["type"] == "APP"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Providers", len(providers_df))
            
            with col2:
                st.metric("Physicians", physician_count)
            
            with col3:
                st.metric("APPs", app_count)
    
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
        
        # Auto-save settings when changed
        if st.button("💾 Save Settings", type="primary"):
            save_rules(
                st.session_state.global_rules,
                st.session_state.shift_types,
                st.session_state.shift_capacity
            )
            st.success("Settings saved successfully!")
            st.rerun()
    
    # Providers Tab
    with tab3:
        try:
            providers_panel()
        except Exception as e:
            st.error(f"Failed to render providers panel: {e}")
            st.error(f"Error details: {traceback.format_exc()}")
    
    # Grid View Tab
    with tab4:
        if st.session_state.events:
            try:
                # Render grid view (now includes editing functionality)
                render_schedule_grid(st.session_state.events, year, month)
            except Exception as e:
                st.error(f"Failed to render grid view: {e}")
                st.error(f"Error details: {traceback.format_exc()}")
        else:
            st.info("No schedule available. Generate a schedule to view it in grid format.")
    
    # Google Calendar Sync Tab
    with tab5:
        st.header("📅 Google Calendar Sync")
        st.info("Google Calendar integration will be implemented in the next phase.")
        st.write("This tab will allow providers to sync their shifts to their Google Calendar.")
    
    # Provider Requests Tab
    with tab6:
        try:
            provider_requests_panel()
        except Exception as e:
            st.error(f"Failed to render requests panel: {e}")
            st.error(f"Error details: {traceback.format_exc()}")
    
    # Data Management Tab
    with tab7:
        try:
            render_data_status()
        except Exception as e:
            st.error(f"Failed to render data status: {e}")
            st.error(f"Error details: {traceback.format_exc()}")

def main():
    """Main application function."""
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.error(f"Error details: {traceback.format_exc()}")
        st.stop()
    
    # Mobile detection
    try:
        from ui.responsive import is_mobile
        if is_mobile():
            render_mobile_interface()
        else:
            render_desktop_interface()
    except ImportError:
        # Fallback to desktop interface if mobile components don't exist
        try:
            render_desktop_interface()
        except Exception as e:
            st.error(f"Failed to render desktop interface: {e}")
            st.error(f"Error details: {traceback.format_exc()}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to detect mobile/desktop: {e}")
        st.error(f"Error details: {traceback.format_exc()}")
        st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        st.error(f"Error details: {traceback.format_exc()}")
        st.stop()
