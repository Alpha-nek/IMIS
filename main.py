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

# Define nocturnists locally to avoid import issues
NOCTURNISTS = {"JT", "OI", "AT", "CM", "YD", "RS"}

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
        save_providers, save_rules, save_schedule, load_providers, load_rules
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
        /* Four-color palette */
        :root {
            --primary-nile: #18535B;      /* Dark Blue */
            --secondary-nile: #0F4C75;    /* Darker Blue */
            --light-nile: #80A4ED;        /* Light Blue */
            --accent-nile: #1E3A8A;       /* Accent Blue */
            --primary-cream: #E6E1C5;     /* Cream */
            --light-cream: #F5F5F0;       /* Light Cream */
            --white: #FFFFFF;
            --primary-coral: #FF674D;     /* Coral */
            --secondary-coral: #E55A3C;   /* Darker Coral */
            --light-coral: #FF8A7A;       /* Light Coral */
            --gray-light: #F8F9FA;
            --gray-dark: #475569;
        }
        
        /* Header styling with logo */
        .main-header {
            background: linear-gradient(135deg, var(--primary-nile) 0%, var(--primary-coral) 50%, var(--light-nile) 100%);
            padding: 3rem 2rem;
            border-radius: 25px;
            margin-bottom: 2.5rem;
            color: white;
            text-align: center;
            box-shadow: 0 12px 35px rgba(24, 83, 91, 0.4);
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
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="60" r="0.8" fill="rgba(255,255,255,0.08)"/><circle cx="90" cy="30" r="0.6" fill="rgba(255,255,255,0.08)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.4;
        }
        
        .main-header::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 30% 20%, rgba(255, 103, 77, 0.2) 0%, transparent 50%),
                        radial-gradient(circle at 70% 80%, rgba(128, 164, 237, 0.2) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .header-content {
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .logo-container {
            background: rgba(255, 255, 255, 0.25);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
            border: 3px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
        }
        
        .logo-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            border-radius: 20px;
        }
        
        .logo-container img {
            height: 140px;
            width: auto;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 2;
            transition: transform 0.3s ease;
        }
        
        .logo-container:hover img {
            transform: scale(1.05);
        }
        
        .header-text {
            text-align: left;
            flex: 1;
            max-width: 600px;
        }
        
        .header-text h1 {
            font-size: 3.5rem;
            font-weight: 900;
            margin: 0;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.5);
            color: var(--white);
            letter-spacing: -1px;
            line-height: 1.1;
        }
        
        .header-text p {
            font-size: 1.5rem;
            margin: 1rem 0 0 0;
            opacity: 0.95;
            font-weight: 500;
            color: var(--primary-cream);
            line-height: 1.3;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-top: 0.75rem;
            color: var(--light-cream);
            font-weight: 400;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .header-subtitle::before {
            content: '✨';
            font-size: 1.2rem;
        }
        
        .header-subtitle::after {
            content: '✨';
            font-size: 1.2rem;
        }
        
        /* Mobile responsive header */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 2rem;
                padding: 0 1rem;
            }
            
            .header-text {
                text-align: center;
            }
            
            .header-text h1 {
                font-size: 2.8rem;
            }
            
            .header-text p {
                font-size: 1.3rem;
            }
            
            .logo-container img {
                height: 120px;
            }
            
            .logo-container {
                padding: 1.25rem;
            }
        }
        
        @media (max-width: 480px) {
            .header-text h1 {
                font-size: 2.4rem;
            }
            
            .header-text p {
                font-size: 1.1rem;
            }
            
            .logo-container img {
                height: 100px;
            }
        }
        
        /* Global four-color theme styling */
        .stButton > button {
            background-color: var(--primary-coral) !important;
            border-color: var(--primary-coral) !important;
            color: white !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            padding: 0.75rem 1.5rem !important;
        }
        
        .stButton > button:hover {
            background-color: var(--secondary-coral) !important;
            border-color: var(--secondary-coral) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 20px rgba(255, 103, 77, 0.4) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) !important;
        }
        
        /* Secondary button styling */
        .stButton > button[kind="secondary"] {
            background-color: transparent !important;
            border-color: var(--primary-nile) !important;
            color: var(--primary-nile) !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--primary-nile) !important;
            color: white !important;
            box-shadow: 0 8px 20px rgba(24, 83, 91, 0.4) !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: var(--primary-cream);
            border-radius: 12px;
            padding: 0.75rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            color: var(--gray-dark);
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-coral) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(255, 103, 77, 0.3);
        }
        
        .stTabs [aria-selected="false"]:hover {
            background-color: var(--light-coral) !important;
            color: white !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, var(--primary-cream) 0%, white 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 5px solid var(--primary-coral);
            box-shadow: 0 6px 20px rgba(255, 103, 77, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 103, 77, 0.2);
        }
        
        /* Status indicators */
        .status-success {
            border-left-color: #10B981;
            background: linear-gradient(135deg, #D1FAE5 0%, #F0FDF4 100%);
        }
        
        .status-error {
            border-left-color: var(--primary-coral);
            background: linear-gradient(135deg, #FEE2E2 0%, #FEF2F2 100%);
        }
        
        .status-warning {
            border-left-color: var(--light-nile);
            background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
        }
        
        /* Data editor styling */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(255, 103, 77, 0.1);
        }
        
        /* Calendar styling */
        .calendar-container {
            background: var(--primary-cream);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 6px 20px rgba(255, 103, 77, 0.1);
        }
        
        /* Form styling */
        .stTextInput > div > div > input {
            border-color: var(--light-nile) !important;
            border-radius: 10px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-coral) !important;
            box-shadow: 0 0 0 3px rgba(255, 103, 77, 0.2) !important;
        }
        
        .stSelectbox > div > div > div {
            border-color: var(--light-nile) !important;
            border-radius: 10px !important;
        }
        
        .stSelectbox > div > div > div:focus-within {
            border-color: var(--primary-coral) !important;
            box-shadow: 0 0 0 3px rgba(255, 103, 77, 0.2) !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--primary-cream) !important;
            border-radius: 10px !important;
            color: var(--gray-dark) !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: var(--light-coral) !important;
            color: white !important;
        }
        
        /* Success/Error messages */
        .stAlert {
            border-radius: 10px !important;
            border-left: 5px solid !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            border-left-color: var(--primary-coral) !important;
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
    
    # Main content area with four-color theme
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E6E1C5 0%, #FFFFFF 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(255, 103, 77, 0.1);">
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📅 Calendar", "⚙️ Settings", "👥 Providers", "📊 Grid View", 
        "🔄 Sync", "📝 Requests", "💾 Data", "🐛 Debug Test"
    ])
    
    # Calendar Tab
    with tab1:
        st.markdown("### 📅 Schedule Calendar")
        
        # Month navigation
        render_month_navigation()
        
        # Calendar display
        if st.session_state.events:
            st.markdown('<div class="calendar-container">', unsafe_allow_html=True)
            render_calendar(st.session_state.events)
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
                # Check if validation has violations
                has_violations = len(validation.get("violations", [])) > 0 or len(validation.get("provider_violations", {})) > 0
                
                if not has_violations:
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
            
            # Detailed violations - redesigned with tabs and columns
            if has_violations:
                st.markdown("### 🔍 Violation Details")
                
                # Categorize violations by type
                block_violations = []
                rest_violations = []
                capacity_violations = []
                weekend_violations = []
                night_violations = []
                general_violations = []
                
                # Categorize general violations
                for violation in validation["violations"]:
                    violation_lower = violation.lower()
                    if "block" in violation_lower or "consecutive" in violation_lower:
                        block_violations.append(violation)
                    elif "rest" in violation_lower or "between" in violation_lower:
                        rest_violations.append(violation)
                    elif "capacity" in violation_lower or "over" in violation_lower:
                        capacity_violations.append(violation)
                    elif "weekend" in violation_lower:
                        weekend_violations.append(violation)
                    elif "night" in violation_lower:
                        night_violations.append(violation)
                    else:
                        general_violations.append(violation)
                
                # Categorize provider violations
                if validation["provider_violations"]:
                    for provider, violations in validation["provider_violations"].items():
                        for violation in violations:
                            violation_lower = violation.lower()
                            if "block" in violation_lower or "consecutive" in violation_lower:
                                block_violations.append(f"**{provider}:** {violation}")
                            elif "rest" in violation_lower or "between" in violation_lower:
                                rest_violations.append(f"**{provider}:** {violation}")
                            elif "capacity" in violation_lower or "over" in violation_lower:
                                capacity_violations.append(f"**{provider}:** {violation}")
                            elif "weekend" in violation_lower:
                                weekend_violations.append(f"**{provider}:** {violation}")
                            elif "night" in violation_lower:
                                night_violations.append(f"**{provider}:** {violation}")
                            else:
                                general_violations.append(f"**{provider}:** {violation}")
                
                # Create tabs for different violation types
                tab_names = []
                if block_violations:
                    tab_names.append("Block Issues")
                if rest_violations:
                    tab_names.append("Rest Periods")
                if capacity_violations:
                    tab_names.append("Capacity")
                if weekend_violations:
                    tab_names.append("Weekend")
                if night_violations:
                    tab_names.append("Night Shifts")
                if general_violations:
                    tab_names.append("General")
                
                if tab_names:
                    violation_tabs = st.tabs(tab_names)
                    
                    tab_index = 0
                    
                    if block_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### Block & Consecutive Shift Issues")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(block_violations) // 2 + len(block_violations) % 2
                            
                            with cols[0]:
                                for violation in block_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in block_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                    
                    if rest_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### Rest Period Violations")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(rest_violations) // 2 + len(rest_violations) % 2
                            
                            with cols[0]:
                                for violation in rest_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in rest_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                    
                    if capacity_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### Capacity & Over-Assignment Issues")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(capacity_violations) // 2 + len(capacity_violations) % 2
                            
                            with cols[0]:
                                for violation in capacity_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in capacity_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                    
                    if weekend_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### Weekend Shift Issues")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(weekend_violations) // 2 + len(weekend_violations) % 2
                            
                            with cols[0]:
                                for violation in weekend_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in weekend_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                    
                    if night_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### Night Shift Issues")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(night_violations) // 2 + len(night_violations) % 2
                            
                            with cols[0]:
                                for violation in night_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in night_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                    
                    if general_violations:
                        with violation_tabs[tab_index]:
                            st.markdown("#### General Schedule Issues")
                            # Use columns for better layout
                            cols = st.columns(2)
                            mid_point = len(general_violations) // 2 + len(general_violations) % 2
                            
                            with cols[0]:
                                for violation in general_violations[:mid_point]:
                                    st.markdown(f"• {violation}")
                            
                            with cols[1]:
                                for violation in general_violations[mid_point:]:
                                    st.markdown(f"• {violation}")
                        tab_index += 1
                else:
                    # Fallback if no violations to categorize
                    with st.expander("🔍 View All Violations", expanded=False):
                        for violation in validation["violations"]:
                            st.markdown(f"• {violation}")
                        
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
            
            st.session_state.global_rules.expected_shifts_per_month = st.number_input(
                "Expected Shifts Per Month", 
                min_value=1, max_value=31, value=getattr(st.session_state.global_rules, 'expected_shifts_per_month', 15),
                help="Expected shifts per month (15 for 30-day months, 16 for 31-day months)"
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
        
        # Use the correct shift type structure from constants
        correct_shift_types = [
            {"key": "R12", "label": "7am–7pm Rounder", "start": "07:00", "end": "19:00", "color": "#16a34a"},
            {"key": "A12", "label": "7am–7pm Admitter", "start": "07:00", "end": "19:00", "color": "#f59e0b"},
            {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
            {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
            {"key": "NB", "label": "Night Bridge", "start": "23:00", "end": "07:00", "color": "#06b6d4"},
            {"key": "APP", "label": "APP Provider", "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
        ]
        
        # Update session state if it doesn't match the correct structure
        if not hasattr(st.session_state, 'shift_types') or len(st.session_state.shift_types) != len(correct_shift_types):
            st.session_state.shift_types = correct_shift_types.copy()
        
        for i, shift_type in enumerate(st.session_state.shift_types):
            # Ensure the shift type has the correct structure
            if not isinstance(shift_type, dict) or 'key' not in shift_type:
                st.session_state.shift_types[i] = correct_shift_types[i]
                shift_type = st.session_state.shift_types[i]
            
            with st.expander(f"Shift Type: {shift_type.get('label', f'Shift {i+1}')}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.session_state.shift_types[i]['key'] = st.text_input(
                        "Key", value=shift_type.get('key', ''), key=f"shift_key_{i}"
                    )
                
                with col2:
                    st.session_state.shift_types[i]['label'] = st.text_input(
                        "Label", value=shift_type.get('label', ''), key=f"shift_label_{i}"
                    )
                
                with col3:
                    st.session_state.shift_types[i]['start'] = st.text_input(
                        "Start Time", value=shift_type.get('start', '07:00'), key=f"shift_start_{i}"
                    )
                
                with col4:
                    st.session_state.shift_types[i]['end'] = st.text_input(
                        "End Time", value=shift_type.get('end', '19:00'), key=f"shift_end_{i}"
                    )
                
                # Color picker for shift type
                st.session_state.shift_types[i]['color'] = st.color_picker(
                    "Color", value=shift_type.get('color', '#1f77b4'), key=f"shift_color_{i}"
                )
        
        # Shift Capacity Configuration
        st.subheader("📊 Shift Capacity")
        
        # Use the correct capacity mapping
        default_capacities = {"R12": 13, "A12": 1, "A10": 2, "N12": 4, "NB": 1, "APP": 2}
        
        for shift_type in st.session_state.shift_types:
            if not isinstance(shift_type, dict) or 'key' not in shift_type:
                continue
            
            shift_key = shift_type.get('key', 'Unknown')
            shift_label = shift_type.get('label', shift_key)
            default_capacity = default_capacities.get(shift_key, 1)
            
            capacity = st.number_input(
                f"Capacity for {shift_label} ({shift_key})", 
                min_value=1, max_value=20, 
                value=st.session_state.shift_capacity.get(shift_key, default_capacity),
                key=f"capacity_{shift_key}"
            )
            st.session_state.shift_capacity[shift_key] = capacity
        
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
    
    # Debug Test Tab
    with tab8:
        st.header("🐛 Debug Test")
        st.markdown("Run quick tests to identify scheduling issues.")
        
        if st.button("🔍 Run Shift Count Test", type="primary"):
            try:
                # Load current data
                providers_df, _ = load_providers()
                if providers_df.empty:
                    providers_df = pd.read_csv('IMIS_initials.csv')
                providers = providers_df['initials'].tolist()
                
                global_rules_dict, shift_types, shift_capacity, provider_rules = load_rules()
                global_rules = RuleConfig(**global_rules_dict) if global_rules_dict else RuleConfig()
                
                # Get current month info
                current_date = datetime.now()
                year, month = current_date.year, current_date.month
                month_days = make_month_days(year, month)
                days_in_month = len(month_days)
                from core.utils import get_expected_shifts_for_month
                expected_shifts = get_expected_shifts_for_month(year, month)
                
                st.markdown(f"**Testing {month}/{year} ({days_in_month} days)**")
                st.markdown(f"**Expected shifts: {expected_shifts}**")
                
                # Generate schedule
                with st.spinner("Generating schedule..."):
                    events = generate_schedule(year, month, providers, shift_types, shift_capacity, 
                                             provider_rules, global_rules)
                
                st.success(f"Generated {len(events)} events")
                
                # Validate rules
                validation_result = validate_rules(events, providers, global_rules, provider_rules)
                
                # Display enhanced results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", len(events))
                with col2:
                    st.metric("Total Violations", validation_result['summary']['total_violations'])
                with col3:
                    st.metric("Coverage Gaps", validation_result['summary']['coverage_gaps_count'])
                with col4:
                    st.metric("Schedule Valid", "✅" if validation_result['is_valid'] else "❌")
                
                # Enhanced Provider Analysis
                st.markdown("### 📊 Enhanced Provider Analysis")
                provider_stats = validation_result['summary']['provider_stats']
                
                # Create enhanced analysis dataframe
                analysis_data = []
                for provider, stats in provider_stats.items():
                    total_shifts = stats['total_shifts']
                    expected_shifts = stats.get('expected_shifts', 'N/A')
                    shift_types = stats.get('shift_types', [])
                    
                    # Determine status based on expected shifts
                    if provider in APP_PROVIDER_INITIALS:
                        status = "🟢 APP Provider"
                        expected_display = "N/A"
                        status_detail = "APP providers have different rules"
                    elif expected_shifts == 'N/A':
                        status = "⚠️ No Expected Shifts"
                        expected_display = "N/A"
                        status_detail = "Expected shifts not calculated"
                    else:
                        expected_shifts = int(expected_shifts)
                        if total_shifts > expected_shifts:
                            status = "❌ EXCEEDS Expected"
                            status_detail = f"Has {total_shifts} shifts, expected {expected_shifts}"
                        elif total_shifts < (expected_shifts - 1):
                            status = "⚠️ Below Expected"
                            status_detail = f"Has {total_shifts} shifts, expected {expected_shifts}"
                        else:
                            status = "✅ Within Range"
                            status_detail = f"Has {total_shifts} shifts, expected {expected_shifts}"
                        expected_display = expected_shifts
                    
                    analysis_data.append({
                        "Provider": provider,
                        "Total Shifts": total_shifts,
                        "Expected": expected_display,
                        "Status": status,
                        "Shift Types": ", ".join(shift_types) if shift_types else "None",
                        "Weekend": stats['weekend_shifts'],
                        "Night": stats['night_shifts'],
                        "Rounder": stats['rounder_shifts'],
                        "Admitting": stats['admitting_shifts'],
                        "Details": status_detail
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
                
                # Enhanced Violations Display
                if validation_result['violations']:
                    st.markdown("### ❌ All Violations Found")
                    
                    # Group violations by type
                    if validation_result.get('preference_violations'):
                        st.markdown("#### 🚫 Shift Type Preference Violations")
                        for violation in validation_result['preference_violations']:
                            st.error(violation)
                    
                    if validation_result.get('rest_violations'):
                        st.markdown("#### 😴 Rest Day Violations")
                        for violation in validation_result['rest_violations']:
                            st.error(violation)
                    
                    # Other violations
                    other_violations = [v for v in validation_result['violations'] 
                                      if v not in validation_result.get('preference_violations', [])
                                      and v not in validation_result.get('rest_violations', [])]
                    if other_violations:
                        st.markdown("#### ⚠️ Other Violations")
                        for violation in other_violations:
                            st.warning(violation)
                
                # Coverage Gaps Analysis
                if validation_result.get('coverage_gaps'):
                    st.markdown("### 🔍 Coverage Gaps Analysis")
                    st.markdown("These are days/shifts where not all required slots are filled:")
                    
                    # Group gaps by shift type
                    gap_by_type = {}
                    for gap in validation_result['coverage_gaps']:
                        # Extract shift type from gap message
                        if "R12" in gap:
                            shift_type = "R12"
                        elif "A12" in gap:
                            shift_type = "A12"
                        elif "A10" in gap:
                            shift_type = "A10"
                        elif "N12" in gap:
                            shift_type = "N12"
                        elif "NB" in gap:
                            shift_type = "NB"
                        elif "APP" in gap:
                            shift_type = "APP"
                        else:
                            shift_type = "Unknown"
                        
                        if shift_type not in gap_by_type:
                            gap_by_type[shift_type] = []
                        gap_by_type[shift_type].append(gap)
                    
                    # Display gaps by type
                    for shift_type, gaps in gap_by_type.items():
                        with st.expander(f"{shift_type} Gaps ({len(gaps)} gaps)"):
                            for gap in gaps:
                                st.info(gap)
                
                # Summary Statistics
                st.markdown("### 📈 Summary Statistics")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Preference Violations", validation_result['summary']['preference_violations_count'])
                    st.metric("Rest Violations", validation_result['summary']['rest_violations_count'])
                
                with summary_col2:
                    st.metric("Providers Used", validation_result['summary']['providers_used'])
                    st.metric("Total Providers", len(providers))
                
                with summary_col3:
                    coverage_percentage = ((len(events) / (validation_result['summary']['total_events'] + validation_result['summary']['coverage_gaps_count'])) * 100) if (validation_result['summary']['total_events'] + validation_result['summary']['coverage_gaps_count']) > 0 else 0
                    st.metric("Coverage %", f"{coverage_percentage:.1f}%")
                
                # Shift distribution
                st.markdown("### Shift Distribution")
                shift_type_counts = {}
                for event in events:
                    shift_type = event.extendedProps.get("shift_type")
                    if shift_type:
                        shift_type_counts[shift_type] = shift_type_counts.get(shift_type, 0) + 1
                
                shift_dist_df = pd.DataFrame([
                    {"Shift Type": k, "Count": v} for k, v in shift_type_counts.items()
                ])
                st.dataframe(shift_dist_df, use_container_width=True)
                
                # Coverage analysis
                total_available = 0
                for day in month_days:
                    for shift_type, capacity in shift_capacity.items():
                        total_available += capacity
                
                coverage_pct = (len(events) / total_available * 100) if total_available > 0 else 0
                st.markdown(f"**Coverage: {len(events)}/{total_available} shifts ({coverage_pct:.1f}%)**")
                
            except Exception as e:
                st.error(f"Test failed: {e}")
                st.error(f"Error details: {traceback.format_exc()}")
        
        if st.button("🔧 Test Scheduler Logic", type="secondary"):
            try:
                st.markdown("### Scheduler Logic Analysis")
                
                # Load data
                providers_df, _ = load_providers()
                if providers_df.empty:
                    providers_df = pd.read_csv('IMIS_initials.csv')
                providers = providers_df['initials'].tolist()
                global_rules_dict, shift_types, shift_capacity, provider_rules = load_rules()
                global_rules = RuleConfig(**global_rules_dict) if global_rules_dict else RuleConfig()
                
                # Provider analysis
                app_providers = [p for p in providers if p in APP_PROVIDER_INITIALS]
                nocturnists = [p for p in providers if p in NOCTURNISTS]
                physician_providers = [p for p in providers if p not in APP_PROVIDER_INITIALS and p not in NOCTURNISTS]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Providers", len(providers))
                with col2:
                    st.metric("APP Providers", len(app_providers))
                with col3:
                    st.metric("Physician Providers", len(physician_providers))
                
                st.markdown("**Provider Types:**")
                st.markdown(f"- APP Providers: {', '.join(app_providers)}")
                st.markdown(f"- Nocturnists: {', '.join(nocturnists)}")
                st.markdown(f"- Other Physicians: {', '.join(physician_providers)}")
                
                st.markdown("**Global Rules:**")
                st.markdown(f"- Expected shifts per month: {global_rules.expected_shifts_per_month}")
                st.markdown(f"- Min days between shifts: {global_rules.min_days_between_shifts}")
                
                st.markdown("**Shift Capacity:**")
                for shift_type, capacity in shift_capacity.items():
                    st.markdown(f"- {shift_type}: {capacity} slots per day")
                
            except Exception as e:
                st.error(f"Logic test failed: {e}")
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
