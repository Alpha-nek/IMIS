# =============================================================================
# IMIS Scheduler - Main Application
# =============================================================================
# 
# This is the main Streamlit application that imports from modular components.
# Run with: streamlit run main.py

import streamlit as st
import sys
import os

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
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state with defaults."""
    st.set_page_config(page_title="IMIS Scheduler", layout="wide", initial_sidebar_state="collapsed")
    
    # Ensure data directory exists
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize session state with defaults
    st.session_state.setdefault("month", date.today().replace(day=1))
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
    
    # Load default providers
    if "providers_df" not in st.session_state or st.session_state.providers_df.empty:
        try:
            if os.path.exists("IMIS_initials.csv"):
                providers_df = pd.read_csv("IMIS_initials.csv")
                providers_df = providers_df.dropna()
                providers_df["initials"] = providers_df["initials"].astype(str).str.strip().str.upper()
                providers_df = providers_df[providers_df["initials"] != ""]
                providers_df = providers_df[providers_df["initials"] != "nan"]
                providers_df = providers_df[providers_df["initials"] != "NO"]
                if not providers_df.empty:
                    st.session_state["providers_df"] = providers_df
                    st.session_state["providers_loaded"] = True
                else:
                    default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                    st.session_state["providers_df"] = default_providers
                    st.session_state["providers_loaded"] = True
            else:
                default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
                st.session_state["providers_df"] = default_providers
                st.session_state["providers_loaded"] = True
        except Exception as e:
            st.error(f"Failed to load providers: {e}")
            default_providers = pd.DataFrame({"initials": PROVIDER_INITIALS_DEFAULT})
            st.session_state["providers_df"] = default_providers
            st.session_state["providers_loaded"] = True

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
    
    # Simple interface for now
    st.subheader("📅 Month Navigation")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Previous Month"):
            st.session_state.month = st.session_state.month - relativedelta(months=1)
            st.rerun()
    with col2:
        st.write(f"**Current:** {st.session_state.month.strftime('%B %Y')}")
    with col3:
        if st.button("Next Month →"):
            st.session_state.month = st.session_state.month + relativedelta(months=1)
            st.rerun()
    
    # Test provider display
    if not st.session_state.providers_df.empty:
        st.subheader("👥 Loaded Providers")
        providers_list = st.session_state.providers_df["initials"].tolist()
        st.write(f"Total providers: {len(providers_list)}")
        st.write("First 10 providers:", providers_list[:10])
    
    # Show modular structure status
    st.subheader("🧪 Modular Structure Status")
    st.success("✅ Modular structure is working correctly!")
    st.info("This is a simplified version. The full functionality will be added as we continue modularizing.")

if __name__ == "__main__":
    main()
