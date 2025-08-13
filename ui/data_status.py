"""
Data Status Display Component for IMIS Scheduler
Shows the status of saved data and provides data management options
"""
import streamlit as st
import os
from datetime import datetime
from core.data_manager import (
    DATA_DIR, PROVIDERS_FILE, RULES_FILE, SCHEDULES_FILE,
    load_all_schedules, load_providers, load_rules
)

def render_data_status():
    """Render data status and management panel."""
    st.subheader("💾 Data Management")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        st.info("No saved data found. Data will be automatically created when you first use the app.")
        return
    
    # Data files status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if os.path.exists(PROVIDERS_FILE):
            providers_df, _ = load_providers()
            if not providers_df.empty:
                st.success(f"Providers: {len(providers_df)}")
            else:
                st.warning("Providers: Empty")
        else:
            st.error("Providers: Not saved")
    
    with col2:
        if os.path.exists(RULES_FILE):
            _, shift_types, _ = load_rules()
            if shift_types:
                st.success(f"Rules: {len(shift_types)} shift types")
            else:
                st.warning("Rules: Empty")
        else:
            st.error("Rules: Not saved")
    
    with col3:
        if os.path.exists(SCHEDULES_FILE):
            schedules = load_all_schedules()
            if schedules:
                st.success(f"Schedules: {len(schedules)} months")
            else:
                st.warning("Schedules: Empty")
        else:
            st.error("Schedules: Not saved")
    
    with col4:
        # Show last modified time
        files = [PROVIDERS_FILE, RULES_FILE, SCHEDULES_FILE]
        latest_time = None
        for file in files:
            if os.path.exists(file):
                file_time = os.path.getmtime(file)
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
        
        if latest_time:
            last_modified = datetime.fromtimestamp(latest_time)
            st.info(f"📅 Last saved: {last_modified.strftime('%m/%d %H:%M')}")
        else:
            st.info("📅 No data saved yet")
    
    # Data management options
    with st.expander("🔧 Data Management Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.session_state.get("providers_df") is not None:
                    st.session_state.providers_df = st.session_state.providers_df.iloc[0:0]  # Clear DataFrame
                st.session_state.providers_loaded = False
                st.session_state.events = []
                st.session_state.validation_results = None
                
                # Clear saved files
                for file in [PROVIDERS_FILE, RULES_FILE, SCHEDULES_FILE]:
                    if os.path.exists(file):
                        os.remove(file)
                
                st.success("All data cleared! Refresh the page to see changes.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reload Data", type="secondary"):
                from core.data_manager import auto_load_session_state
                auto_load_session_state()
                st.success("Data reloaded from saved files!")
                st.rerun()
        
        # Show saved schedules
        if os.path.exists(SCHEDULES_FILE):
            schedules = load_all_schedules()
            if schedules:
                st.subheader("📅 Saved Schedules")
                for month_key, schedule_data in schedules.items():
                    year, month = month_key.split("-")
                    event_count = len(schedule_data.get("events", []))
                    last_updated = schedule_data.get("last_updated", "Unknown")
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.write(f"**{month}/{year}**: {event_count} events")
                    with col2:
                        if st.button("Load", key=f"load_{month_key}"):
                            st.session_state.current_year = int(year)
                            st.session_state.current_month = int(month)
                            st.session_state.events = schedule_data.get("events", [])
                            st.success(f"Loaded schedule for {month}/{year}")
                            st.rerun()
                    with col3:
                        st.caption(f"Updated: {last_updated[:10]}")
