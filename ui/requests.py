# =============================================================================
# Provider Requests UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import uuid
from datetime import date, datetime
from typing import List, Dict, Any
import pandas as pd

from models.constants import APP_PROVIDER_INITIALS

def provider_requests_panel():
    """Panel for managing provider requests (vacations, blackout dates, shift swaps)."""
    st.subheader("📝 Provider Requests Management")
    
    # Initialize requests in session state
    if "provider_requests" not in st.session_state:
        st.session_state.provider_requests = {
            "vacations": [],
            "blackout_dates": [],
            "shift_swaps": []
        }
    
    # Get all providers
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded. Please load providers first.")
        return
    
    all_providers = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
    app_providers = sorted(APP_PROVIDER_INITIALS)
    
    # Create provider options with separators
    provider_options = ["(Select Provider)"]
    if all_providers:
        provider_options.append("--- Physicians ---")
        provider_options.extend(all_providers)
    if app_providers:
        provider_options.append("--- APPs ---")
        provider_options.extend(app_providers)
    
    # Request type selection
    request_type = st.selectbox(
        "Request Type",
        options=["Vacation Request", "Blackout Date Request", "Shift Swap Request"],
        key="request_type_select"
    )
    
    if request_type == "Vacation Request":
        vacation_request_form(provider_options)
    elif request_type == "Blackout Date Request":
        blackout_date_request_form(provider_options)
    elif request_type == "Shift Swap Request":
        shift_swap_request_form(provider_options)
    
    # Display existing requests
    display_existing_requests()

def vacation_request_form(provider_options):
    """Form for submitting vacation requests."""
    st.subheader("🏖️ Vacation Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="vacation_provider")
        start_date = st.date_input("Start Date", key="vacation_start")
    with col2:
        end_date = st.date_input("End Date", key="vacation_end")
        reason = st.text_area("Reason (optional)", key="vacation_reason")
    
    if st.button("Submit Vacation Request"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        elif start_date >= end_date:
            st.error("End date must be after start date.")
        else:
            request = {
                "id": str(uuid.uuid4()),
                "type": "vacation",
                "provider": provider,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            st.session_state.provider_requests["vacations"].append(request)
            st.success("Vacation request submitted successfully!")

def blackout_date_request_form(provider_options):
    """Form for submitting blackout date requests."""
    st.subheader("🚫 Blackout Date Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", options=provider_options, key="blackout_provider")
        blackout_date = st.date_input("Blackout Date", key="blackout_date")
    with col2:
        reason = st.text_area("Reason (optional)", key="blackout_reason")
    
    if st.button("Submit Blackout Request"):
        if provider == "(Select Provider)" or provider.startswith("---"):
            st.error("Please select a provider.")
        else:
            request = {
                "id": str(uuid.uuid4()),
                "type": "blackout",
                "provider": provider,
                "date": blackout_date.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            st.session_state.provider_requests["blackout_dates"].append(request)
            st.success("Blackout date request submitted successfully!")

def shift_swap_request_form(provider_options):
    """Form for submitting shift swap requests."""
    st.subheader("�� Shift Swap Request")
    
    col1, col2 = st.columns(2)
    with col1:
        provider1 = st.selectbox("Provider 1", options=provider_options, key="swap_provider1")
        day1 = st.date_input("Provider 1's Day", key="swap_day1")
    with col2:
        provider2 = st.selectbox("Provider 2", options=provider_options, key="swap_provider2")
        day2 = st.date_input("Provider 2's Day", key="swap_day2")
    
    reason = st.text_area("Reason for swap (optional)", key="swap_reason")
    
    if st.button("Submit Swap Request"):
        if (provider1 == "(Select Provider)" or provider1.startswith("---") or
            provider2 == "(Select Provider)" or provider2.startswith("---")):
            st.error("Please select both providers.")
        elif provider1 == provider2:
            st.error("Providers must be different.")
        elif day1 == day2:
            st.error("Days must be different.")
        else:
            request = {
                "id": str(uuid.uuid4()),
                "type": "swap",
                "provider1": provider1,
                "day1": day1.isoformat(),
                "provider2": provider2,
                "day2": day2.isoformat(),
                "reason": reason,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            st.session_state.provider_requests["shift_swaps"].append(request)
            st.success("Shift swap request submitted successfully!")

def display_existing_requests():
    """Display existing requests with approval/rejection options."""
    st.subheader("📋 Existing Requests")
    
    requests = st.session_state.provider_requests
    
    # Display vacations
    if requests["vacations"]:
        st.write("**��️ Vacation Requests**")
        for req in requests["vacations"]:
            with st.expander(f"{req['provider']} - {req['start_date']} to {req['end_date']}"):
                st.write(f"**Provider:** {req['provider']}")
                st.write(f"**Dates:** {req['start_date']} to {req['end_date']}")
                if req['reason']:
                    st.write(f"**Reason:** {req['reason']}")
                st.write(f"**Status:** {req['status']}")
                
                if req['status'] == "pending":
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approve", key=f"approve_vacation_{req['id']}"):
                            req['status'] = "approved"
                            st.success("Request approved!")
                            st.rerun()
                    with col2:
                        if st.button("❌ Reject", key=f"reject_vacation_{req['id']}"):
                            req['status'] = "rejected"
                            st.error("Request rejected!")
                            st.rerun()
    
    # Display blackout dates
    if requests["blackout_dates"]:
        st.write("**🚫 Blackout Date Requests**")
        for req in requests["blackout_dates"]:
            with st.expander(f"{req['provider']} - {req['date']}"):
                st.write(f"**Provider:** {req['provider']}")
                st.write(f"**Date:** {req['date']}")
                if req['reason']:
                    st.write(f"**Reason:** {req['reason']}")
                st.write(f"**Status:** {req['status']}")
                
                if req['status'] == "pending":
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approve", key=f"approve_blackout_{req['id']}"):
                            req['status'] = "approved"
                            st.success("Request approved!")
                            st.rerun()
                    with col2:
                        if st.button("❌ Reject", key=f"reject_blackout_{req['id']}"):
                            req['status'] = "rejected"
                            st.error("Request rejected!")
                            st.rerun()
    
    # Display shift swaps
    if requests["shift_swaps"]:
        st.write("**🔄 Shift Swap Requests**")
        for req in requests["shift_swaps"]:
            with st.expander(f"{req['provider1']} ↔ {req['provider2']}"):
                st.write(f"**Provider 1:** {req['provider1']} on {req['day1']}")
                st.write(f"**Provider 2:** {req['provider2']} on {req['day2']}")
                if req['reason']:
                    st.write(f"**Reason:** {req['reason']}")
                st.write(f"**Status:** {req['status']}")
                
                if req['status'] == "pending":
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approve", key=f"approve_swap_{req['id']}"):
                            req['status'] = "approved"
                            # Execute the swap
                            if execute_shift_swap(req['provider1'], req['day1'], req['provider2'], req['day2']):
                                st.success("Request approved and swap executed!")
                            else:
                                st.error("Request approved but swap failed!")
                            st.rerun()
                    with col2:
                        if st.button("❌ Reject", key=f"reject_swap_{req['id']}"):
                            req['status'] = "rejected"
                            st.error("Request rejected!")
                            st.rerun()

def execute_shift_swap(provider1: str, day1: str, provider2: str, day2: str) -> bool:
    """Execute a shift swap between two providers."""
    try:
        # Convert string dates to date objects
        day1_date = date.fromisoformat(day1)
        day2_date = date.fromisoformat(day2)
        
        # Find the events to swap
        events = st.session_state.get("events", [])
        event1 = None
        event2 = None
        
        for event in events:
            event_date = datetime.fromisoformat(event["start"]).date()
            event_provider = event.get("extendedProps", {}).get("provider", "")
            
            if event_date == day1_date and event_provider == provider1:
                event1 = event
            elif event_date == day2_date and event_provider == provider2:
                event2 = event
        
        if event1 and event2:
            # Swap the providers
            event1["extendedProps"]["provider"] = provider2
            event1["title"] = event1["title"].replace(provider1, provider2)
            
            event2["extendedProps"]["provider"] = provider1
            event2["title"] = event2["title"].replace(provider2, provider1)
            
            return True
        else:
            st.error("Could not find the events to swap.")
            return False
            
    except Exception as e:
        st.error(f"Error during shift swap: {e}")
        return False
