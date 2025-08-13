# =============================================================================
# Provider Requests UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from typing import List, Dict, Any
import json

def provider_requests_panel():
    """Main provider requests panel."""
    st.header("📝 Provider Requests")
    
    # Request types
    request_type = st.selectbox(
        "Request Type",
        options=["Vacation Request", "Blackout Date", "Shift Swap", "General Request"],
        key="request_type"
    )
    
    # Provider selection
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
        selected_provider = st.selectbox("Provider", options=providers, key="request_provider")
    else:
        selected_provider = st.text_input("Provider Initials", key="request_provider")
    
    # Request form based on type
    if request_type == "Vacation Request":
        vacation_request_form(selected_provider)
    elif request_type == "Blackout Date":
        blackout_date_request_form(selected_provider)
    elif request_type == "Shift Swap":
        shift_swap_request_form(selected_provider)
    else:
        general_request_form(selected_provider)
    
    # Display existing requests
    st.subheader("📋 Existing Requests")
    display_existing_requests()

def vacation_request_form(provider: str):
    """Vacation request form."""
    st.subheader("🏖️ Vacation Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", key="vacation_start")
    
    with col2:
        end_date = st.date_input("End Date", key="vacation_end")
    
    reason = st.text_area("Reason for Vacation", key="vacation_reason")
    priority = st.select_slider("Priority", options=["Low", "Medium", "High"], value="Medium", key="vacation_priority")
    
    if st.button("Submit Vacation Request", type="primary"):
        if not provider or not reason or start_date >= end_date:
            st.error("Please fill in all fields correctly. Start date must be before end date.")
        else:
            request = {
                "id": f"vac_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "Vacation Request",
                "provider": provider,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
                "priority": priority,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            
            if "requests" not in st.session_state:
                st.session_state.requests = []
            
            st.session_state.requests.append(request)
            st.success("Vacation request submitted successfully!")
            st.rerun()

def blackout_date_request_form(provider: str):
    """Blackout date request form."""
    st.subheader("🚫 Blackout Date Request")
    
    blackout_date = st.date_input("Blackout Date", key="blackout_date")
    reason = st.text_area("Reason for Blackout", key="blackout_reason")
    priority = st.select_slider("Priority", options=["Low", "Medium", "High"], value="Medium", key="blackout_priority")
    
    if st.button("Submit Blackout Request", type="primary"):
        if not provider or not reason:
            st.error("Please fill in all fields.")
        else:
            request = {
                "id": f"blk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "Blackout Date",
                "provider": provider,
                "date": blackout_date.isoformat(),
                "reason": reason,
                "priority": priority,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            
            if "requests" not in st.session_state:
                st.session_state.requests = []
            
            st.session_state.requests.append(request)
            st.success("Blackout date request submitted successfully!")
            st.rerun()

def shift_swap_request_form(provider: str):
    """Shift swap request form."""
    st.subheader("🔄 Shift Swap Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Shift**")
        current_date = st.date_input("Current Shift Date", key="swap_current_date")
        current_shift = st.selectbox("Current Shift Type", options=["Day", "Night", "APP"], key="swap_current_shift")
    
    with col2:
        st.write("**Desired Shift**")
        desired_date = st.date_input("Desired Shift Date", key="swap_desired_date")
        desired_shift = st.selectbox("Desired Shift Type", options=["Day", "Night", "APP"], key="swap_desired_shift")
    
    swap_with = st.text_input("Swap with Provider (Optional)", key="swap_with_provider")
    reason = st.text_area("Reason for Swap", key="swap_reason")
    
    if st.button("Submit Swap Request", type="primary"):
        if not provider or not reason:
            st.error("Please fill in all required fields.")
        else:
            request = {
                "id": f"swap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "Shift Swap",
                "provider": provider,
                "current_date": current_date.isoformat(),
                "current_shift": current_shift,
                "desired_date": desired_date.isoformat(),
                "desired_shift": desired_shift,
                "swap_with": swap_with if swap_with else None,
                "reason": reason,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            
            if "requests" not in st.session_state:
                st.session_state.requests = []
            
            st.session_state.requests.append(request)
            st.success("Shift swap request submitted successfully!")
            st.rerun()

def general_request_form(provider: str):
    """General request form."""
    st.subheader("📝 General Request")
    
    request_title = st.text_input("Request Title", key="general_title")
    request_description = st.text_area("Request Description", key="general_description")
    priority = st.select_slider("Priority", options=["Low", "Medium", "High"], value="Medium", key="general_priority")
    
    if st.button("Submit General Request", type="primary"):
        if not provider or not request_title or not request_description:
            st.error("Please fill in all fields.")
        else:
            request = {
                "id": f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "General Request",
                "provider": provider,
                "title": request_title,
                "description": request_description,
                "priority": priority,
                "status": "pending",
                "submitted_at": datetime.now().isoformat()
            }
            
            if "requests" not in st.session_state:
                st.session_state.requests = []
            
            st.session_state.requests.append(request)
            st.success("General request submitted successfully!")
            st.rerun()

def display_existing_requests():
    """Display existing requests with approval/rejection options."""
    if "requests" not in st.session_state or not st.session_state.requests:
        st.info("No requests submitted yet.")
        return
    
    # Filter requests by status
    pending_requests = [req for req in st.session_state.requests if req["status"] == "pending"]
    processed_requests = [req for req in st.session_state.requests if req["status"] != "pending"]
    
    # Display pending requests
    if pending_requests:
        st.subheader("⏳ Pending Requests")
        for i, request in enumerate(pending_requests):
            with st.expander(f"{request['type']} - {request['provider']} ({request['priority']})", expanded=True):
                display_request_details(request)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("✅ Approve", key=f"approve_{request['id']}"):
                        request["status"] = "approved"
                        request["processed_at"] = datetime.now().isoformat()
                        st.success("Request approved!")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Reject", key=f"reject_{request['id']}"):
                        request["status"] = "rejected"
                        request["processed_at"] = datetime.now().isoformat()
                        st.success("Request rejected!")
                        st.rerun()
                
                with col3:
                    if st.button("🔄 Execute Swap", key=f"execute_{request['id']}"):
                        if request["type"] == "Shift Swap":
                            execute_shift_swap(request)
                        else:
                            st.info("This action is only available for shift swap requests.")
    
    # Display processed requests
    if processed_requests:
        st.subheader("✅ Processed Requests")
        for request in processed_requests:
            status_color = "green" if request["status"] == "approved" else "red"
            with st.expander(f"{request['type']} - {request['provider']} ({request['status']})", expanded=False):
                display_request_details(request)
                st.markdown(f"**Status:** :{status_color}[{request['status'].upper()}]")
                if "processed_at" in request:
                    st.markdown(f"**Processed:** {request['processed_at']}")

def display_request_details(request: Dict[str, Any]):
    """Display request details."""
    st.write(f"**Provider:** {request['provider']}")
    st.write(f"**Type:** {request['type']}")
    st.write(f"**Priority:** {request['priority']}")
    st.write(f"**Submitted:** {request['submitted_at']}")
    
    if request['type'] == "Vacation Request":
        st.write(f"**Period:** {request['start_date']} to {request['end_date']}")
    elif request['type'] == "Blackout Date":
        st.write(f"**Date:** {request['date']}")
    elif request['type'] == "Shift Swap":
        st.write(f"**Current:** {request['current_date']} ({request['current_shift']})")
        st.write(f"**Desired:** {request['desired_date']} ({request['desired_shift']})")
        if request.get('swap_with'):
            st.write(f"**Swap with:** {request['swap_with']}")
    elif request['type'] == "General Request":
        st.write(f"**Title:** {request['title']}")
    
    st.write(f"**Reason:** {request['reason']}")

def execute_shift_swap(request: Dict[str, Any]):
    """Execute a shift swap request."""
    st.info("Shift swap execution would be implemented here.")
    st.write("This would involve:")
    st.write("1. Finding the current schedule")
    st.write("2. Swapping the shifts")
    st.write("3. Updating the calendar")
    st.write("4. Notifying affected providers")
    
    # For now, just mark as executed
    request["status"] = "executed"
    request["processed_at"] = datetime.now().isoformat()
    st.success("Shift swap executed!")
    st.rerun()
