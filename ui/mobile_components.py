# =============================================================================
# Mobile-Specific UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import json

def mobile_request_form() -> Dict[str, Any]:
    """Mobile-optimized request form for providers."""
    st.markdown("### 📝 Submit Request")
    
    # Request type selection
    request_type = st.selectbox(
        "Request Type",
        options=["Vacation Request", "Blackout Date", "Shift Swap", "General Request"],
        key="mobile_request_type"
    )
    
    # Provider selection
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
        selected_provider = st.selectbox("Provider", options=providers, key="mobile_provider")
    else:
        selected_provider = st.text_input("Provider Initials", key="mobile_provider")
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", key="mobile_start_date")
    with col2:
        end_date = st.date_input("End Date", key="mobile_end_date")
    
    # Reason
    reason = st.text_area("Reason", placeholder="Please provide a reason for your request...", key="mobile_reason")
    
    # Priority
    priority = st.select_slider(
        "Priority",
        options=["Low", "Medium", "High", "Urgent"],
        value="Medium",
        key="mobile_priority"
    )
    
    # Submit button
    if st.button("Submit Request", type="primary", use_container_width=True):
        if not selected_provider or not reason:
            st.error("Please fill in all required fields.")
            return {}
        
        request = {
            "id": f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": request_type,
            "provider": selected_provider,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "reason": reason,
            "priority": priority,
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
            "submitted_via": "mobile"
        }
        
        # Store in session state
        if "mobile_requests" not in st.session_state:
            st.session_state.mobile_requests = []
        st.session_state.mobile_requests.append(request)
        
        st.success("Request submitted successfully!")
        return request
    
    return {}

def mobile_quick_actions() -> None:
    """Mobile quick action buttons."""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📅 View My Shifts", use_container_width=True):
            st.session_state.mobile_view = "my_shifts"
            st.rerun()
        
        if st.button("📊 View Schedule", use_container_width=True):
            st.session_state.mobile_view = "schedule"
            st.rerun()
    
    with col2:
        if st.button("📝 New Request", use_container_width=True):
            st.session_state.mobile_view = "new_request"
            st.rerun()
        
        if st.button("🔔 Notifications", use_container_width=True):
            st.session_state.mobile_view = "notifications"
            st.rerun()

def mobile_my_shifts_view() -> None:
    """Mobile view for providers to see their shifts."""
    st.markdown("### 📅 My Shifts")
    
    if "providers_df" not in st.session_state or st.session_state.providers_df.empty:
        st.warning("No providers loaded.")
        return
    
    # Provider selection
    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
    selected_provider = st.selectbox("Select Provider", options=providers, key="mobile_my_shifts_provider")
    
    if not selected_provider:
        return
    
    # Filter events for this provider
    my_events = []
    for event in st.session_state.get("events", []):
        if hasattr(event, 'extendedProps'):
            provider = event.extendedProps.get("provider", "")
        elif isinstance(event, dict) and 'extendedProps' in event:
            provider = event['extendedProps'].get("provider", "")
        else:
            continue
        
        if provider == selected_provider:
            my_events.append(event)
    
    if not my_events:
        st.info(f"No shifts found for {selected_provider}.")
        return
    
    # Display shifts in mobile-friendly format
    st.markdown(f"#### {selected_provider}'s Shifts")
    
    # Group by month
    shifts_by_month = {}
    for event in my_events:
        if hasattr(event, 'start'):
            event_date = event.start.date()
        elif isinstance(event, dict) and 'start' in event:
            event_date = datetime.fromisoformat(event['start']).date()
        else:
            continue
        
        month_key = event_date.strftime("%B %Y")
        if month_key not in shifts_by_month:
            shifts_by_month[month_key] = []
        shifts_by_month[month_key].append((event_date, event))
    
    # Display each month
    for month, events in shifts_by_month.items():
        with st.expander(f"📅 {month} ({len(events)} shifts)", expanded=True):
            # Sort events by date
            events.sort(key=lambda x: x[0])
            
            for event_date, event in events:
                # Get event details
                if hasattr(event, 'extendedProps'):
                    shift_type = event.extendedProps.get("shift_type", "")
                    shift_label = event.extendedProps.get("shift_label", "")
                elif isinstance(event, dict) and 'extendedProps' in event:
                    shift_type = event['extendedProps'].get("shift_type", "")
                    shift_label = event['extendedProps'].get("shift_label", "")
                else:
                    shift_type = ""
                    shift_label = ""
                
                # Create mobile-friendly shift card
                st.markdown(f"""
                <div style="
                    background: white;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                    border-left: 4px solid #1f77b4;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{event_date.strftime('%A, %B %d')}</strong><br>
                            <small style="color: #666;">{shift_label}</small>
                        </div>
                        <div style="
                            background: #1f77b4;
                            color: white;
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 12px;
                            font-weight: 500;
                        ">
                            {shift_type}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def mobile_notifications_view() -> None:
    """Mobile notifications view."""
    st.markdown("### 🔔 Notifications")
    
    # Sample notifications - in a real app, these would come from a database
    notifications = [
        {
            "id": 1,
            "type": "request_approved",
            "title": "Vacation Request Approved",
            "message": "Your vacation request for Dec 15-20 has been approved.",
            "timestamp": "2024-12-10T10:30:00",
            "read": False
        },
        {
            "id": 2,
            "type": "schedule_change",
            "title": "Schedule Update",
            "message": "Your shift on Dec 12 has been moved to Dec 13.",
            "timestamp": "2024-12-09T14:15:00",
            "read": True
        },
        {
            "id": 3,
            "type": "reminder",
            "title": "Shift Reminder",
            "message": "You have a night shift tomorrow (Dec 11).",
            "timestamp": "2024-12-10T08:00:00",
            "read": False
        }
    ]
    
    for notification in notifications:
        unread_style = "border-left: 4px solid #1f77b4;" if not notification["read"] else ""
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            {unread_style}
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <strong>{notification['title']}</strong><br>
                    <small style="color: #666;">{notification['message']}</small><br>
                    <small style="color: #999;">{notification['timestamp']}</small>
                </div>
                <div style="margin-left: 8px;">
                    {"" if notification["read"] else "🔵"}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Mark All as Read", use_container_width=True):
        st.success("All notifications marked as read!")

def mobile_navigation() -> str:
    """Mobile navigation component."""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("��", help="Home", use_container_width=True):
            return "home"
    
    with col2:
        if st.button("��", help="Calendar", use_container_width=True):
            return "calendar"
    
    with col3:
        if st.button("��", help="Requests", use_container_width=True):
            return "requests"
    
    with col4:
        if st.button("⚙️", help="Settings", use_container_width=True):
            return "settings"
    
    return "home"
