# =============================================================================
# Grid View Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from typing import List, Dict, Any
import calendar

from models.data_models import SEvent
from models.constants import DEFAULT_SHIFT_TYPES

def create_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Create a grid view of the schedule with shift types as rows and dates as columns.
    Handles both SEvent objects and dictionaries.
    """
    # Get month days
    month_days = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        month_days.append(date(year, month, day))
    
    # Define the correct order of shift types for the grid
    shift_type_order = [
        {"key": "R12", "label": "7am–7pm Rounder", "color": "#16a34a"},
        {"key": "A12", "label": "7am–7pm Admitter", "color": "#f59e0b"},
        {"key": "A10", "label": "10am–10pm Admitter", "color": "#ef4444"},
        {"key": "N12", "label": "7pm–7am (Night)", "color": "#7c3aed"},
        {"key": "NB", "label": "Night Bridge", "color": "#06b6d4"},
        {"key": "APP", "label": "APP Provider", "color": "#8b5cf6"},
    ]
    
    # Create grid data with shift types as rows
    grid_data = []
    
    for shift_type in shift_type_order:
        shift_key = shift_type["key"]
        shift_label = shift_type["label"]
        
        row = {
            "Shift Type": shift_label,
            "Shift Key": shift_key,
            "Color": shift_type["color"]
        }
        
        # Add a column for each day of the month
        for day in month_days:
            day_key = day.strftime("%m/%d")
            day_events = []
            
            for e in events:
                # Handle both SEvent objects and dictionaries
                if hasattr(e, 'start'):
                    # It's an SEvent object
                    event_date = e.start.date()
                    event_shift_type = e.extendedProps.get("shift_type")
                    provider = e.extendedProps.get("provider", "")
                elif isinstance(e, dict) and 'start' in e:
                    # It's a dictionary with start field
                    try:
                        event_date = datetime.fromisoformat(e['start']).date()
                        event_shift_type = e.get('extendedProps', {}).get("shift_type")
                        provider = e.get('extendedProps', {}).get("provider", "")
                    except (ValueError, TypeError):
                        continue
                else:
                    # Unknown format, skip
                    continue
                
                if event_date == day and event_shift_type == shift_key:
                    day_events.append(provider)
            
            # Take only the first provider (one provider per cell)
            row[day_key] = day_events[0] if day_events else ""
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid with professional styling and color coding.
    Shows shift types as rows and dates as columns with editable dropdowns.
    """
    if not events:
        st.info("No schedule to display. Generate a schedule first.")
        return pd.DataFrame()
    
    df = create_schedule_grid(events, year, month)
    
    if df.empty:
        st.info("No events found for this month.")
        return df
    
    # Get the date columns (all columns except Shift Type, Shift Key, Color)
    date_cols = [col for col in df.columns if col not in ["Shift Type", "Shift Key", "Color"]]
    
    # Get available providers
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
    else:
        providers = []
    
    # Add "None" option for empty cells
    provider_options = ["None"] + providers
    
    st.markdown("### 📊 Schedule Grid View")
    st.markdown("**Shift Types:** 7am–7pm Rounder | 7am–7pm Admitter | 10am–10pm Admitter | 7pm–7am (Night) | Night Bridge | APP Provider")
    st.markdown("**Instructions:** Use the dropdowns below to assign providers to shifts. Changes will be applied when you click 'Apply Grid Changes to Calendar'.")
    
    # Add CSS for better grid styling
    st.markdown("""
    <style>
        .grid-container {
            overflow-x: auto;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }
        .stSelectbox > div > div {
            min-width: 120px;
        }
        .stButton > button {
            margin: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create editable grid with dropdowns
    st.markdown("#### Edit Schedule")
    
    # Wrap the grid in a container for better scrolling
    with st.container():
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Create a form for the grid
    with st.form("schedule_grid_form"):
        # Display shift type labels with color coding
        shift_colors = {
            "7am–7pm Rounder": "#16a34a",
            "7am–7pm Admitter": "#f59e0b", 
            "10am–10pm Admitter": "#ef4444",
            "7pm–7am (Night)": "#7c3aed",
            "Night Bridge": "#06b6d4",
            "APP Provider": "#8b5cf6"
        }
        
        # Create header row with dates - use wider columns
        header_cols = st.columns([3] + [2] * len(date_cols))
        with header_cols[0]:
            st.markdown("**Shift Type**")
        for i, date_col in enumerate(date_cols):
            with header_cols[i + 1]:
                st.markdown(f"**{date_col}**")
        
        # Create rows for each shift type
        updated_data = {}
        
        for idx, row in df.iterrows():
            shift_type = row["Shift Type"]
            shift_key = row["Shift Key"]
            
            # Create columns for this row - use wider columns
            cols = st.columns([3] + [2] * len(date_cols))
            
            # Shift type label with color
            with cols[0]:
                color = shift_colors.get(shift_type, "#ffffff")
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                    {shift_type}
                </div>
                """, unsafe_allow_html=True)
            
            # Provider dropdowns for each date
            for i, date_col in enumerate(date_cols):
                with cols[i + 1]:
                    current_provider = row[date_col] if row[date_col] else "None"
                    
                    # Create unique key for each dropdown
                    dropdown_key = f"grid_{shift_key}_{date_col}"
                    
                    # Create dropdown with better styling
                    selected_provider = st.selectbox(
                        "Provider",
                        options=provider_options,
                        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
                        key=dropdown_key,
                        label_visibility="collapsed"
                    )
                    
                    # Store the selection
                    if selected_provider != "None":
                        updated_data[f"{shift_key}_{date_col}"] = selected_provider
                    else:
                        updated_data[f"{shift_key}_{date_col}"] = ""
        
        # Submit button
        submitted = st.form_submit_button("🔄 Apply Grid Changes to Calendar", type="primary")
        
        if submitted:
            # Apply changes to events
            updated_events = apply_grid_changes_to_calendar(updated_data, events, year, month)
            st.session_state.events = updated_events
            
            # Auto-save the updated schedule
            from core.data_manager import save_schedule
            save_schedule(year, month, st.session_state.events)
            
            st.success("Grid changes applied to calendar and saved!")
            st.rerun()
    
    # Close the grid container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display read-only summary
    st.markdown("---")
    st.markdown("#### 📈 Schedule Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = len(events)
        st.metric("Total Events", total_events)
    
    with col2:
        unique_providers = set()
        for event in events:
            if hasattr(event, 'extendedProps'):
                provider = event.extendedProps.get("provider", "")
            elif isinstance(event, dict) and 'extendedProps' in event:
                provider = event['extendedProps'].get("provider", "")
            else:
                continue
            if provider:
                unique_providers.add(provider)
        st.metric("Providers Used", len(unique_providers))
    
    with col3:
        # Count days with events
        days_with_events = 0
        for date_col in date_cols:
            if any(df[date_col] != ""):
                days_with_events += 1
        st.metric("Days with Events", days_with_events)
    
    with col4:
        total_days = len(date_cols)
        coverage_percent = (days_with_events / total_days * 100) if total_days > 0 else 0
        st.metric("Coverage", f"{coverage_percent:.1f}%")
    
    # Provider statistics
    st.markdown("### 📊 Provider Statistics")
    
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
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
        
        # Show provider utilization
        st.markdown("#### Provider Utilization")
        provider_counts = {}
        
        for event in events:
            if hasattr(event, 'extendedProps'):
                provider = event.extendedProps.get("provider", "")
            elif isinstance(event, dict) and 'extendedProps' in event:
                provider = event['extendedProps'].get("provider", "")
            else:
                continue
            
            if provider:
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        if provider_counts:
            # Create a DataFrame for provider utilization
            utilization_df = pd.DataFrame([
                {"Provider": provider, "Shifts": count}
                for provider, count in provider_counts.items()
            ]).sort_values("Shifts", ascending=False)
            
            st.dataframe(
                utilization_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Provider": st.column_config.TextColumn("Provider", width="medium"),
                    "Shifts": st.column_config.NumberColumn("Shifts", width="small")
                }
            )
    
    return df

def apply_grid_changes_to_calendar(updated_data: Dict[str, str], original_events: List[Any], year: int, month: int) -> List[Any]:
    """
    Apply changes from grid to calendar events.
    """
    updated_events = []
    
    # Create a mapping of existing events by key
    existing_events = {}
    for event in original_events:
        if hasattr(event, 'start'):
            event_date = event.start.date()
            event_shift_type = event.extendedProps.get("shift_type")
            event_provider = event.extendedProps.get("provider", "")
        elif isinstance(event, dict) and 'start' in event:
            try:
                event_date = datetime.fromisoformat(event['start']).date()
                event_shift_type = event.get('extendedProps', {}).get("shift_type")
                event_provider = event.get('extendedProps', {}).get("provider", "")
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        key = f"{event_shift_type}_{event_date.strftime('%m/%d')}"
        existing_events[key] = event
    
    # Process grid changes
    for grid_key, new_provider in updated_data.items():
        shift_type, date_str = grid_key.split('_', 1)
        
        # Parse date
        try:
            month_str, day_str = date_str.split('/')
            event_date = date(year, int(month_str), int(day_str))
        except (ValueError, TypeError):
            continue
        
        key = f"{shift_type}_{date_str}"
        
        if new_provider:  # Provider assigned
            if key in existing_events:
                # Update existing event
                event = existing_events[key]
                if hasattr(event, 'extendedProps'):
                    event.extendedProps["provider"] = new_provider
                    event.title = f"{new_provider} - {shift_type}"
                elif isinstance(event, dict):
                    event['extendedProps']['provider'] = new_provider
                    event['title'] = f"{new_provider} - {shift_type}"
                updated_events.append(event)
            else:
                # Create new event
                new_event = {
                    "id": f"{shift_type}_{new_provider}_{event_date.isoformat()}",
                    "title": f"{new_provider} - {shift_type}",
                    "start": datetime.combine(event_date, datetime.min.time()).isoformat(),
                    "end": datetime.combine(event_date, datetime.min.time()).isoformat(),
                    "extendedProps": {
                        "provider": new_provider,
                        "shift_type": shift_type
                    }
                }
                updated_events.append(new_event)
        else:  # No provider assigned - remove event if it exists
            if key in existing_events:
                # Don't add this event to updated_events (effectively removing it)
                pass
    
    # Add events that weren't changed
    for event in original_events:
        if hasattr(event, 'start'):
            event_date = event.start.date()
            event_shift_type = event.extendedProps.get("shift_type")
        elif isinstance(event, dict) and 'start' in event:
            try:
                event_date = datetime.fromisoformat(event['start']).date()
                event_shift_type = event.get('extendedProps', {}).get("shift_type")
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        key = f"{event_shift_type}_{event_date.strftime('%m/%d')}"
        
        # Only add if not already processed
        if key not in updated_data:
            updated_events.append(event)
    
    return updated_events
