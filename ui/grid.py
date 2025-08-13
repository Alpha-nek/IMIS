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
        {"key": "R12", "label": "7am Rounders", "color": "#16a34a"},
        {"key": "A12", "label": "7am Admitter", "color": "#f59e0b"},
        {"key": "A10", "label": "10am Admitter", "color": "#ef4444"},
        {"key": "N12", "label": "Night Shift", "color": "#7c3aed"},
        {"key": "NB", "label": "Bridge", "color": "#06b6d4"},
        {"key": "APP", "label": "APP", "color": "#8b5cf6"},
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
            
            # Join multiple providers with commas if there are multiple
            row[day_key] = ", ".join(day_events) if day_events else ""
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid with professional styling and color coding.
    Shows shift types as rows and dates as columns.
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
    
    # Create column configuration with color coding
    column_config = {
        "Shift Type": st.column_config.TextColumn(
            "Shift Type", 
            width="medium",
            help="Type of shift"
        )
    }
    
    # Add date columns
    for date_col in date_cols:
        column_config[date_col] = st.column_config.TextColumn(
            date_col, 
            width="small",
            help=f"Providers assigned on {date_col}"
        )
    
    st.markdown("### 📊 Schedule Grid View")
    st.markdown("**Shift Types:** 7am Rounders | 7am Admitter | 10am Admitter | Night Shift | Bridge | APP")
    
    # Display the grid with styling
    display_df = df[["Shift Type"] + date_cols]
    
    # Apply color coding to the dataframe
    def color_shift_types(val):
        if val in ["7am Rounders", "7am Admitter", "10am Admitter", "Night Shift", "Bridge", "APP"]:
            colors = {
                "7am Rounders": "#16a34a",
                "7am Admitter": "#f59e0b", 
                "10am Admitter": "#ef4444",
                "Night Shift": "#7c3aed",
                "Bridge": "#06b6d4",
                "APP": "#8b5cf6"
            }
            return f'background-color: {colors.get(val, "#ffffff")}; color: white; font-weight: bold;'
        return ''
    
    # Apply styling
    styled_df = display_df.style.applymap(color_shift_types, subset=['Shift Type'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
    
    # Add summary statistics
    st.markdown("---")
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
    st.markdown("### 📈 Provider Statistics")
    
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

def apply_grid_changes_to_calendar(grid_df: pd.DataFrame, original_events: List[Any]) -> List[Any]:
    """
    Apply changes from grid to calendar events.
    Handles both SEvent objects and dictionaries.
    """
    # This is a simplified version - you'll need to implement the full logic
    # based on your specific grid editing requirements
    
    updated_events = original_events.copy()
    
    # Process grid changes here
    # This would involve comparing the grid data with the original events
    # and updating the events accordingly
    
    return updated_events
