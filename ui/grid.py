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
    Create a grid view of the schedule. Handles both SEvent objects and dictionaries.
    """
    # Get month days
    month_days = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        month_days.append(date(year, month, day))
    
    # Get shift types
    shift_types = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES)
    shift_keys = [s["key"] for s in shift_types]
    
    # Create grid data
    grid_data = []
    for day in month_days:
        day_events = []
        for e in events:
            # Handle both SEvent objects and dictionaries
            if hasattr(e, 'start'):
                event_date = e.start.date()
            elif isinstance(e, dict) and 'start' in e:
                event_date = datetime.fromisoformat(e['start']).date()
            else:
                continue
            
            if event_date == day:
                day_events.append(e)
        
        row = {"Date": day.strftime("%Y-%m-%d"), "Day": day.strftime("%A")}
        
        # Group events by shift type
        for shift_key in shift_keys:
            shift_events = []
            for e in day_events:
                # Handle both SEvent objects and dictionaries
                if hasattr(e, 'extendedProps'):
                    event_shift_type = e.extendedProps.get("shift_type")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    event_shift_type = e['extendedProps'].get("shift_type")
                else:
                    continue
                
                if event_shift_type == shift_key:
                    shift_events.append(e)
            
            providers = []
            for e in shift_events:
                # Handle both SEvent objects and dictionaries
                if hasattr(e, 'extendedProps'):
                    provider = e.extendedProps.get("provider", "")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    provider = e['extendedProps'].get("provider", "")
                else:
                    continue
                
                if provider:
                    providers.append(provider)
            
            row[shift_key] = ", ".join(providers) if providers else ""
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid with professional styling. Handles both SEvent objects and dictionaries.
    """
    df = create_schedule_grid(events, year, month)
    
    # Add summary statistics
    if not df.empty:
        total_events = sum(len([e for e in events if hasattr(e, 'start') or (isinstance(e, dict) and 'start' in e)]) for _, row in df.iterrows() if any(row[2:]))  # Skip Date and Day columns
        providers_used = set()
        for e in events:
            if hasattr(e, 'extendedProps'):
                provider = e.extendedProps.get("provider", "")
            elif isinstance(e, dict) and 'extendedProps' in e:
                provider = e['extendedProps'].get("provider", "")
            else:
                continue
            if provider:
                providers_used.add(provider)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", total_events)
        with col2:
            st.metric("Providers Used", len(providers_used))
        with col3:
            coverage = len([row for _, row in df.iterrows() if any(row[2:])])  # Days with events
            st.metric("Days Covered", f"{coverage}/{len(df)}")
    
    # Display grid with column configuration
    column_config = {
        "Date": st.column_config.DateColumn("Date", format="MM/DD"),
        "Day": st.column_config.TextColumn("Day", width="medium")
    }
    
    # Add shift type columns
    shift_types = st.session_state.get("shift_types", DEFAULT_SHIFT_TYPES)
    for shift_type in shift_types:
        column_config[shift_type["key"]] = st.column_config.TextColumn(
            shift_type["label"], 
            width="medium",
            help=f"{shift_type['start_time']} - {shift_type['end_time']}"
        )
    
    st.dataframe(
        df, 
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )
    
    return df

def apply_grid_changes_to_calendar(grid_df: pd.DataFrame, original_events: List[Any]) -> List[Any]:
    """
    Apply changes from grid to calendar events. Handles both SEvent objects and dictionaries.
    """
    # This is a simplified version - you'll need to implement the full logic
    # based on your specific grid editing requirements
    
    updated_events = original_events.copy()
    
    # Process grid changes here
    # This would involve comparing the grid data with the original events
    # and updating the events accordingly
    
    return updated_events
