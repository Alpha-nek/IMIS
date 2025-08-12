# =============================================================================
# Grid View Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Any
import calendar

from models.data_models import SEvent
from models.constants import DEFAULT_SHIFT_TYPES

def create_schedule_grid(events: List[SEvent], year: int, month: int) -> pd.DataFrame:
    """
    Create a grid view of the schedule.
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
        day_events = [e for e in events if e.start.date() == day]
        
        row = {"Date": day.strftime("%Y-%m-%d"), "Day": day.strftime("%A")}
        
        # Group events by shift type
        for shift_key in shift_keys:
            shift_events = [e for e in day_events if e.extendedProps.get("shift_type") == shift_key]
            providers = [e.extendedProps.get("provider", "") for e in shift_events]
            row[shift_key] = ", ".join(providers) if providers else ""
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[SEvent], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid.
    """
    df = create_schedule_grid(events, year, month)
    
    # Display grid
    st.dataframe(df, use_container_width=True)
    
    return df

def apply_grid_changes_to_calendar(grid_df: pd.DataFrame, original_events: List[SEvent]) -> List[SEvent]:
    """
    Apply changes from grid to calendar events.
    """
    # This is a simplified version - you'll need to implement the full logic
    # based on your specific grid editing requirements
    
    updated_events = original_events.copy()
    
    # Process grid changes here
    # This would involve comparing the grid data with the original events
    # and updating the events accordingly
    
    return updated_events#initial
