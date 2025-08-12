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
    Create a grid view of the schedule.
    Handles both SEvent objects and dictionaries.
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
                # It's an SEvent object
                event_date = e.start.date()
            elif isinstance(e, dict) and 'start' in e:
                # It's a dictionary with start field
                try:
                    event_date = datetime.fromisoformat(e['start']).date()
                except (ValueError, TypeError):
                    continue
            else:
                # Unknown format, skip
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
                    # It's an SEvent object
                    event_shift_type = e.extendedProps.get("shift_type")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    # It's a dictionary
                    event_shift_type = e['extendedProps'].get("shift_type")
                else:
                    continue
                
                if event_shift_type == shift_key:
                    shift_events.append(e)
            
            # Extract provider names
            providers = []
            for e in shift_events:
                if hasattr(e, 'extendedProps'):
                    # It's an SEvent object
                    provider = e.extendedProps.get("provider", "")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    # It's a dictionary
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
    Render the schedule grid.
    Handles both SEvent objects and dictionaries.
    """
    df = create_schedule_grid(events, year, month)
    
    # Display grid
    st.dataframe(df, use_container_width=True)
    
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
