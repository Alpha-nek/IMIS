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
        
        row = {
            "Date": day.strftime("%Y-%m-%d"), 
            "Day": day.strftime("%A"),
            "Day_Short": day.strftime("%a")
        }
        
        # Group events by shift type
        for shift_type in shift_types:
            shift_key = shift_type["key"]
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
            
            # Use shift label instead of key for better readability
            column_name = shift_type["label"]
            row[column_name] = ", ".join(providers) if providers else ""
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid with professional styling.
    Handles both SEvent objects and dictionaries.
    """
    if not events:
        st.info("No schedule to display. Generate a schedule first.")
        return pd.DataFrame()
    
    df = create_schedule_grid(events, year, month)
    
    if df.empty:
        st.info("No events found for this month.")
        return df
    
    # Reorder columns for better display
    date_cols = ["Date", "Day", "Day_Short"]
    shift_cols = [col for col in df.columns if col not in date_cols]
    
    # Create a styled dataframe
    st.markdown("### 📊 Schedule Grid View")
    st.markdown("Edit assignments directly in the grid below")
    
    # Display the grid with styling
    st.dataframe(
        df[date_cols + shift_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MM/DD/YYYY"),
            "Day": st.column_config.TextColumn("Day", width="medium"),
            "Day_Short": st.column_config.TextColumn("Day", width="small"),
        }
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
        days_with_events = len([row for _, row in df.iterrows() if any(row[col] != "" for col in shift_cols)])
        st.metric("Days with Events", days_with_events)
    
    with col4:
        total_days = len(df)
        coverage_percent = (days_with_events / total_days * 100) if total_days > 0 else 0
        st.metric("Coverage", f"{coverage_percent:.1f}%")
    
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
