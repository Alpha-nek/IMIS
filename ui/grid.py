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
<<<<<<< HEAD
    Create a grid view of the schedule. Handles both SEvent objects and dictionaries.
=======
    Create a grid view of the schedule.
    Handles both SEvent objects and dictionaries.
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
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
<<<<<<< HEAD
                event_date = e.start.date()
            elif isinstance(e, dict) and 'start' in e:
                event_date = datetime.fromisoformat(e['start']).date()
            else:
=======
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
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
                continue
            
            if event_date == day:
                day_events.append(e)
        
        row = {
            "Date": day.strftime("%Y-%m-%d"), 
            "Day": day.strftime("%A"),
            "Day_Short": day.strftime("%a")
        }
        
        # Group events by shift type
<<<<<<< HEAD
        for shift_key in shift_keys:
=======
        for shift_type in shift_types:
            shift_key = shift_type["key"]
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
            shift_events = []
            for e in day_events:
                # Handle both SEvent objects and dictionaries
                if hasattr(e, 'extendedProps'):
<<<<<<< HEAD
                    event_shift_type = e.extendedProps.get("shift_type")
                elif isinstance(e, dict) and 'extendedProps' in e:
=======
                    # It's an SEvent object
                    event_shift_type = e.extendedProps.get("shift_type")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    # It's a dictionary
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
                    event_shift_type = e['extendedProps'].get("shift_type")
                else:
                    continue
                
                if event_shift_type == shift_key:
                    shift_events.append(e)
            
<<<<<<< HEAD
            providers = []
            for e in shift_events:
                # Handle both SEvent objects and dictionaries
                if hasattr(e, 'extendedProps'):
                    provider = e.extendedProps.get("provider", "")
                elif isinstance(e, dict) and 'extendedProps' in e:
=======
            # Extract provider names
            providers = []
            for e in shift_events:
                if hasattr(e, 'extendedProps'):
                    # It's an SEvent object
                    provider = e.extendedProps.get("provider", "")
                elif isinstance(e, dict) and 'extendedProps' in e:
                    # It's a dictionary
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
                    provider = e['extendedProps'].get("provider", "")
                else:
                    continue
                
                if provider:
                    providers.append(provider)
            
<<<<<<< HEAD
            row[shift_key] = ", ".join(providers) if providers else ""
=======
            # Use shift label instead of key for better readability
            column_name = shift_type["label"]
            row[column_name] = ", ".join(providers) if providers else ""
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
        
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
<<<<<<< HEAD
    Render the schedule grid with professional styling. Handles both SEvent objects and dictionaries.
=======
    Render the schedule grid with professional styling.
    Handles both SEvent objects and dictionaries.
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
    """
    if not events:
        st.info("No schedule to display. Generate a schedule first.")
        return pd.DataFrame()
    
    df = create_schedule_grid(events, year, month)
    
<<<<<<< HEAD
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
=======
    if df.empty:
        st.info("No events found for this month.")
        return df
    
    # Reorder columns for better display
    date_cols = ["Date", "Day", "Day_Short"]
    shift_cols = [col for col in df.columns if col not in date_cols]
    
    # Create a styled dataframe
    st.markdown("### �� Schedule Grid View")
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
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
    
    return df

def apply_grid_changes_to_calendar(grid_df: pd.DataFrame, original_events: List[Any]) -> List[Any]:
    """
<<<<<<< HEAD
    Apply changes from grid to calendar events. Handles both SEvent objects and dictionaries.
=======
    Apply changes from grid to calendar events.
    Handles both SEvent objects and dictionaries.
>>>>>>> d7a7c3502f1bb718aa8e4a31a3a568d6e1d9c7fe
    """
    # This is a simplified version - you'll need to implement the full logic
    # based on your specific grid editing requirements
    
    updated_events = original_events.copy()
    
    # Process grid changes here
    # This would involve comparing the grid data with the original events
    # and updating the events accordingly
    
    return updated_events
