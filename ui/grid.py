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
    Create a grid view of the schedule with multiple rows per shift type based on capacity.
    Handles both SEvent objects and dictionaries.
    """
    # Get month days
    month_days = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        month_days.append(date(year, month, day))
    
    # Define the correct order of shift types for the grid with capacities
    shift_type_order = [
        {"key": "R12", "label": "7am–7pm Rounder", "color": "#16a34a", "capacity": 13},
        {"key": "A12", "label": "7am–7pm Admitter", "color": "#f59e0b", "capacity": 1},
        {"key": "A10", "label": "10am–10pm Admitter", "color": "#ef4444", "capacity": 2},
        {"key": "N12", "label": "7pm–7am (Night)", "color": "#7c3aed", "capacity": 4},
        {"key": "NB", "label": "Night Bridge", "color": "#06b6d4", "capacity": 1},
        {"key": "APP", "label": "APP Provider", "color": "#8b5cf6", "capacity": 2},
    ]
    
    # Create grid data with multiple rows per shift type
    grid_data = []
    
    for shift_type in shift_type_order:
        shift_key = shift_type["key"]
        shift_label = shift_type["label"]
        capacity = shift_type["capacity"]
        color = shift_type["color"]
        
        # Create multiple rows for this shift type based on capacity
        for slot_num in range(1, capacity + 1):
            row = {
                "Shift Type": shift_label,
                "Slot": f"Slot {slot_num}",
                "Shift Key": shift_key,
                "Color": color,
                "Capacity": capacity
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
                
                # Assign provider to this slot if available
                if slot_num <= len(day_events):
                    row[day_key] = day_events[slot_num - 1]
                else:
                    row[day_key] = ""
            
            grid_data.append(row)
    
    return pd.DataFrame(grid_data)

def render_schedule_grid(events: List[Any], year: int, month: int) -> pd.DataFrame:
    """
    Render the schedule grid with professional styling and color coding.
    Shows multiple rows per shift type based on capacity with editable dropdowns.
    """
    if not events:
        st.info("No schedule to display. Generate a schedule first.")
        return pd.DataFrame()
    
    df = create_schedule_grid(events, year, month)
    
    if df.empty:
        st.info("No events found for this month.")
        return df
    
    # Get the date columns (all columns except Shift Type, Slot, Shift Key, Color, Capacity)
    date_cols = [col for col in df.columns if col not in ["Shift Type", "Slot", "Shift Key", "Color", "Capacity"]]
    
    # Get available providers
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
    else:
        providers = []
    
    # Add "None" option for empty cells
    provider_options = ["None"] + providers
    
    st.markdown("### 📊 Schedule Grid View")
    st.markdown("**Shift Types:** 7am–7pm Rounder (13 slots) | 7am–7pm Admitter (1 slot) | 10am–10pm Admitter (2 slots) | 7pm–7am Night (4 slots) | Night Bridge (1 slot) | APP Provider (2 slots)")
    st.markdown("**Instructions:** Each row represents a slot for that shift type. Use the dropdowns to assign providers to specific slots.")
    
    # Add CSS for better grid styling
    st.markdown("""
    <style>
        .grid-container {
            overflow-x: auto;
            padding: 20px;
            border: 3px solid #e0e0e0;
            border-radius: 12px;
            background: #f8f9fa;
            margin: 15px 0;
            max-width: 100%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .grid-table {
            border-collapse: separate;
            border-spacing: 3px;
            width: max-content;
            min-width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .grid-header {
            background: linear-gradient(135deg, #1f77b4, #1565c0);
            color: white;
            font-weight: bold;
            padding: 15px 10px;
            text-align: center;
            border: 1px solid #0d47a1;
            min-width: 120px;
            position: sticky;
            top: 0;
            z-index: 10;
            font-size: 14px;
        }
        
        .grid-header:first-child {
            position: sticky;
            left: 0;
            z-index: 20;
            min-width: 180px;
            background: linear-gradient(135deg, #1565c0, #0d47a1);
        }
        
        .grid-header:nth-child(2) {
            position: sticky;
            left: 180px;
            z-index: 20;
            min-width: 100px;
            background: linear-gradient(135deg, #1565c0, #0d47a1);
        }
        
        .grid-cell {
            background: white;
            border: 2px solid #dee2e6;
            padding: 12px 8px;
            min-width: 120px;
            min-height: 60px;
            text-align: center;
            vertical-align: middle;
            font-size: 13px;
        }
        
        .grid-cell:first-child {
            position: sticky;
            left: 0;
            z-index: 15;
            background: #f8f9fa;
            border-right: 3px solid #1f77b4;
            font-weight: bold;
            min-width: 180px;
            text-align: left;
        }
        
        .grid-cell:nth-child(2) {
            position: sticky;
            left: 180px;
            z-index: 15;
            background: #f8f9fa;
            border-right: 3px solid #1f77b4;
            font-weight: bold;
            min-width: 100px;
            text-align: center;
        }
        
        .stSelectbox > div > div {
            min-width: 110px;
            min-height: 45px;
        }
        
        .stSelectbox select {
            font-size: 13px;
            padding: 8px;
        }
        
        .shift-type-label {
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            text-align: center;
            font-weight: bold;
            font-size: 13px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: inline-block;
            width: 100%;
        }
        
        .slot-label {
            background: #6c757d;
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .scroll-hint {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border: 2px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
            color: #1565c0;
            font-weight: bold;
        }
        
        .capacity-info {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create editable grid with dropdowns
    st.markdown("#### Edit Schedule")
    
    # Add scroll hint
    st.markdown("""
    <div class="scroll-hint">
        💡 <strong>Tip:</strong> Scroll horizontally to see all days. The shift type and slot columns stay fixed while you scroll through the dates.
    </div>
    """, unsafe_allow_html=True)
    
    # Wrap the grid in a container for better scrolling
    with st.container():
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Create a form for the grid
    with st.form("schedule_grid_form"):
        # Create the grid using HTML table for better control
        grid_html = """
        <table class="grid-table">
            <thead>
                <tr>
                    <th class="grid-header">Shift Type</th>
                    <th class="grid-header">Slot</th>
        """
        
        # Add date headers
        for date_col in date_cols:
            grid_html += f'<th class="grid-header">{date_col}</th>'
        
        grid_html += """
                </tr>
            </thead>
            <tbody>
        """
        
        # Create rows for each shift type slot
        updated_data = {}
        current_shift_type = None
        
        for idx, row in df.iterrows():
            shift_type = row["Shift Type"]
            shift_key = row["Shift Key"]
            slot = row["Slot"]
            color = row["Color"]
            
            # Add a separator row when shift type changes
            if current_shift_type != shift_type:
                if current_shift_type is not None:
                    grid_html += '<tr><td colspan="' + str(len(date_cols) + 2) + '" style="height: 10px; background: #f8f9fa;"></td></tr>'
                current_shift_type = shift_type
            
            # Start row
            grid_html += f"""
                <tr>
                    <td class="grid-cell">
                        <div class="shift-type-label" style="background: {color};">
                            {shift_type}
                        </div>
                    </td>
                    <td class="grid-cell">
                        <div class="slot-label">
                            {slot}
                        </div>
                    </td>
            """
            
            # Add cells for each date
            for date_col in date_cols:
                current_provider = row[date_col] if row[date_col] else "None"
                dropdown_key = f"grid_{shift_key}_{slot.replace(' ', '_')}_{date_col}"
                
                grid_html += f'<td class="grid-cell">'
                grid_html += f'<div style="min-height: 60px; display: flex; align-items: center; justify-content: center;">'
                
                # We'll add the dropdown here, but need to handle it with Streamlit
                grid_html += f'<div id="{dropdown_key}_container"></div>'
                grid_html += '</div></td>'
                
                # Store the selection for later processing
                if current_provider != "None":
                    updated_data[f"{shift_key}_{slot.replace(' ', '_')}_{date_col}"] = current_provider
                else:
                    updated_data[f"{shift_key}_{slot.replace(' ', '_')}_{date_col}"] = ""
            
            grid_html += "</tr>"
        
        grid_html += """
            </tbody>
        </table>
        """
        
        # Display the HTML table
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Now add the Streamlit dropdowns in a more organized way
        st.markdown("### Provider Assignments")
        st.markdown("Use the dropdowns below to assign providers to specific slots:")
        
        # Group by shift type for better organization
        current_shift_type = None
        
        for idx, row in df.iterrows():
            shift_type = row["Shift Type"]
            shift_key = row["Shift Key"]
            slot = row["Slot"]
            
            # Add shift type header when it changes
            if current_shift_type != shift_type:
                if current_shift_type is not None:
                    st.markdown("---")
                current_shift_type = shift_type
                
                # Get capacity for this shift type
                capacity = row["Capacity"]
                st.markdown(f"**{shift_type}** ({capacity} slots)")
            
            # Create columns for this slot's dates
            cols = st.columns(len(date_cols))
            
            for i, date_col in enumerate(date_cols):
                with cols[i]:
                    current_provider = row[date_col] if row[date_col] else "None"
                    dropdown_key = f"grid_{shift_key}_{slot.replace(' ', '_')}_{date_col}"
                    
                    selected_provider = st.selectbox(
                        f"{slot} - {date_col}",
                        options=provider_options,
                        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
                        key=dropdown_key,
                        label_visibility="collapsed"
                    )
                    
                    # Store the selection
                    if selected_provider != "None":
                        updated_data[f"{shift_key}_{slot.replace(' ', '_')}_{date_col}"] = selected_provider
                    else:
                        updated_data[f"{shift_key}_{slot.replace(' ', '_')}_{date_col}"] = ""
        
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
        # Parse the grid key: shift_type_slot_date
        parts = grid_key.split('_', 2)
        if len(parts) != 3:
            continue
            
        shift_type, slot, date_str = parts
        
        # Parse date
        try:
            month_str, day_str = date_str.split('/')
            event_date = date(year, int(month_str), int(day_str))
        except (ValueError, TypeError):
            continue
        
        # For now, we'll use the original key format (without slot) for compatibility
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
