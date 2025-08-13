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
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: #f8f9fa;
            margin: 10px 0;
            max-width: 100%;
        }
        
        .grid-table {
            border-collapse: separate;
            border-spacing: 2px;
            width: max-content;
            min-width: 100%;
        }
        
        .grid-header {
            background: #1f77b4;
            color: white;
            font-weight: bold;
            padding: 12px 8px;
            text-align: center;
            border: 1px solid #1565c0;
            min-width: 150px;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .grid-header:first-child {
            position: sticky;
            left: 0;
            z-index: 20;
            min-width: 200px;
            background: #1565c0;
        }
        
        .grid-cell {
            background: white;
            border: 2px solid #dee2e6;
            padding: 8px;
            min-width: 150px;
            min-height: 50px;
            text-align: center;
            vertical-align: middle;
        }
        
        .grid-cell:first-child {
            position: sticky;
            left: 0;
            z-index: 15;
            background: #f8f9fa;
            border-right: 3px solid #1f77b4;
            font-weight: bold;
            min-width: 200px;
        }
        
        .stSelectbox > div > div {
            min-width: 140px;
            min-height: 40px;
        }
        
        .stSelectbox select {
            font-size: 14px;
            padding: 8px;
        }
        
        .shift-type-label {
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .scroll-hint {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            text-align: center;
            color: #1565c0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create editable grid with dropdowns
    st.markdown("#### Edit Schedule")
    
    # Add scroll hint
    st.markdown("""
    <div class="scroll-hint">
        💡 <strong>Tip:</strong> Scroll horizontally to see all days. The shift type column stays fixed while you scroll through the dates.
    </div>
    """, unsafe_allow_html=True)
    
    # Wrap the grid in a container for better scrolling
    with st.container():
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Create a form for the grid
    with st.form("schedule_grid_form"):
        # Define shift colors for the visual grid
        shift_colors = {
            "7am–7pm Rounder": "#16a34a",
            "7am–7pm Admitter": "#f59e0b", 
            "10am–10pm Admitter": "#ef4444",
            "7pm–7am (Night)": "#7c3aed",
            "Night Bridge": "#06b6d4",
            "APP Provider": "#8b5cf6"
        }
        
        # Create the grid using HTML table for better control
        grid_html = """
        <table class="grid-table">
            <thead>
                <tr>
                    <th class="grid-header">Shift Type</th>
        """
        
        # Add date headers
        for date_col in date_cols:
            grid_html += f'<th class="grid-header">{date_col}</th>'
        
        grid_html += """
                </tr>
            </thead>
            <tbody>
        """
        
        # Create rows for each shift type
        updated_data = {}
        
        for idx, row in df.iterrows():
            shift_type = row["Shift Type"]
            shift_key = row["Shift Key"]
            
            # Get color for this shift type
            color = shift_colors.get(shift_type, "#ffffff")
            
            # Start row
            grid_html += f"""
                <tr>
                    <td class="grid-cell">
                        <div class="shift-type-label" style="background: {color};">
                            {shift_type}
                        </div>
                    </td>
            """
            
            # Add cells for each date
            for date_col in date_cols:
                current_provider = row[date_col] if row[date_col] else "None"
                dropdown_key = f"grid_{shift_key}_{date_col}"
                
                grid_html += f'<td class="grid-cell">'
                grid_html += f'<div style="min-height: 50px; display: flex; align-items: center; justify-content: center;">'
                
                # We'll add the dropdown here, but need to handle it with Streamlit
                grid_html += f'<div id="{dropdown_key}_container"></div>'
                grid_html += '</div></td>'
                
                # Store the selection for later processing
                if current_provider != "None":
                    updated_data[f"{shift_key}_{date_col}"] = current_provider
                else:
                    updated_data[f"{shift_key}_{date_col}"] = ""
            
            grid_html += "</tr>"
        
        grid_html += """
            </tbody>
        </table>
        """
        
        # Display the HTML table
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Now add the Streamlit dropdowns in a more organized way
        st.markdown("### Provider Assignments")
        st.markdown("Use the dropdowns below to assign providers to shifts:")
        
        # Create a more compact layout for the dropdowns
        for idx, row in df.iterrows():
            shift_type = row["Shift Type"]
            shift_key = row["Shift Key"]
            
            st.markdown(f"**{shift_type}**")
            
            # Create columns for this shift type's dates
            cols = st.columns(len(date_cols))
            
            for i, date_col in enumerate(date_cols):
                with cols[i]:
                    current_provider = row[date_col] if row[date_col] else "None"
                    dropdown_key = f"grid_{shift_key}_{date_col}"
                    
                    selected_provider = st.selectbox(
                        f"{date_col}",
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
            
            st.markdown("---")
        
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
