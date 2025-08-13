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
    Render the schedule grid using st.data_editor with proper column configuration.
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
    
    # Create a more compact grid for editing
    # Create row labels that combine shift type and slot
    row_labels = []
    row_meta = []
    
    for idx, row in df.iterrows():
        shift_type = row["Shift Type"]
        shift_key = row["Shift Key"]
        slot = row["Slot"]
        color = row["Color"]
        
        # Create row label
        row_label = f"{shift_key} — {shift_type} ({slot})"
        row_labels.append(row_label)
        
        # Store metadata
        row_meta.append({
            "row_label": row_label,
            "shift_key": shift_key,
            "shift_type": shift_type,
            "slot": slot,
            "color": color
        })
    
    # Create grid dataframe with proper structure
    grid_data = {}
    
    # Add color column first with colored text instead of hex codes
    color_labels = []
    for row in row_meta:
        shift_key = row["shift_key"]
        shift_type = row["shift_type"]
        
        # Create colored text labels based on shift type
        if shift_key == "R12":
            color_labels.append("🟢 Rounder")
        elif shift_key == "A12":
            color_labels.append("🟡 Admitter")
        elif shift_key == "A10":
            color_labels.append("🔴 Admitter")
        elif shift_key == "N12":
            color_labels.append("🟣 Night")
        elif shift_key == "NB":
            color_labels.append("🔵 Bridge")
        elif shift_key == "APP":
            color_labels.append("🟪 APP")
        else:
            color_labels.append(shift_type)
    
    grid_data["Color"] = color_labels
    
    # Add date columns
    for date_col in date_cols:
        grid_data[date_col] = []
        for idx, row in df.iterrows():
            grid_data[date_col].append(row[date_col] if row[date_col] else "")
    
    grid_df = pd.DataFrame(grid_data, index=row_labels)
    
    # Calculate height to avoid vertical scroll
    height_px = min(2200, 110 + len(row_meta) * 38)
    
    # Create column configuration for the data editor
    col_config = {
        "Color": st.column_config.TextColumn(
            "Shift Type",
            disabled=True,
            help="Shift type color indicator",
            width="small"
        )
    }
    
    # Add column configuration for each date
    for date_col in date_cols:
        # Determine which providers can be assigned to this column based on shift types
        shift_types_in_col = set()
        for meta in row_meta:
            if meta["row_label"] in grid_df.index:
                shift_types_in_col.add(meta["shift_key"])
        
        # Set options based on shift types in this column
        if "APP" in shift_types_in_col and len(shift_types_in_col) == 1:
            # If ONLY APP shifts are available, only APP providers can be assigned
            app_providers = ["None"] + [p for p in providers if p in ["JA", "DN", "KP", "AR"]]
            options = app_providers
            help_text = f"Assignments for {date_col} (APP providers only)"
        elif "APP" not in shift_types_in_col:
            # If NO APP shifts are available, only physician providers
            physician_providers = ["None"] + [p for p in providers if p not in ["JA", "DN", "KP", "AR"]]
            options = physician_providers
            help_text = f"Assignments for {date_col} (Physicians only)"
        else:
            # Mixed shift types - allow both provider types
            options = provider_options
            help_text = f"Assignments for {date_col} (All providers)"
        
        col_config[date_col] = st.column_config.SelectboxColumn(
            options=options,
            help=help_text,
            width="small"
        )
    
    # Add CSS for better styling
    st.markdown("""
    <style>
        .grid-container {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        
        .scroll-hint {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            text-align: center;
            color: #1565c0;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add scroll hint
    st.markdown("""
    <div class="scroll-hint">
        💡 <strong>Tip:</strong> Scroll horizontally to see all days. The shift type column stays fixed while you scroll through the dates.
    </div>
    """, unsafe_allow_html=True)
    
    # Wrap in container
    with st.container():
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)
        
        # Use st.data_editor for the grid
        edited_grid = st.data_editor(
            grid_df,
            num_rows="fixed",
            use_container_width=True,
            height=height_px,
            column_config=col_config,
            key="schedule_grid_editor",
            hide_index=False
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle grid changes
    if edited_grid is not None and not edited_grid.equals(grid_df):
        # Apply changes to events
        updated_events = apply_grid_changes_to_calendar(edited_grid, events, year, month, row_meta)
        st.session_state.events = updated_events
        
        # Auto-save the updated schedule
        from core.data_manager import save_schedule
        save_schedule(year, month, st.session_state.events)
        
        st.success("✅ Grid changes applied to calendar and saved!")
        st.rerun()
    
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

def apply_grid_changes_to_calendar(edited_grid: pd.DataFrame, original_events: List[Any], year: int, month: int, row_meta: List[Dict]) -> List[Any]:
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
    for row_label in edited_grid.index:
        # Find the corresponding row metadata
        row_info = None
        for meta in row_meta:
            if meta["row_label"] == row_label:
                row_info = meta
                break
        
        if not row_info:
            continue
        
        shift_key = row_info["shift_key"]
        
        # Process each date column
        for col in edited_grid.columns:
            if col == "Color":
                continue
            
            new_provider = edited_grid.at[row_label, col]
            
            # Parse date
            try:
                month_str, day_str = col.split('/')
                event_date = date(year, int(month_str), int(day_str))
            except (ValueError, TypeError):
                continue
            
            key = f"{shift_key}_{col}"
            
            if new_provider and new_provider != "None":  # Provider assigned
                if key in existing_events:
                    # Update existing event
                    event = existing_events[key]
                    if hasattr(event, 'extendedProps'):
                        event.extendedProps["provider"] = new_provider
                        event.title = f"{new_provider} - {shift_key}"
                    elif isinstance(event, dict):
                        event['extendedProps']['provider'] = new_provider
                        event['title'] = f"{new_provider} - {shift_key}"
                    updated_events.append(event)
                else:
                    # Create new event
                    new_event = {
                        "id": f"{shift_key}_{new_provider}_{event_date.isoformat()}",
                        "title": f"{new_provider} - {shift_key}",
                        "start": datetime.combine(event_date, datetime.min.time()).isoformat(),
                        "end": datetime.combine(event_date, datetime.min.time()).isoformat(),
                        "extendedProps": {
                            "provider": new_provider,
                            "shift_type": shift_key
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
        
        # Only add if not in the target month or not processed
        if event_date.year != year or event_date.month != month:
            updated_events.append(event)
    
    return updated_events
