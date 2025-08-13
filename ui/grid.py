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
                    # Handle SEvent objects
                    if hasattr(e, 'start') and hasattr(e, 'extendedProps'):
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
    
    grid_data["Shift Type"] = color_labels
    
    # Add date columns
    for date_col in date_cols:
        grid_data[date_col] = []
        for idx, row in df.iterrows():
            grid_data[date_col].append(row[date_col] if row[date_col] else "")
    
    grid_df = pd.DataFrame(grid_data)
    
    # Calculate height to avoid vertical scroll
    height_px = min(2200, 110 + len(row_meta) * 38)
    
    # Create column configuration for the data editor
    col_config = {
        "Shift Type": st.column_config.TextColumn(
            "Shift Type",
            disabled=True,
            help="Shift type color indicator",
            width="large"
        )
    }
    
    # Add column configuration for each date
    for date_col in date_cols:
        # Determine which providers can be assigned to this column based on shift types
        shift_types_in_col = set()
        for meta in row_meta:
            if meta["row_label"] in grid_df.index:
                shift_types_in_col.add(meta["shift_key"])
        
        # Filter providers based on shift type
        available_providers = []
        for provider in providers:
            # For now, allow all providers for all shift types
            # This can be enhanced later with provider-specific restrictions
            available_providers.append(provider)
        
        col_config[date_col] = st.column_config.SelectboxColumn(
            date_col,
            options=provider_options,
            help=f"Assign provider to {date_col}",
            width="small"
        )
    
    # Add custom CSS for sticky first column and better styling
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
        
        /* Comprehensive sticky column CSS for Streamlit data editor */
        /* Target the data editor container */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
            max-width: 100% !important;
            position: relative !important;
        }
        
        /* Target the table element inside data editor */
        [data-testid="stDataFrame"] table {
            border-collapse: collapse !important;
            width: 100% !important;
        }
        
        /* Make the first column sticky */
        [data-testid="stDataFrame"] th:first-child,
        [data-testid="stDataFrame"] td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 200px !important;
            max-width: 250px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Ensure header stays on top */
        [data-testid="stDataFrame"] thead th:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1001 !important;
            background: white !important;
            min-width: 200px !important;
            max-width: 250px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Alternative selectors for different Streamlit versions */
        .stDataFrame th:first-child,
        .stDataFrame td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 200px !important;
            max-width: 250px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        .stDataFrame thead th:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1001 !important;
            background: white !important;
            min-width: 200px !important;
            max-width: 250px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Ensure proper cell sizing for other columns */
        [data-testid="stDataFrame"] td:not(:first-child) {
            min-width: 120px !important;
            max-width: 150px !important;
        }
        
        .stDataFrame td:not(:first-child) {
            min-width: 120px !important;
            max-width: 150px !important;
        }
        
        /* Additional selectors for data editor specific elements */
        [data-testid="stDataFrame"] [data-testid="stDataFrame"] th:first-child,
        [data-testid="stDataFrame"] [data-testid="stDataFrame"] td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 200px !important;
            max-width: 250px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Force horizontal scrolling */
        [data-testid="stDataFrame"] > div {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        
        .stDataFrame > div {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        
        /* Additional selectors for data editor table structure */
        [data-testid="stDataFrame"] table thead tr th:first-child,
        [data-testid="stDataFrame"] table tbody tr td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 250px !important;
            max-width: 300px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Target the actual table cells more specifically */
        [data-testid="stDataFrame"] table th:first-child,
        [data-testid="stDataFrame"] table td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 250px !important;
            max-width: 300px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Ensure the table container allows horizontal scrolling */
        [data-testid="stDataFrame"] > div > div {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        
        /* Target any element with role="grid" (data editor uses this) */
        [role="grid"] th:first-child,
        [role="grid"] td:first-child {
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: white !important;
            min-width: 250px !important;
            max-width: 300px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Force the container to be scrollable */
        [data-testid="stDataFrame"] {
            overflow-x: scroll !important;
            max-width: 100% !important;
        }
        
        /* Ensure the table doesn't wrap */
        [data-testid="stDataFrame"] table {
            white-space: nowrap !important;
            min-width: max-content !important;
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
        # Count filled slots vs total available slots
        total_slots = 0
        filled_slots = 0
        
        for date_col in date_cols:
            for idx, row in df.iterrows():
                total_slots += 1
                if row[date_col] and row[date_col] != "":
                    filled_slots += 1
        
        st.metric("Filled Slots", f"{filled_slots}/{total_slots}")
    
    with col4:
        # Calculate coverage based on filled slots
        coverage_percent = (filled_slots / total_slots * 100) if total_slots > 0 else 0
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
        
        # Show enhanced provider utilization
        st.markdown("#### 📊 Enhanced Provider Utilization")
        
        # Add warning for providers with insufficient shifts
        days_in_month = len(date_cols)
        min_required_shifts = 15 if days_in_month == 30 else 16
        
        provider_stats = {}
        
        for event in events:
            if hasattr(event, 'extendedProps'):
                provider = event.extendedProps.get("provider", "")
                shift_type = event.extendedProps.get("shift_type", "")
            elif isinstance(event, dict) and 'extendedProps' in event:
                provider = event['extendedProps'].get("provider", "")
                shift_type = event['extendedProps'].get("shift_type", "")
            else:
                continue
            
            if provider:
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        "total_shifts": 0,
                        "weekend_shifts": 0,
                        "night_shifts": 0,
                        "rounder_shifts": 0,
                        "admitting_shifts": 0,
                        "app_shifts": 0
                    }
                
                provider_stats[provider]["total_shifts"] += 1
                
                # Count weekend shifts
                if hasattr(event, 'start'):
                    event_date = event.start.date()
                elif isinstance(event, dict) and 'start' in event:
                    try:
                        event_date = datetime.fromisoformat(event['start']).date()
                    except (ValueError, TypeError):
                        continue
                else:
                    continue
                
                if event_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    provider_stats[provider]["weekend_shifts"] += 1
                
                # Count shift types
                if shift_type in ["N12", "NB"]:
                    provider_stats[provider]["night_shifts"] += 1
                elif shift_type == "R12":
                    provider_stats[provider]["rounder_shifts"] += 1
                elif shift_type in ["A12", "A10"]:
                    provider_stats[provider]["admitting_shifts"] += 1
                elif shift_type == "APP":
                    provider_stats[provider]["app_shifts"] += 1
        
        if provider_stats:
            # Create a DataFrame for enhanced provider utilization
            utilization_data = []
            for provider, stats in provider_stats.items():
                utilization_data.append({
                    "Provider": provider,
                    "Total Shifts": stats["total_shifts"],
                    "Weekend Shifts": stats["weekend_shifts"],
                    "Night Shifts": stats["night_shifts"],
                    "Rounder Shifts": stats["rounder_shifts"],
                    "Admitting Shifts": stats["admitting_shifts"],
                    "APP Shifts": stats["app_shifts"]
                })
            
            utilization_df = pd.DataFrame(utilization_data).sort_values("Total Shifts", ascending=False)
            
            # Add warning for providers with insufficient shifts
            insufficient_providers = []
            for _, row in utilization_df.iterrows():
                if row["Total Shifts"] < min_required_shifts:
                    insufficient_providers.append(f"{row['Provider']} ({row['Total Shifts']} shifts)")
            
            if insufficient_providers:
                st.warning(f"⚠️ **Providers with insufficient shifts (need {min_required_shifts}):** {', '.join(insufficient_providers)}")
            
            st.dataframe(
                utilization_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Provider": st.column_config.TextColumn("Provider", width="medium"),
                    "Total Shifts": st.column_config.NumberColumn("Total", width="small"),
                    "Weekend Shifts": st.column_config.NumberColumn("Weekend", width="small"),
                    "Night Shifts": st.column_config.NumberColumn("Night", width="small"),
                    "Rounder Shifts": st.column_config.NumberColumn("Rounder", width="small"),
                    "Admitting Shifts": st.column_config.NumberColumn("Admitting", width="small"),
                    "APP Shifts": st.column_config.NumberColumn("APP", width="small")
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
        if hasattr(event, 'start') and hasattr(event, 'extendedProps'):
            # It's an SEvent object
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
            if col == "Shift Type":
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
                        # It's an SEvent object
                        event.extendedProps["provider"] = new_provider
                        event.title = f"{new_provider} - {shift_key}"
                    elif isinstance(event, dict):
                        event['extendedProps']['provider'] = new_provider
                        event['title'] = f"{new_provider} - {shift_key}"
                    updated_events.append(event)
                else:
                    # Create new event as SEvent object
                    from models.data_models import SEvent
                    import uuid
                    
                    # Get shift config
                    shift_config = None
                    for shift_type in [
                        {"key": "R12", "label": "7am–7pm Rounder", "start": "07:00", "end": "19:00", "color": "#16a34a"},
                        {"key": "A12", "label": "7am–7pm Admitter", "start": "07:00", "end": "19:00", "color": "#f59e0b"},
                        {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
                        {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
                        {"key": "NB", "label": "Night Bridge", "start": "23:00", "end": "07:00", "color": "#06b6d4"},
                        {"key": "APP", "label": "APP Provider", "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
                    ]:
                        if shift_type["key"] == shift_key:
                            shift_config = shift_type
                            break
                    
                    if shift_config:
                        start_time = datetime.combine(event_date, datetime.strptime(shift_config["start"], "%H:%M").time())
                        end_time = datetime.combine(event_date, datetime.strptime(shift_config["end"], "%H:%M").time())
                        
                        # Handle overnight shifts
                        if shift_config["end"] < shift_config["start"]:
                            end_time += timedelta(days=1)
                        
                        new_event = SEvent(
                            id=str(uuid.uuid4()),
                            title=f"{new_provider} - {shift_key}",
                            start=start_time,
                            end=end_time,
                            backgroundColor=shift_config["color"],
                            extendedProps={
                                "provider": new_provider,
                                "shift_type": shift_key,
                                "shift_label": shift_config["label"]
                            }
                        )
                        updated_events.append(new_event)
            else:  # No provider assigned - remove event if it exists
                if key in existing_events:
                    # Don't add this event to updated_events (effectively removing it)
                    pass
    
    # Add events that weren't changed
    for event in original_events:
        if hasattr(event, 'start') and hasattr(event, 'extendedProps'):
            # It's an SEvent object
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
