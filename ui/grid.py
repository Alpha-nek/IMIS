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
    
    # Build shift types for the grid dynamically from settings with capacity
    shift_type_order = []
    try:
        session_shift_types = st.session_state.get("shift_types", [])
        capacity_map = st.session_state.get("shift_capacity", {})
        for stype in session_shift_types:
            key = stype.get("key")
            if not key:
                continue
            shift_type_order.append({
                "key": key,
                "label": stype.get("label", key),
                "color": stype.get("color", "#777777"),
                "capacity": int(capacity_map.get(key, 1))
            })
        # Fallback to defaults if settings are missing
        if not shift_type_order:
            shift_type_order = [
                {"key": "R12", "label": "7am–7pm Rounder", "color": "#16a34a", "capacity": 13},
                {"key": "A12", "label": "7am–7pm Admitter", "color": "#f59e0b", "capacity": 1},
                {"key": "A10", "label": "10am–10pm Admitter", "color": "#ef4444", "capacity": 2},
                {"key": "N12", "label": "7pm–7am (Night)", "color": "#7c3aed", "capacity": 4},
                {"key": "NB", "label": "Night Bridge", "color": "#06b6d4", "capacity": 1},
                {"key": "APP", "label": "APP Provider", "color": "#8b5cf6", "capacity": 2},
            ]
    except Exception:
        pass
    
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
        
        # Create colored text labels based on shift type with slot number
        slot_num = row["slot"].replace("Slot ", "")
        if shift_key == "R12":
            color_labels.append(f"🟢 Rounder #{slot_num}")
        elif shift_key == "A12":
            color_labels.append(f"🟡 Admitter #{slot_num}")
        elif shift_key == "A10":
            color_labels.append(f"🔴 Admitter #{slot_num}")
        elif shift_key == "N12":
            color_labels.append(f"🟣 Night #{slot_num}")
        elif shift_key == "NB":
            color_labels.append(f"🔵 Bridge #{slot_num}")
        elif shift_key == "APP":
            color_labels.append(f"🟪 APP #{slot_num}")
        else:
            color_labels.append(f"{shift_type} #{slot_num}")
    
    grid_data["Shift Type"] = color_labels
    
    # Add date columns
    for date_col in date_cols:
        grid_data[date_col] = []
        for idx, row in df.iterrows():
            grid_data[date_col].append(row[date_col] if row[date_col] else "")
    
    grid_df = pd.DataFrame(grid_data)
    
    # Calculate height to avoid vertical scroll
    height_px = min(2200, 110 + len(row_meta) * 38)
    
    # Calculate optimal width for first column based on longest label
    max_label_length = max(len(label) for label in color_labels)
    # Base width calculation: ~8px per character + padding + emoji space
    optimal_width = max(140, min(200, max_label_length * 8 + 40))
    
    # Create column configuration for the data editor
    col_config = {
        "Shift Type": st.column_config.TextColumn(
            "Shift Type",
            disabled=True,
            help="Shift type and slot number",
            width=optimal_width  # Dynamic width based on content
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
    
    # Get selected provider from calendar filter for highlighting
    # Use a unique key for grid context to avoid key collisions with main calendar
    selected_provider = st.session_state.get("grid_calendar_provider_filter", st.session_state.get("calendar_provider_filter", "All Providers"))
    
    # Add visual indicator for selected provider
    if selected_provider != "All Providers":
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            text-align: center;
            font-weight: bold;
            color: #856404;
        ">
            🎯 <strong>Highlighting Provider:</strong> {selected_provider}
            <br>
            <small>All cells containing "{selected_provider}" are highlighted in yellow</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Removed freeze-pane banner for a cleaner UI
    
    # Add custom CSS for sticky column, better styling, and provider highlighting
    st.markdown(f"""
    <style>
        .grid-container {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
        }}
        
        
        
        /* Enhanced freeze-pane functionality for first column */
        [data-testid="stDataFrame"] {{
            overflow-x: auto !important;
            max-width: 100% !important;
            position: relative !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 8px !important;
        }}
        
        /* Table styling */
        [data-testid="stDataFrame"] table {{
            border-collapse: separate !important;
            border-spacing: 0 !important;
            width: 100% !important;
        }}
        
        /* Freeze-pane: First column (Shift Type) - Dynamic width */
        [data-testid="stDataFrame"] th:first-child,
        [data-testid="stDataFrame"] td:first-child {{
            position: sticky !important;
            left: 0 !important;
            z-index: 1000 !important;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
            width: {optimal_width}px !important;
            min-width: {optimal_width}px !important;
            max-width: {optimal_width}px !important;
            border-right: 3px solid #FF674D !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 8px rgba(255, 103, 77, 0.15) !important;
            font-weight: 600 !important;
            padding: 8px 12px !important;
        }}
        
        /* Enhanced header for first column */
        [data-testid="stDataFrame"] thead th:first-child {{
            position: sticky !important;
            left: 0 !important;
            z-index: 1001 !important;
            background: linear-gradient(135deg, #FF674D 0%, #ff8a65 100%) !important;
            color: white !important;
            width: {optimal_width}px !important;
            min-width: {optimal_width}px !important;
            max-width: {optimal_width}px !important;
            border-right: 3px solid #d32f2f !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            box-shadow: 2px 0 8px rgba(255, 103, 77, 0.3) !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 10px 12px !important;
        }}
        
        /* Hover effect for first column */
        [data-testid="stDataFrame"] tbody tr:hover td:first-child {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
            border-right: 3px solid #ff5722 !important;
        }}
        
        /* Provider highlighting styles */
        .highlight-provider {{
            background-color: #fff3cd !important;
            border: 2px solid #ffc107 !important;
            font-weight: bold !important;
            color: #856404 !important;
        }}
        
        /* Date columns styling - optimized for scrolling */
        [data-testid="stDataFrame"] th:not(:first-child),
        [data-testid="stDataFrame"] td:not(:first-child) {{
            min-width: 85px !important;
            max-width: 95px !important;
            text-align: center !important;
            padding: 6px 4px !important;
            border-left: 1px solid #dee2e6 !important;
        }}
        
        /* Date column headers */
        [data-testid="stDataFrame"] thead th:not(:first-child) {{
            background: linear-gradient(135deg, #2196f3 0%, #42a5f5 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: center !important;
            border-bottom: 2px solid #1976d2 !important;
            font-size: 12px !important;
        }}
        
        /* Improve overall table styling */
        [data-testid="stDataFrame"] table {{
            border: none !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }}
        
        /* Regular table cells */
        [data-testid="stDataFrame"] td:not(:first-child) {{
            border-left: 1px solid #e0e0e0 !important;
            border-bottom: 1px solid #e0e0e0 !important;
            padding: 6px 4px !important;
            background: white !important;
        }}
        
        /* Row hover effects (excluding first column) */
        [data-testid="stDataFrame"] tbody tr:hover td:not(:first-child) {{
            background-color: #f5f5f5 !important;
        }}
        
        /* Weekend column highlighting */
        [data-testid="stDataFrame"] th:nth-child(7n+1):not(:first-child),
        [data-testid="stDataFrame"] th:nth-child(7n):not(:first-child),
        [data-testid="stDataFrame"] td:nth-child(7n+1):not(:first-child),
        [data-testid="stDataFrame"] td:nth-child(7n):not(:first-child) {{
            background-color: #fff3e0 !important;
        }}
        
        /* Smooth scrolling */
        [data-testid="stDataFrame"] {{
            scroll-behavior: smooth !important;
        }}
    </style>
    
    <script>
        // Function to highlight the selected provider in the grid
        function highlightSelectedProvider() {{
            const selectedProvider = "{selected_provider}";
            
            // Remove existing highlights
            document.querySelectorAll('.highlight-provider').forEach(el => {{
                el.classList.remove('highlight-provider');
            }});
            
            // Don't highlight if "All Providers" is selected
            if (selectedProvider === "All Providers") {{
                return;
            }}
            
            // Find all cells containing the selected provider and highlight them
            const cells = document.querySelectorAll('[data-testid="stDataFrame"] td');
            cells.forEach(cell => {{
                if (cell.textContent.trim() === selectedProvider) {{
                    cell.classList.add('highlight-provider');
                }}
            }});
        }}
        
        // Run highlighting when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            // Wait a bit for the data editor to fully load
            setTimeout(highlightSelectedProvider, 1000);
        }});
        
        // Also run highlighting when the data editor updates
        const observer = new MutationObserver(function(mutations) {{
            mutations.forEach(function(mutation) {{
                if (mutation.type === 'childList') {{
                    setTimeout(highlightSelectedProvider, 500);
                }}
            }});
        }});
        
        // Start observing when the data editor is available
        setTimeout(function() {{
            const dataFrame = document.querySelector('[data-testid="stDataFrame"]');
            if (dataFrame) {{
                observer.observe(dataFrame, {{ childList: true, subtree: true }});
            }}
        }}, 1000);
    </script>
    """, unsafe_allow_html=True)
    
    # Wrap in container
    with st.container():
        st.markdown('<div id="imis-grid-container" class="grid-container">', unsafe_allow_html=True)
        
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

        # Inject a tiny helper to run in a sandboxed iframe that can access the parent DOM
        # to apply highlights inside the grid editor (Streamlit blocks <script> in markdown).
        try:
            import streamlit.components.v1 as components
            components.html(f"""
            <script>
            (function() {{
                const selectedProvider = "{selected_provider}";
                function highlight() {{
                    try {{
                        const doc = window.parent && window.parent.document ? window.parent.document : document;
                        const container = doc.querySelector('#imis-grid-container');
                        if (!container) return;
                        // Clear previous highlights
                        container.querySelectorAll('.highlight-provider').forEach(el => el.classList.remove('highlight-provider'));
                        if (!selectedProvider || selectedProvider === 'All Providers') return;
                        // Highlight exact matches in visible cells only
                        const cells = container.querySelectorAll('[data-testid="stDataFrame"] td');
                        cells.forEach(cell => {{
                            const text = (cell.textContent || '').trim();
                            if (text === selectedProvider) {{
                                cell.classList.add('highlight-provider');
                            }}
                        }});
                    }} catch (e) {{}}
                }}
                // Initial run and observe changes
                setTimeout(highlight, 300);
                const doc = window.parent && window.parent.document ? window.parent.document : document;
                const observer = new MutationObserver(() => setTimeout(highlight, 200));
                observer.observe(doc.body, {{childList: true, subtree: true}});
            }})();
            </script>
            """, height=0)
        except Exception:
            pass
    
    # Handle grid changes without forcing a full rerun (keeps calendar visible)
    if edited_grid is not None and not edited_grid.equals(grid_df):
        # Apply changes to events
        updated_events = apply_grid_changes_to_calendar(edited_grid, events, year, month, row_meta)
        st.session_state.events = updated_events
        
        # Auto-save the updated schedule
        from core.data_manager import save_schedule
        save_schedule(year, month, st.session_state.events)
        
        # Analyze for common violations and show warnings
        provider_rules = st.session_state.get("provider_rules", {})
        _show_grid_change_warnings(st.session_state.events, provider_rules)

        st.toast("Grid changes applied and saved. Calendar updated.")

        # Live calendar preview below the grid so users immediately see updates
        try:
            from ui.calendar import render_calendar
            st.markdown("---")
            st.markdown("#### 📅 Live Calendar Preview (updates with grid changes)")
            render_calendar(st.session_state.events, height=520)
        except Exception:
            pass
    
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

def _show_grid_change_warnings(events: List[Any], provider_rules: Dict[str, Dict]) -> None:
    """Display warnings for double assignments, block > 7, and preference violations."""
    # Normalize events
    normalized: List[Dict[str, Any]] = []
    for e in events:
        try:
            if hasattr(e, 'start') and hasattr(e, 'extendedProps'):
                normalized.append({
                    'start': e.start.isoformat() if hasattr(e.start, 'isoformat') else str(e.start),
                    'extendedProps': {
                        'provider': e.extendedProps.get('provider', ''),
                        'shift_type': e.extendedProps.get('shift_type') or e.extendedProps.get('shift_key')
                    }
                })
            elif isinstance(e, dict) and 'start' in e:
                normalized.append(e)
        except Exception:
            continue

    # Maps
    date_to_provider_counts: Dict[date, Dict[str, int]] = {}
    provider_dates: Dict[str, List[date]] = {}
    pref_violations: List[str] = []

    for ev in normalized:
        try:
            d = datetime.fromisoformat(ev['start']).date()
        except Exception:
            continue
        provider = (ev.get('extendedProps', {}).get('provider') or '').strip().upper()
        shift_type = ev.get('extendedProps', {}).get('shift_type')
        if not provider:
            continue

        date_to_provider_counts.setdefault(d, {})
        date_to_provider_counts[d][provider] = date_to_provider_counts[d].get(provider, 0) + 1

        provider_dates.setdefault(provider, []).append(d)

        pr = provider_rules.get(provider, {})
        sp = pr.get('shift_preferences', {})
        if shift_type in sp and sp[shift_type] is False:
            pref_violations.append(f"{provider} assigned {shift_type} against preference on {d}")

    warnings: List[str] = []

    # Double assignment
    for d, counts in date_to_provider_counts.items():
        doubles = [p for p, c in counts.items() if c > 1]
        if doubles:
            warnings.append(f"Double assignment on {d}: {', '.join(sorted(doubles))}")

    # Block length > 7
    for provider, dates in provider_dates.items():
        if not dates:
            continue
        ds = sorted(set(dates))
        longest = 1
        run = 1
        for i in range(1, len(ds)):
            if (ds[i] - ds[i-1]).days == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest > 7:
            warnings.append(f"Block longer than 7 days for {provider} (longest {longest})")

    # Pref violations (limit list length)
    if pref_violations:
        uniq = sorted(set(pref_violations))
        warnings.extend(uniq[:10])
        if len(uniq) > 10:
            warnings.append(f"… and {len(uniq) - 10} more preference violations")

    if warnings:
        st.warning("\n".join([f"• {w}" for w in warnings]))
