# =============================================================================
# Provider Management UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Any
import json

def providers_panel():
    """Main providers management panel."""
    st.header("👥 Provider Management")
    
    # Provider statistics at the top
    if not st.session_state.providers_df.empty:
        providers_df = st.session_state.providers_df.copy()
        providers_df["initials"] = providers_df["initials"].astype(str).str.upper()
        
        physician_count = len(providers_df[providers_df["type"] == "Physician"])
        app_count = len(providers_df[providers_df["type"] == "APP"])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Providers", len(providers_df))
        with col2:
            st.metric("Physicians", physician_count)
        with col3:
            st.metric("APPs", app_count)
        with col4:
            if st.session_state.events:
                unique_providers = set()
                for event in st.session_state.events:
                    provider_val = None
                    if hasattr(event, 'extendedProps'):
                        provider_val = event.extendedProps.get("provider", "")
                    elif isinstance(event, dict) and 'extendedProps' in event:
                        provider_val = event['extendedProps'].get("provider", "")
                    if provider_val:
                        unique_providers.add(provider_val)
                st.metric("Currently Scheduled", len(unique_providers))
            else:
                st.metric("Currently Scheduled", 0)
        
        st.markdown("---")
        
        # Compact provider selection with dropdown
        st.markdown("#### 🔍 Provider Selection & Actions")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Create a formatted list for the dropdown
            provider_options = []
            for _, row in providers_df.iterrows():
                provider_options.append(f"{row['initials']} - {row['name']} ({row['type']})")
            
            selected_provider_full = st.selectbox(
                "Select a provider for actions:",
                options=provider_options,
                help="Choose a provider to view details, edit rules, or perform actions"
            )
        
        with col2:
            filter_type = st.selectbox(
                "Filter by type",
                options=["All", "Physician", "APP"],
                key="provider_type_filter"
            )
        
        with col3:
            if st.button("📋 View Selected Provider", type="primary"):
                if selected_provider_full:
                    # Extract initials from the selected option
                    selected_initials = selected_provider_full.split(" - ")[0]
                    st.session_state.selected_provider_for_rules = selected_initials
                    st.rerun()
        
        # Filter providers based on type
        if filter_type != "All":
            filtered_df = providers_df[providers_df["type"] == filter_type]
        else:
            filtered_df = providers_df
        
        # Show selected provider details
        if selected_provider_full:
            selected_initials = selected_provider_full.split(" - ")[0]
            provider_info = providers_df[providers_df["initials"] == selected_initials]
            
            if not provider_info.empty:
                provider_info = provider_info.iloc[0]
                
                st.markdown("---")
                st.markdown("#### 📊 Selected Provider Details")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Provider:** {provider_info['name']}")
                    st.markdown(f"**Initials:** {provider_info['initials']}")
                    st.markdown(f"**Type:** {provider_info['type']}")
                
                with col2:
                    # Show current schedule count for this provider
                    if st.session_state.events:
                        provider_events = 0
                        for event in st.session_state.events:
                            provider_val = None
                            if hasattr(event, 'extendedProps'):
                                provider_val = event.extendedProps.get("provider", "")
                            elif isinstance(event, dict) and 'extendedProps' in event:
                                provider_val = event['extendedProps'].get("provider", "")
                            if provider_val == selected_initials:
                                provider_events += 1
                        st.metric("Current Shifts", provider_events)
                    else:
                        st.metric("Current Shifts", 0)
                    
                    # Show provider rules status
                    if "provider_rules" in st.session_state and selected_initials in st.session_state.provider_rules:
                        rules = st.session_state.provider_rules[selected_initials]
                        st.markdown(f"**Min Shifts:** {rules.get('min_shifts', 'Not set')}")
                        st.markdown(f"**Max Shifts:** {rules.get('max_shifts', 'Not set')}")
                    else:
                        st.markdown("**Rules:** Not configured")
                
                with col3:
                    if st.button("📋 View/Edit Rules", key=f"view_rules_{selected_initials}"):
                        st.session_state.selected_provider_for_rules = selected_initials
                        st.rerun()
                    
                    if st.button("📅 View Schedule", key=f"view_schedule_{selected_initials}"):
                        st.info(f"Go to Calendar tab and select '{selected_initials}' from the provider filter to view their schedule.")
                    
                    if st.button("📊 View Stats", key=f"view_stats_{selected_initials}"):
                        st.info(f"Provider statistics will be shown here for {selected_initials}.")
        

        
        # Provider rules section
        st.markdown("---")
        st.subheader("⚙️ Provider Rules")
        provider_rules_panel()
    else:
        st.info("No providers loaded. Please load a CSV file with provider data or add new providers.")
    
    st.markdown("---")
    
    # Management actions in expandable sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Add new provider section
        with st.expander("➕ Add New Provider", expanded=False):
            add_new_provider()
    
    with col2:
        # Load providers section
        with st.expander("📁 Load Providers from CSV", expanded=False):
            load_providers_from_csv()
    
    # Clean up providers section (full width)
    with st.expander("🧹 Clean Up Providers", expanded=False):
        cleanup_providers()

def add_new_provider():
    """Add a new provider with comprehensive information."""
    st.markdown("#### Add New Provider")
    st.markdown("Fill in the provider information below. All fields marked with * are required.")
    
    with st.form("add_provider_form"):
        # Basic Information
        st.markdown("**Basic Information**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initials = st.text_input(
                "Provider Initials *",
                placeholder="e.g., JT",
                max_chars=5,
                help="Enter 2-5 character initials (will be converted to uppercase)"
            )
            
            provider_name = st.text_input(
                "Full Name *",
                placeholder="e.g., Joel T Park",
                help="Enter the provider's full name"
            )
        
        with col2:
            provider_type = st.selectbox(
                "Provider Type *",
                options=["Physician", "APP"],
                help="Select whether this is a Physician or APP provider"
            )
            # Removed duplicate quick day/night preference (exists in detailed rules below)
        
        st.markdown("---")
        
        # Shift Preferences
        st.markdown("**Shift Preferences**")
        st.markdown("Select which types of shifts this provider can work:")
        
        # Get shift types from constants
        from models.constants import DEFAULT_SHIFT_TYPES
        shift_types = [shift["key"] for shift in DEFAULT_SHIFT_TYPES]
        shift_labels = [shift["label"] for shift in DEFAULT_SHIFT_TYPES]
        
        # Create shift preference checkboxes with HARD RULE defaults
        shift_preferences = {}
        cols = st.columns(3)
        
        for i, (shift_key, shift_label) in enumerate(zip(shift_types, shift_labels)):
            with cols[i % 3]:
                # HARD RULE: Default to False - providers must opt-in to shift types
                # Only set to True for obvious cases (APP providers for APP shifts)
                default_value = False
                if provider_type == "APP" and shift_key == "APP":
                    default_value = True
                # Note: Physician providers must explicitly choose their shift types
                
                shift_preferences[shift_key] = st.checkbox(
                    f"{shift_label} ({shift_key})",
                    value=default_value,
                    key=f"shift_pref_{shift_key}",
                    help=f"Can this provider work {shift_label} shifts? (HARD RULE: Must be checked to assign this shift type)"
                )
        
        st.markdown("---")
        
        # Workload Preferences
        st.markdown("**Workload Preferences**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # FTE (Full Time Employment) setting
            fte_percentage = st.slider(
                "FTE Percentage",
                min_value=0.1, max_value=2.0, step=0.1,
                value=1.0,
                help="Full Time Employment percentage. 1.0 = 100% full time (15-16 shifts), 0.8 = 80% (12-13 shifts), etc."
            )
            
            # Calculate and display expected shifts based on FTE
            from core.utils import get_expected_shifts_for_month
            from datetime import datetime
            current_date = datetime.now()
            base_expected_30 = get_expected_shifts_for_month(current_date.year, 4)  # April has 30 days
            base_expected_31 = get_expected_shifts_for_month(current_date.year, 1)  # January has 31 days
            
            expected_30 = int(round(base_expected_30 * fte_percentage))
            expected_31 = int(round(base_expected_31 * fte_percentage))
            
            st.info(f"📊 **Expected Shifts:** {expected_30} (30-day), {expected_31} (31-day)")
        
        with col2:
            min_weekend_shifts = st.number_input(
                "Min Weekend Shifts",
                min_value=0, max_value=10,
                value=1,
                help="Minimum number of weekend shifts per month"
            )
            
            max_weekend_shifts = st.number_input(
                "Max Weekend Shifts",
                min_value=0, max_value=10,
                value=4,
                help="Maximum number of weekend shifts per month"
            )
        
        st.markdown("---")
        
        # Day/Night Percentage
        st.markdown("**Day vs Night Shift Distribution**")
        st.markdown("Set the preferred percentage of day shifts (night shifts will be the remainder):")
        
        day_percentage = st.slider(
            "Day Shifts Percentage",
            min_value=0, max_value=100,
            value=70,
            help="Percentage of shifts that should be day shifts (7am-7pm). Night shifts will be 100% minus this value."
        )
        
        night_percentage = 100 - day_percentage
        
        # Show the actual percentages
        st.info(f"📊 **Distribution:** {day_percentage}% Day Shifts, {night_percentage}% Night Shifts")
        
        # Shift Timing Preference
        st.markdown("**Shift Timing Preference**")
        timing_preference = st.selectbox(
            "When should shifts be scheduled?",
            options=["Even Distribution - No preference", "Front-Loaded - Prefer shifts in first half of month", "Back-Loaded - Prefer shifts in second half of month"],
            help="Choose when this provider prefers to work their shifts during the month"
        )
        
        # Convert to value
        timing_values = ["even_distribution", "front_loaded", "back_loaded"]
        timing_options = ["Even Distribution - No preference", "Front-Loaded - Prefer shifts in first half of month", "Back-Loaded - Prefer shifts in second half of month"]
        timing_preference_value = timing_values[timing_options.index(timing_preference)]
        
        # Senior Provider Status
        st.markdown("**Senior Provider Status**")
        is_senior = st.checkbox(
            "👑 Senior Provider",
            value=False,
            help="Check this box if this provider is a senior provider who only works 7am-7pm rounding shifts"
        )
        
        if is_senior:
            st.info("👑 **Senior Provider:** Will only be assigned 7am-7pm rounding shifts (R12)")
            # Automatically set shift preferences for senior providers
            shift_preferences = {
                "R12": True,   # Only rounding shifts
                "A12": False,  # No admitting shifts
                "A10": False,  # No admitting shifts
                "N12": False,  # No night shifts
                "NB": False,   # No night bridge
                "APP": False   # No APP shifts
            }
        
        # Unavailable Days of the Week
        st.markdown("**Unavailable Days of the Week**")
        st.markdown("Select days when this provider should not be assigned shifts:")
        
        days_of_week = [
            (0, "Monday"),
            (1, "Tuesday"), 
            (2, "Wednesday"),
            (3, "Thursday"),
            (4, "Friday"),
            (5, "Saturday"),
            (6, "Sunday")
        ]
        
        unavailable_days_of_week = []
        cols = st.columns(7)
        for i, (day_num, day_name) in enumerate(days_of_week):
            with cols[i]:
                if st.checkbox(day_name, key=f"new_provider_unavailable_{day_num}"):
                    unavailable_days_of_week.append(day_num)
        
        # Submit button
        submitted = st.form_submit_button("➕ Add Provider", type="primary")
        
        if submitted:
            # Validate required fields
            if not initials or not provider_name:
                st.error("❌ Please fill in all required fields (marked with *)")
                return
            
            # Clean and validate initials
            initials = initials.strip().upper()
            if len(initials) < 2 or len(initials) > 5:
                st.error("❌ Initials must be 2-5 characters long")
                return
            
            # Check if initials already exist
            if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
                existing_initials = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                if initials in existing_initials:
                    st.error(f"❌ Provider with initials '{initials}' already exists")
                    return
            
            # Create new provider data
            new_provider = {
                "initials": initials,
                "name": provider_name.strip(),
                "type": provider_type
            }
            
            # Add to providers dataframe
            if "providers_df" in st.session_state:
                new_df = pd.concat([st.session_state.providers_df, pd.DataFrame([new_provider])], ignore_index=True)
            else:
                new_df = pd.DataFrame([new_provider])
            
            st.session_state.providers_df = new_df
            st.session_state.providers_loaded = True
            
            # Create provider rules
            if "provider_rules" not in st.session_state:
                st.session_state.provider_rules = {}
            
            st.session_state.provider_rules[initials] = {
                "fte": fte_percentage,
                "min_weekend_shifts": min_weekend_shifts,
                "max_weekend_shifts": max_weekend_shifts,
                "day_percentage": day_percentage,
                "shift_preferences": shift_preferences,
                "unavailable_dates": [],
                "unavailable_days_of_week": unavailable_days_of_week,
                "shift_timing_preference": timing_preference_value,
                "is_senior": is_senior,
                "vacations": []
            }
            
            # Update senior providers list if needed
            if is_senior:
                from core.provider_types import add_senior_provider
                add_senior_provider(initials)
            
            # Auto-save providers and rules
            from core.data_manager import save_providers, save_rules
            save_providers(new_df, st.session_state.provider_rules)
            save_rules(
                st.session_state.global_rules,
                st.session_state.shift_types,
                st.session_state.shift_capacity,
                st.session_state.provider_rules
            )
            
            st.success(f"✅ Successfully added provider {initials} - {provider_name}!")
            st.info(f"📋 Provider rules have been set up with your preferences. You can edit them in the Provider Rules section.")
            st.rerun()

def load_providers_from_csv():
    """Load providers from CSV file."""
    uploaded_file = st.file_uploader(
        "Choose a CSV file with provider data",
        type=['csv'],
        help="CSV should have columns: initials, name, type (Physician/APP)"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['initials', 'name', 'type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Clean and validate data
            df = df.dropna(subset=['initials', 'name'])
            df['initials'] = df['initials'].astype(str).str.upper().str.strip()
            df['name'] = df['name'].astype(str).str.strip()
            df['type'] = df['type'].astype(str).str.strip()
            
            # Validate provider types
            valid_types = ['Physician', 'APP']
            invalid_types = df[~df['type'].isin(valid_types)]['type'].unique()
            if len(invalid_types) > 0:
                st.warning(f"Invalid provider types found: {invalid_types}. Converting to 'Physician'.")
                df['type'] = df['type'].apply(lambda x: 'Physician' if x not in valid_types else x)
            
            # Store in session state
            st.session_state.providers_df = df
            st.session_state.providers_loaded = True
            
            # Auto-save providers
            from core.data_manager import save_providers
            save_providers(df, st.session_state.get('provider_rules', {}))
            
            st.success(f"✅ Successfully loaded and saved {len(df)} providers!")
            
            # Show preview
            with st.expander("Preview loaded data"):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")

def cleanup_providers():
    """Clean up old/non-current providers."""
    st.markdown("#### 🧹 Clean Up Providers")
    st.markdown("Remove providers who are no longer active or current.")
    
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers_df = st.session_state.providers_df.copy()
        
        # Show current providers
        st.markdown("**Current Providers:**")
        
        # Create a list of providers to remove
        providers_to_remove = st.multiselect(
            "Select providers to remove:",
            options=providers_df["initials"].tolist(),
            format_func=lambda x: f"{x} - {providers_df[providers_df['initials'] == x]['name'].iloc[0]} ({providers_df[providers_df['initials'] == x]['type'].iloc[0]})",
            help="Select providers who are no longer active"
        )
        
        if providers_to_remove:
            st.warning(f"⚠️ **Warning:** You are about to remove {len(providers_to_remove)} provider(s). This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Remove Selected Providers", type="secondary"):
                    # Remove providers from dataframe
                    new_df = providers_df[~providers_df["initials"].isin(providers_to_remove)]
                    st.session_state.providers_df = new_df
                    
                    # Remove from provider rules
                    if "provider_rules" in st.session_state:
                        for provider in providers_to_remove:
                            if provider in st.session_state.provider_rules:
                                del st.session_state.provider_rules[provider]
                    
                    # Auto-save
                    from core.data_manager import save_providers, save_rules
                    save_providers(new_df, st.session_state.provider_rules)
                    save_rules(
                        st.session_state.global_rules,
                        st.session_state.shift_types,
                        st.session_state.shift_capacity,
                        st.session_state.provider_rules
                    )
                    
                    st.success(f"✅ Successfully removed {len(providers_to_remove)} provider(s)!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", type="secondary"):
                    st.rerun()
    else:
        st.info("No providers to clean up.")

def provider_rules_selector():
    """Provider selection for rules editing."""
    if st.session_state.providers_df.empty:
        st.warning("No providers available.")
        return None
    
    providers = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
    selected_provider = st.selectbox(
        "Select Provider to Edit Rules",
        options=providers,
        key="provider_rules_selector"
    )
    
    return selected_provider

def provider_rules_panel():
    """Panel for editing provider-specific rules."""
    # Check if a provider was selected from quick actions
    if hasattr(st.session_state, 'selected_provider_for_rules') and st.session_state.selected_provider_for_rules:
        selected_provider = st.session_state.selected_provider_for_rules
        # Clear the selection after using it
        del st.session_state.selected_provider_for_rules
    else:
        selected_provider = provider_rules_selector()
    
    if not selected_provider:
        st.info("Select a provider above to edit their rules, or use the Quick Actions tab to select a provider.")
        return
    
    # Get provider info for display
    if "providers_df" in st.session_state and not st.session_state.providers_df.empty:
        providers_df = st.session_state.providers_df
        provider_info = providers_df[providers_df["initials"] == selected_provider]
        if not provider_info.empty:
            provider_name = provider_info.iloc[0]["name"]
            provider_type = provider_info.iloc[0]["type"]
        else:
            provider_name = selected_provider
            provider_type = "Unknown"
    else:
        provider_name = selected_provider
        provider_type = "Unknown"
    
    st.markdown(f"### Rules for {selected_provider} - {provider_name}")
    st.markdown(f"**Type:** {provider_type}")
    
    # Initialize provider rules if not exists
    if selected_provider not in st.session_state.provider_rules:
        st.session_state.provider_rules[selected_provider] = {
            "fte": 1.0,  # Default to 100% full time
            "min_weekend_shifts": 1,
            "max_weekend_shifts": 4,
            "min_night_shifts": 2,
            "max_night_shifts": 8,
            "day_percentage": 70,
            "night_percentage": 30,
            "day_night_preference": "No Preference",
            "shift_preferences": {
                # HARD RULE: All shift types default to False - providers must opt-in
                "R12": False,
                "A12": False,
                "A10": False,
                "N12": False,
                "NB": False,
                "APP": False
            },
            "unavailable_dates": [],
            "unavailable_days_of_week": [],
            "shift_timing_preference": "even_distribution",
            "vacations": []
        }
    
    provider_rules = st.session_state.provider_rules[selected_provider]
    
    # Create tabs for different rule categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Shift Limits", "🎯 Shift Preferences", "🚫 Unavailable Dates", "🏖️ Vacations", "⏰ Timing Preferences", "👑 Senior Status"])
    
    with tab1:
        st.markdown("#### FTE (Full Time Employment)")
        st.markdown("Set the provider's FTE percentage to determine expected shifts:")
        
        fte_percentage = st.slider(
            "FTE Percentage",
            min_value=0.1, max_value=2.0, step=0.1,
            value=provider_rules.get("fte", 1.0),
            key=f"fte_{selected_provider}",
            help="Full Time Employment percentage. 1.0 = 100% full time (15-16 shifts), 0.8 = 80% (12-13 shifts), etc."
        )
        provider_rules["fte"] = fte_percentage
        
        # Calculate and display expected shifts based on FTE
        from core.utils import get_expected_shifts_for_month
        from datetime import datetime
        current_date = datetime.now()
        base_expected_30 = get_expected_shifts_for_month(current_date.year, 4)  # April has 30 days
        base_expected_31 = get_expected_shifts_for_month(current_date.year, 1)  # January has 31 days
        
        expected_30 = int(round(base_expected_30 * fte_percentage))
        expected_31 = int(round(base_expected_31 * fte_percentage))
        
        st.info(f"📊 **Expected Shifts:** {expected_30} shifts (30-day months), {expected_31} shifts (31-day months)")
        
        # Day/Night distribution
        st.markdown("#### Day vs Night Shift Distribution")
        st.markdown("Set the preferred percentage of day shifts (night shifts will be the remainder):")
        
        day_percentage = st.slider(
            "Day Shifts Percentage",
            min_value=0, max_value=100,
            value=provider_rules.get("day_percentage", 70),
            key=f"day_percentage_{selected_provider}",
            help="Percentage of shifts that should be day shifts (7am-7pm). Night shifts will be 100% minus this value."
        )
        
        provider_rules["day_percentage"] = day_percentage
        provider_rules["night_percentage"] = 100 - day_percentage
        
        st.info(f"📊 **Distribution:** {day_percentage}% Day Shifts, {100-day_percentage}% Night Shifts")
        
        # Calculate and display night shift limits based on FTE and day/night percentage
        night_percentage = 100 - day_percentage
        night_shifts_30 = int(round(expected_30 * night_percentage / 100))
        night_shifts_31 = int(round(expected_31 * night_percentage / 100))
        
        st.markdown("#### Calculated Night Shift Limits")
        st.info(f"🌙 **Night Shifts:** {night_shifts_30} shifts (30-day months), {night_shifts_31} shifts (31-day months)")
        
        # Store the calculated values
        provider_rules["min_night_shifts"] = max(0, night_shifts_30 - 2)  # Allow some flexibility
        provider_rules["max_night_shifts"] = night_shifts_31 + 2  # Allow some flexibility
        
        # Weekend shift limits
        st.markdown("#### Weekend Shift Limits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            provider_rules["min_weekend_shifts"] = st.number_input(
                "Min Weekend Shifts per Month",
                min_value=0, max_value=10,
                value=provider_rules.get("min_weekend_shifts", 1),
                key=f"min_weekend_{selected_provider}",
                help="Minimum number of weekend shifts this provider should work per month"
            )
        
        with col2:
            provider_rules["max_weekend_shifts"] = st.number_input(
                "Max Weekend Shifts per Month",
                min_value=0, max_value=10,
                value=provider_rules.get("max_weekend_shifts", 4),
                key=f"max_weekend_{selected_provider}",
                help="Maximum number of weekend shifts this provider can work per month"
            )
        

        
        # Day/Night preference
        provider_rules["day_night_preference"] = st.selectbox(
            "Day/Night Preference",
            options=["No Preference", "Day Shifts Preferred", "Night Shifts Preferred", "Day Shifts Only", "Night Shifts Only"],
            index=["No Preference", "Day Shifts Preferred", "Night Shifts Preferred", "Day Shifts Only", "Night Shifts Only"].index(
                provider_rules.get("day_night_preference", "No Preference")
            ),
            key=f"day_night_pref_{selected_provider}",
            help="Select the provider's preference for day vs night shifts"
        )
    
    with tab2:
        st.markdown("#### Shift Type Preferences")
        st.markdown("**HARD RULE:** Select which types of shifts this provider can work. Only checked shift types will be assigned to this provider.")
        
        # Get shift types from constants
        from models.constants import DEFAULT_SHIFT_TYPES
        shift_types = [shift["key"] for shift in DEFAULT_SHIFT_TYPES]
        shift_labels = [shift["label"] for shift in DEFAULT_SHIFT_TYPES]
        
        # Initialize shift preferences if not exists
        if "shift_preferences" not in provider_rules:
            provider_rules["shift_preferences"] = {
                # HARD RULE: All shift types default to False - providers must opt-in
                "R12": False,
                "A12": False,
                "A10": False,
                "N12": False,
                "NB": False,
                "APP": False
            }
        
        # Create shift preference checkboxes
        cols = st.columns(3)
        
        for i, (shift_key, shift_label) in enumerate(zip(shift_types, shift_labels)):
            with cols[i % 3]:
                # HARD RULE: Default to False unless explicitly set
                current_value = provider_rules["shift_preferences"].get(shift_key, False)
                
                provider_rules["shift_preferences"][shift_key] = st.checkbox(
                    f"{shift_label} ({shift_key})",
                    value=current_value,
                    key=f"shift_pref_{selected_provider}_{shift_key}",
                    help=f"HARD RULE: Must be checked to assign {shift_label} shifts to this provider"
                )
    
    with tab3:
        st.markdown("#### Unavailable Dates")
        st.markdown("Add specific dates when this provider cannot work.")
        # Ensure key exists for older saved rules
        provider_rules.setdefault("unavailable_dates", [])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            unavailable_date = st.date_input(
                "Add Unavailable Date",
                key=f"unavailable_date_{selected_provider}"
            )
        
        with col2:
            if st.button("Add", key=f"add_unavailable_{selected_provider}"):
                if unavailable_date not in provider_rules["unavailable_dates"]:
                    provider_rules["unavailable_dates"].append(unavailable_date)
                    st.success(f"Added {unavailable_date} as unavailable date.")
                    st.rerun()
                else:
                    st.warning("Date already marked as unavailable.")
        
        # Display current unavailable dates
        if provider_rules.get("unavailable_dates"):
            st.markdown("**Current Unavailable Dates:**")
            for i, date_val in enumerate(sorted(provider_rules["unavailable_dates"])):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {date_val}")
                with col2:
                    if st.button("Remove", key=f"remove_unavailable_{selected_provider}_{i}"):
                        provider_rules["unavailable_dates"].remove(date_val)
                        st.success(f"Removed {date_val} from unavailable dates.")
                        st.rerun()
        else:
            st.info("No unavailable dates set.")
    
    with tab4:
        st.markdown("#### Vacation Periods")
        st.markdown("Add vacation periods when this provider will be away.")
        # Ensure key exists for older saved rules
        provider_rules.setdefault("vacations", [])
        
        col1, col2 = st.columns(2)
        with col1:
            vacation_start = st.date_input(
                "Vacation Start Date",
                key=f"vacation_start_{selected_provider}"
            )
        
        with col2:
            vacation_end = st.date_input(
                "Vacation End Date",
                key=f"vacation_end_{selected_provider}"
            )
        
        if st.button("Add Vacation", key=f"add_vacation_{selected_provider}"):
            if vacation_start <= vacation_end:
                vacation_period = {"start": vacation_start, "end": vacation_end}
                if vacation_period not in provider_rules["vacations"]:
                    provider_rules["vacations"].append(vacation_period)
                    st.success(f"Added vacation period: {vacation_start} to {vacation_end}")
                    st.rerun()
                else:
                    st.warning("Vacation period already exists.")
            else:
                st.error("Start date must be before or equal to end date.")
        
        # Display current vacations
        if provider_rules["vacations"]:
            st.markdown("**Current Vacation Periods:**")
            for i, vacation in enumerate(provider_rules["vacations"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {vacation['start']} to {vacation['end']}")
                with col2:
                    if st.button("Remove", key=f"remove_vacation_{selected_provider}_{i}"):
                        provider_rules["vacations"].remove(vacation)
                        st.success(f"Removed vacation period: {vacation['start']} to {vacation['end']}")
                        st.rerun()
        else:
            st.info("No vacation periods set.")
    
    with tab5:
        st.markdown("#### Shift Timing Preferences")
        st.markdown("Set preferences for when shifts should be scheduled during the month:")
        
        # Initialize timing preference if not exists
        if "shift_timing_preference" not in provider_rules:
            provider_rules["shift_timing_preference"] = "even_distribution"
        
        timing_options = ["Even Distribution - No preference", "Front-Loaded - Prefer shifts in first half of month", "Back-Loaded - Prefer shifts in second half of month"]
        timing_values = ["even_distribution", "front_loaded", "back_loaded"]
        
        current_timing = provider_rules.get("shift_timing_preference", "even_distribution")
        current_index = timing_values.index(current_timing) if current_timing in timing_values else 0
        
        timing_preference = st.selectbox(
            "Shift Timing Preference",
            options=timing_options,
            index=current_index,
            key=f"timing_pref_{selected_provider}",
            help="Choose when this provider prefers to work their shifts during the month"
        )
        
        # Convert selection back to value
        timing_preference = timing_values[timing_options.index(timing_preference)]
        provider_rules["shift_timing_preference"] = timing_preference
        
        st.info(f"📅 **Current Preference:** {dict([('even_distribution', 'Even Distribution'), ('front_loaded', 'Front-Loaded'), ('back_loaded', 'Back-Loaded')])[timing_preference]}")
        
        st.markdown("---")
        
        st.markdown("#### Unavailable Days of the Week")
        st.markdown("Select days of the week when this provider should not be assigned shifts:")
        
        # Initialize unavailable days of week if not exists
        if "unavailable_days_of_week" not in provider_rules:
            provider_rules["unavailable_days_of_week"] = []
        
        days_of_week = [
            (0, "Monday"),
            (1, "Tuesday"), 
            (2, "Wednesday"),
            (3, "Thursday"),
            (4, "Friday"),
            (5, "Saturday"),
            (6, "Sunday")
        ]
        
        # Create checkboxes for each day of the week
        cols = st.columns(7)
        for i, (day_num, day_name) in enumerate(days_of_week):
            with cols[i]:
                is_unavailable = st.checkbox(
                    day_name,
                    value=day_num in provider_rules.get("unavailable_days_of_week", []),
                    key=f"unavailable_day_{selected_provider}_{day_num}",
                    help=f"Check if {day_name} should be unavailable"
                )
                
                if is_unavailable and day_num not in provider_rules["unavailable_days_of_week"]:
                    provider_rules["unavailable_days_of_week"].append(day_num)
                elif not is_unavailable and day_num in provider_rules["unavailable_days_of_week"]:
                    provider_rules["unavailable_days_of_week"].remove(day_num)
        
        # Display current unavailable days
        unavailable_days = provider_rules.get("unavailable_days_of_week", [])
        if unavailable_days:
            unavailable_names = [day_name for day_num, day_name in days_of_week if day_num in unavailable_days]
            st.info(f"🚫 **Unavailable Days:** {', '.join(unavailable_names)}")
        else:
            st.info("✅ **Available all days of the week**")
    
    with tab6:
        st.markdown("#### Senior Provider Status")
        st.markdown("Senior providers only work 7am-7pm rounding shifts (R12).")
        
        # Initialize senior status if not exists
        if "is_senior" not in provider_rules:
            provider_rules["is_senior"] = False
        
        # Senior provider checkbox
        is_senior = st.checkbox(
            "👑 Senior Provider",
            value=provider_rules.get("is_senior", False),
            key=f"is_senior_{selected_provider}",
            help="Check this box if this provider is a senior provider who only works 7am-7pm rounding shifts"
        )
        
        provider_rules["is_senior"] = is_senior
        
        if is_senior:
            st.success("👑 **Senior Provider Status:** Active")
            st.info("✅ This provider will only be assigned 7am-7pm rounding shifts (R12)")
            st.info("🔒 All other shift types will be automatically disabled")
            
            # Automatically update shift preferences for senior providers
            provider_rules["shift_preferences"] = {
                "R12": True,   # Only rounding shifts
                "A12": False,  # No admitting shifts
                "A10": False,  # No admitting shifts
                "N12": False,  # No night shifts
                "NB": False,   # No night bridge
                "APP": False   # No APP shifts
            }
            
            # Update senior providers list
            from core.provider_types import add_senior_provider, remove_senior_provider, get_senior_providers
            
            current_seniors = get_senior_providers()
            if selected_provider not in current_seniors:
                add_senior_provider(selected_provider)
                st.success(f"✅ Added {selected_provider} to senior providers list")
        else:
            st.info("👤 **Regular Provider Status**")
            
            # Remove from senior providers list if they were previously a senior
            from core.provider_types import remove_senior_provider, get_senior_providers
            
            current_seniors = get_senior_providers()
            if selected_provider in current_seniors:
                remove_senior_provider(selected_provider)
                st.success(f"✅ Removed {selected_provider} from senior providers list")
        
        # Display current senior providers
        from core.provider_types import get_senior_providers
        current_seniors = get_senior_providers()
        if current_seniors:
            st.markdown("**Current Senior Providers:**")
            st.write(", ".join(sorted(current_seniors)))
        else:
            st.info("No senior providers currently set")
    
    # Save button
    if st.button("💾 Save Rules", type="primary", key=f"save_rules_{selected_provider}"):
        # Auto-save rules
        from core.data_manager import save_rules
        save_rules(
            st.session_state.global_rules,
            st.session_state.shift_types,
            st.session_state.shift_capacity,
            st.session_state.provider_rules
        )
        st.success(f"Rules saved for {selected_provider}!")
