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
    
    # Provider selector section
    from ui.provider_selector import render_provider_selector
    render_provider_selector()
    
    st.markdown("---")
    
    # Load providers section
    with st.expander("📁 Load Providers from CSV", expanded=False):
        load_providers_from_csv()
    
    # Provider list section with improved UI
    if not st.session_state.providers_df.empty:
        st.subheader("📋 Current Providers")
        
        # Provider statistics at the top
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
                    if isinstance(event, dict) and 'extendedProps' in event:
                        provider = event['extendedProps'].get("provider", "")
                        if provider:
                            unique_providers.add(provider)
                st.metric("Currently Scheduled", len(unique_providers))
            else:
                st.metric("Currently Scheduled", 0)
        
        st.markdown("---")
        
        # Search and filter section
        st.markdown("#### 🔍 Search & Filter Providers")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "Search by name or initials",
                placeholder="Enter provider name or initials...",
                key="provider_search"
            )
        
        with col2:
            filter_type = st.selectbox(
                "Filter by type",
                options=["All", "Physician", "APP"],
                key="provider_type_filter"
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=["Name", "Initials", "Type"],
                key="provider_sort"
            )
        
        # Filter and sort providers
        filtered_df = providers_df.copy()
        
        # Apply search filter
        if search_term:
            search_mask = (
                filtered_df["name"].str.contains(search_term, case=False, na=False) |
                filtered_df["initials"].str.contains(search_term.upper(), na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # Apply type filter
        if filter_type != "All":
            filtered_df = filtered_df[filtered_df["type"] == filter_type]
        
        # Apply sorting
        if sort_by == "Name":
            filtered_df = filtered_df.sort_values("name")
        elif sort_by == "Initials":
            filtered_df = filtered_df.sort_values("initials")
        elif sort_by == "Type":
            filtered_df = filtered_df.sort_values("type")
        
        # Display results with pagination
        if len(filtered_df) > 0:
            st.markdown(f"**Showing {len(filtered_df)} of {len(providers_df)} providers**")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Table View", "🎯 Quick Actions"])
            
            with tab1:
                # Display providers in a nice table with better styling
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    column_config={
                        "initials": st.column_config.TextColumn("Provider Initials", width="medium"),
                        "name": st.column_config.TextColumn("Full Name", width="large"),
                        "type": st.column_config.SelectboxColumn("Type", options=["Physician", "APP"], width="medium")
                    },
                    hide_index=True
                )
            
            with tab2:
                st.markdown("#### Quick Provider Actions")
                
                # Provider selection for quick actions
                selected_provider = st.selectbox(
                    "Select a provider for quick actions",
                    options=filtered_df["initials"].tolist(),
                    format_func=lambda x: f"{x} - {filtered_df[filtered_df['initials'] == x]['name'].iloc[0]}",
                    key="quick_action_provider"
                )
                
                if selected_provider:
                    provider_info = filtered_df[filtered_df["initials"] == selected_provider].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Provider:** {provider_info['name']}")
                        st.markdown(f"**Initials:** {provider_info['initials']}")
                        st.markdown(f"**Type:** {provider_info['type']}")
                        
                        # Show current schedule count for this provider
                        if st.session_state.events:
                            provider_events = 0
                            for event in st.session_state.events:
                                if isinstance(event, dict) and 'extendedProps' in event:
                                    if event['extendedProps'].get("provider", "") == selected_provider:
                                        provider_events += 1
                            st.markdown(f"**Current Shifts:** {provider_events}")
                    
                    with col2:
                        if st.button("📋 View Rules", key=f"view_rules_{selected_provider}"):
                            st.session_state.selected_provider_for_rules = selected_provider
                            st.rerun()
                        
                        if st.button("📅 View Schedule", key=f"view_schedule_{selected_provider}"):
                            st.info(f"Go to Calendar tab and select '{selected_provider}' from the provider filter to view their schedule.")
                        
                        if st.button("📊 View Stats", key=f"view_stats_{selected_provider}"):
                            st.info(f"Provider statistics will be shown here for {selected_provider}.")
        else:
            st.warning("No providers match your search criteria.")
        
        # Provider rules section
        st.markdown("---")
        st.subheader("⚙️ Provider Rules")
        provider_rules_panel()
    else:
        st.info("No providers loaded. Please load a CSV file with provider data.")

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
            "min_shifts": 8,
            "max_shifts": 16,
            "min_weekend_shifts": 1,
            "max_weekend_shifts": 4,
            "min_night_shifts": 2,
            "max_night_shifts": 8,
            "unavailable_dates": [],
            "vacations": []
        }
    
    provider_rules = st.session_state.provider_rules[selected_provider]
    
    # Create tabs for different rule categories
    tab1, tab2, tab3 = st.tabs(["📊 Shift Limits", "🚫 Unavailable Dates", "🏖️ Vacations"])
    
    with tab1:
        st.markdown("#### Monthly Shift Limits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            provider_rules["min_shifts"] = st.number_input(
                "Min Shifts per Month",
                min_value=0, max_value=31,
                value=provider_rules["min_shifts"],
                key=f"min_shifts_{selected_provider}",
                help="Minimum number of shifts this provider should work per month"
            )
            
            provider_rules["min_weekend_shifts"] = st.number_input(
                "Min Weekend Shifts per Month",
                min_value=0, max_value=10,
                value=provider_rules["min_weekend_shifts"],
                key=f"min_weekend_{selected_provider}",
                help="Minimum number of weekend shifts this provider should work per month"
            )
            
            provider_rules["min_night_shifts"] = st.number_input(
                "Min Night Shifts per Month",
                min_value=0, max_value=31,
                value=provider_rules["min_night_shifts"],
                key=f"min_night_{selected_provider}",
                help="Minimum number of night shifts this provider should work per month"
            )
        
        with col2:
            provider_rules["max_shifts"] = st.number_input(
                "Max Shifts per Month",
                min_value=1, max_value=31,
                value=provider_rules["max_shifts"],
                key=f"max_shifts_{selected_provider}",
                help="Maximum number of shifts this provider can work per month"
            )
            
            provider_rules["max_weekend_shifts"] = st.number_input(
                "Max Weekend Shifts per Month",
                min_value=0, max_value=10,
                value=provider_rules["max_weekend_shifts"],
                key=f"max_weekend_{selected_provider}",
                help="Maximum number of weekend shifts this provider can work per month"
            )
            
            provider_rules["max_night_shifts"] = st.number_input(
                "Max Night Shifts per Month",
                min_value=0, max_value=31,
                value=provider_rules["max_night_shifts"],
                key=f"max_night_{selected_provider}",
                help="Maximum number of night shifts this provider can work per month"
            )
    
    with tab2:
        st.markdown("#### Unavailable Dates")
        st.markdown("Add specific dates when this provider cannot work.")
        
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
        if provider_rules["unavailable_dates"]:
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
    
    with tab3:
        st.markdown("#### Vacation Periods")
        st.markdown("Add vacation periods when this provider will be away.")
        
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
