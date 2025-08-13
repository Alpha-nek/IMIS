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
    
    # Provider list section
    if not st.session_state.providers_df.empty:
        st.subheader("📋 Current Providers")
        
        # Display providers in a nice table
        providers_display = st.session_state.providers_df.copy()
        providers_display["initials"] = providers_display["initials"].astype(str).str.upper()
        
        st.dataframe(
            providers_display,
            use_container_width=True,
            column_config={
                "initials": st.column_config.TextColumn("Provider Initials", width="medium"),
                "name": st.column_config.TextColumn("Full Name", width="large"),
                "type": st.column_config.SelectboxColumn("Type", options=["Physician", "APP"], width="medium")
            }
        )
        
        # Provider statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Providers", len(providers_display))
        with col2:
            physicians = len(providers_display[providers_display["type"] == "Physician"])
            st.metric("Physicians", physicians)
        with col3:
            apps = len(providers_display[providers_display["type"] == "APP"])
            st.metric("APPs", apps)
        
        # Provider rules section
        st.subheader("⚙️ Provider Rules")
        provider_rules_selector()
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
    selected_provider = provider_rules_selector()
    
    if not selected_provider:
        return
    
    st.markdown(f"### Rules for {selected_provider}")
    
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
    
    # Basic rules
    col1, col2 = st.columns(2)
    
    with col1:
        provider_rules["min_shifts"] = st.number_input(
            "Min Shifts per Month",
            min_value=0, max_value=31,
            value=provider_rules["min_shifts"],
            key=f"min_shifts_{selected_provider}"
        )
        
        provider_rules["min_weekend_shifts"] = st.number_input(
            "Min Weekend Shifts per Month",
            min_value=0, max_value=10,
            value=provider_rules["min_weekend_shifts"],
            key=f"min_weekend_{selected_provider}"
        )
        
        provider_rules["min_night_shifts"] = st.number_input(
            "Min Night Shifts per Month",
            min_value=0, max_value=31,
            value=provider_rules["min_night_shifts"],
            key=f"min_night_{selected_provider}"
        )
    
    with col2:
        provider_rules["max_shifts"] = st.number_input(
            "Max Shifts per Month",
            min_value=1, max_value=31,
            value=provider_rules["max_shifts"],
            key=f"max_shifts_{selected_provider}"
        )
        
        provider_rules["max_weekend_shifts"] = st.number_input(
            "Max Weekend Shifts per Month",
            min_value=0, max_value=10,
            value=provider_rules["max_weekend_shifts"],
            key=f"max_weekend_{selected_provider}"
        )
        
        provider_rules["max_night_shifts"] = st.number_input(
            "Max Night Shifts per Month",
            min_value=0, max_value=31,
            value=provider_rules["max_night_shifts"],
            key=f"max_night_{selected_provider}"
        )
    
    # Unavailable dates
    st.subheader("🚫 Unavailable Dates")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        unavailable_date = st.date_input(
            "Add Unavailable Date",
            key=f"unavailable_date_{selected_provider}"
        )
    
    with col2:
        if st.button("Add", key=f"add_unavailable_{selected_provider}"):
            if unavailable_date not in provider_rules["unavailable_dates"]:
                provider_rules["unavailable_dates"].append(unavailable_date.isoformat())
                st.success(f"Added {unavailable_date} as unavailable")
                st.rerun()
            else:
                st.warning("Date already marked as unavailable")
    
    # Display unavailable dates
    if provider_rules["unavailable_dates"]:
        st.write("**Unavailable Dates:**")
        for i, date_str in enumerate(provider_rules["unavailable_dates"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {date_str}")
            with col2:
                if st.button("Remove", key=f"remove_unavailable_{selected_provider}_{i}"):
                    provider_rules["unavailable_dates"].pop(i)
                    st.rerun()
    
    # Vacations
    st.subheader("🏖️ Vacations")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        vacation_start = st.date_input(
            "Vacation Start",
            key=f"vacation_start_{selected_provider}"
        )
    
    with col2:
        vacation_end = st.date_input(
            "Vacation End",
            key=f"vacation_end_{selected_provider}"
        )
    
    with col3:
        if st.button("Add Vacation", key=f"add_vacation_{selected_provider}"):
            if vacation_start <= vacation_end:
                vacation = {
                    "start": vacation_start.isoformat(),
                    "end": vacation_end.isoformat()
                }
                if vacation not in provider_rules["vacations"]:
                    provider_rules["vacations"].append(vacation)
                    st.success(f"Added vacation from {vacation_start} to {vacation_end}")
                    st.rerun()
                else:
                    st.warning("Vacation period already exists")
            else:
                st.error("Start date must be before end date")
    
    # Display vacations
    if provider_rules["vacations"]:
        st.write("**Vacation Periods:**")
        for i, vacation in enumerate(provider_rules["vacations"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {vacation['start']} to {vacation['end']}")
            with col2:
                if st.button("Remove", key=f"remove_vacation_{selected_provider}_{i}"):
                    provider_rules["vacations"].pop(i)
                    st.rerun()
    
    # Save rules
    if st.button("💾 Save Rules", key=f"save_rules_{selected_provider}"):
        st.session_state.provider_rules[selected_provider] = provider_rules
        
        # Auto-save providers with updated rules
        from core.data_manager import save_providers
        save_providers(st.session_state.providers_df, st.session_state.provider_rules)
        
        st.success(f"Rules saved for {selected_provider}!")
