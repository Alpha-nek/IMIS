# =============================================================================
# Provider Management UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import date, datetime
from typing import List, Dict, Any
import os

from models.constants import PROVIDER_INITIALS_DEFAULT, APP_PROVIDER_INITIALS
from models.data_models import RuleConfig
from core.utils import _normalize_initials_list

def providers_panel():
    """Main providers management panel."""
    st.header("👥 Provider Management")
    
    # Provider loading section
    st.subheader("📥 Load Providers")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("➕ Add Providers", expanded=False):
            new_providers = st.text_area(
                "Enter provider initials (one per line or comma-separated)",
                placeholder="AA\nAD\nAM\nFS\n...",
                help="Enter provider initials, one per line or separated by commas"
            )
            
            if st.button("Add to current list"):
                if new_providers.strip():
                    # Parse input
                    lines = [line.strip() for line in new_providers.split('\n') if line.strip()]
                    new_list = []
                    for line in lines:
                        new_list.extend([x.strip() for x in line.split(',') if x.strip()])
                    
                    # Normalize and add
                    current_list = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
                    merged = _normalize_initials_list(current_list + new_list)
                    st.session_state.providers_df = pd.DataFrame({"initials": merged})
                    st.toast(f"Added {len(merged) - len(current_list)} new provider(s).", icon="✅")
    
    with col2:
        with st.expander("➖ Remove Providers", expanded=False):
            current_list = st.session_state.providers_df["initials"].astype(str).str.upper().tolist()
            to_remove = st.multiselect("Select providers to remove", options=current_list, key="rm_multi")
            if st.button("Remove selected", key="btn_rm"):
                if not to_remove:
                    st.info("No providers selected.")
                else:
                    remaining = [p for p in current_list if p not in set(to_remove)]
                    st.session_state.providers_df = pd.DataFrame({"initials": _normalize_initials_list(remaining)})
                    st.session_state["provider_caps"] = {k: v for k, v in st.session_state.provider_caps.items() if k in remaining}
                    st.toast(f"Removed {len(to_remove)} provider(s).", icon="🗑️")
    
    # Provider-specific rules
    st.subheader("⚙️ Provider-Specific Rules")
    provider_selector()
    provider_rules_panel()

def provider_selector():
    """Provider selection for rules."""
    if st.session_state.providers_df.empty:
        st.warning("No providers loaded.")
        return
    
    all_providers = sorted(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
    app_providers = sorted(APP_PROVIDER_INITIALS)
    
    # Create provider options with separators
    provider_options = ["(Select Provider)"]
    if all_providers:
        provider_options.append("--- Physicians ---")
        provider_options.extend(all_providers)
    if app_providers:
        provider_options.append("--- APPs ---")
        provider_options.extend(app_providers)
    
    selected_provider = st.selectbox("Select Provider for Rules", options=provider_options, key="provider_selector")
    
    if selected_provider != "(Select Provider)" and not selected_provider.startswith("---"):
        st.session_state.selected_provider = selected_provider
    else:
        st.session_state.selected_provider = None

def provider_rules_panel():
    """Panel for editing provider-specific rules."""
    if not st.session_state.get("selected_provider"):
        st.info("Please select a provider above to edit their rules.")
        return
    
    provider = st.session_state.selected_provider
    st.subheader(f"⚙️ Rules for {provider}")
    
    # Initialize provider rules if not exists
    if "provider_rules" not in st.session_state:
        st.session_state.provider_rules = {}
    
    if provider not in st.session_state.provider_rules:
        st.session_state.provider_rules[provider] = {
            "min_shifts": 15,
            "max_shifts": 16,
            "unavailable_dates": [],
            "vacations": [],
            "preferred_shifts": [],
            "blackout_dates": []
        }
    
    rules = st.session_state.provider_rules[provider]
    
    # Basic rules
    col1, col2 = st.columns(2)
    with col1:
        rules["min_shifts"] = st.number_input("Minimum Shifts", min_value=1, max_value=31, value=rules.get("min_shifts", 15), key=f"min_{provider}")
        rules["max_shifts"] = st.number_input("Maximum Shifts", min_value=1, max_value=31, value=rules.get("max_shifts", 16), key=f"max_{provider}")
    
    with col2:
        # Unavailable dates
        st.write("**Unavailable Dates**")
        unavailable_dates = rules.get("unavailable_dates", [])
        new_date = st.date_input("Add unavailable date", key=f"unavailable_{provider}")
        if st.button("Add Date", key=f"add_unavailable_{provider}"):
            if new_date not in unavailable_dates:
                unavailable_dates.append(new_date.isoformat())
                rules["unavailable_dates"] = unavailable_dates
                st.success("Date added!")
                st.rerun()
        
        # Show existing unavailable dates
        if unavailable_dates:
            st.write("Current unavailable dates:")
            for i, date_str in enumerate(unavailable_dates):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(date_str)
                with col2:
                    if st.button("Remove", key=f"remove_unavailable_{provider}_{i}"):
                        unavailable_dates.pop(i)
                        rules["unavailable_dates"] = unavailable_dates
                        st.success("Date removed!")
                        st.rerun()
    
    # Vacations
    st.write("**Vacations**")
    col1, col2 = st.columns(2)
    with col1:
        vacation_start = st.date_input("Vacation Start", key=f"vacation_start_{provider}")
        vacation_end = st.date_input("Vacation End", key=f"vacation_end_{provider}")
    
    with col2:
        if st.button("Add Vacation", key=f"add_vacation_{provider}"):
            if vacation_start < vacation_end:
                vacation = {
                    "start": vacation_start.isoformat(),
                    "end": vacation_end.isoformat()
                }
                vacations = rules.get("vacations", [])
                vacations.append(vacation)
                rules["vacations"] = vacations
                st.success("Vacation added!")
                st.rerun()
            else:
                st.error("End date must be after start date.")
    
    # Show existing vacations
    vacations = rules.get("vacations", [])
    if vacations:
        st.write("Current vacations:")
        for i, vacation in enumerate(vacations):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{vacation['start']} to {vacation['end']}")
            with col2:
                if st.button("Remove", key=f"remove_vacation_{provider}_{i}"):
                    vacations.pop(i)
                    rules["vacations"] = vacations
                    st.success("Vacation removed!")
                    st.rerun()
    
    # Save button
    if st.button("Save Rules", key=f"save_rules_{provider}"):
        st.session_state.provider_rules[provider] = rules
        st.success(f"Rules saved for {provider}!")

def load_providers_from_csv():
    """Load providers from CSV file."""
    try:
        if os.path.exists("IMIS_initials.csv"):
            providers_df = pd.read_csv("IMIS_initials.csv")
            providers_df = providers_df.dropna()
            providers_df["initials"] = providers_df["initials"].astype(str).str.strip().str.upper()
            providers_df = providers_df[providers_df["initials"] != ""]
            providers_df = providers_df[providers_df["initials"] != "nan"]
            providers_df = providers_df[providers_df["initials"] != "NO"]
            
            if not providers_df.empty:
                st.session_state["providers_df"] = providers_df
                st.session_state["providers_loaded"] = True
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        st.error(f"Failed to load providers: {e}")
        return False
