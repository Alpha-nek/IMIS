<<<<<<< HEAD
"""
Data Manager for IMIS Scheduler
Handles automatic saving and loading of app data
"""
import json
import os
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import streamlit as st

# Data file paths
DATA_DIR = "data"
PROVIDERS_FILE = os.path.join(DATA_DIR, "providers.json")
RULES_FILE = os.path.join(DATA_DIR, "rules.json")
SCHEDULES_FILE = os.path.join(DATA_DIR, "schedules.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

def ensure_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_providers(providers_df: pd.DataFrame, provider_rules: Dict) -> None:
    """Save providers and their rules to JSON file."""
    ensure_data_directory()
    
    data = {
        "providers": providers_df.to_dict('records') if not providers_df.empty else [],
        "provider_rules": provider_rules,
        "last_updated": datetime.now().isoformat()
    }
    
    with open(PROVIDERS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_providers() -> tuple[pd.DataFrame, Dict]:
    """Load providers and their rules from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(PROVIDERS_FILE):
        # Return default providers if file doesn't exist
        return pd.DataFrame(), {}
    
    try:
        with open(PROVIDERS_FILE, 'r') as f:
            data = json.load(f)
        
        providers_df = pd.DataFrame(data.get("providers", []))
        provider_rules = data.get("provider_rules", {})
        
        return providers_df, provider_rules
    except Exception as e:
        print(f"Error loading providers: {e}")
        return pd.DataFrame(), {}

def save_rules(global_rules: Any, shift_types: List, shift_capacity: Dict) -> None:
    """Save global rules and shift configuration to JSON file."""
    ensure_data_directory()
    
    data = {
        "global_rules": global_rules.dict() if hasattr(global_rules, 'dict') else global_rules.__dict__,
        "shift_types": shift_types,
        "shift_capacity": shift_capacity,
        "last_updated": datetime.now().isoformat()
    }
    
    with open(RULES_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_rules() -> tuple[Dict, List, Dict]:
    """Load global rules and shift configuration from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(RULES_FILE):
        # Return defaults if file doesn't exist
        return {}, [], {}
    
    try:
        with open(RULES_FILE, 'r') as f:
            data = json.load(f)
        
        global_rules = data.get("global_rules", {})
        shift_types = data.get("shift_types", [])
        shift_capacity = data.get("shift_capacity", {})
        
        return global_rules, shift_types, shift_capacity
    except Exception as e:
        print(f"Error loading rules: {e}")
        return {}, [], {}

def save_schedule(year: int, month: int, events: List[Dict]) -> None:
    """Save schedule for a specific month to JSON file."""
    ensure_data_directory()
    
    # Load existing schedules
    schedules = load_all_schedules()
    
    # Update the specific month's schedule
    month_key = f"{year}-{month:02d}"
    schedules[month_key] = {
        "events": events,
        "year": year,
        "month": month,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save all schedules
    with open(SCHEDULES_FILE, 'w') as f:
        json.dump(schedules, f, indent=2, default=str)

def load_schedule(year: int, month: int) -> List[Dict]:
    """Load schedule for a specific month from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(SCHEDULES_FILE):
        return []
    
    try:
        with open(SCHEDULES_FILE, 'r') as f:
            schedules = json.load(f)
        
        month_key = f"{year}-{month:02d}"
        if month_key in schedules:
            return schedules[month_key].get("events", [])
        return []
    except Exception as e:
        print(f"Error loading schedule: {e}")
        return []

def load_all_schedules() -> Dict:
    """Load all saved schedules."""
    ensure_data_directory()
    
    if not os.path.exists(SCHEDULES_FILE):
        return {}
    
    try:
        with open(SCHEDULES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading all schedules: {e}")
        return {}

def save_settings(settings: Dict) -> None:
    """Save app settings to JSON file."""
    ensure_data_directory()
    
    data = {
        "settings": settings,
        "last_updated": datetime.now().isoformat()
    }
    
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_settings() -> Dict:
    """Load app settings from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(SETTINGS_FILE):
        return {}
    
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        return data.get("settings", {})
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}

def get_default_providers() -> pd.DataFrame:
    """Get default provider data that was built into the original app."""
    default_providers = [
        {"initials": "AA", "name": "Dr. A. A.", "type": "Physician"},
        {"initials": "AD", "name": "Dr. A. D.", "type": "Physician"},
        {"initials": "AI", "name": "Dr. A. I.", "type": "Physician"},
        {"initials": "JA", "name": "APP A.", "type": "APP"},
        {"initials": "DN", "name": "APP D.", "type": "APP"},
        {"initials": "KP", "name": "APP K.", "type": "APP"},
        {"initials": "AR", "name": "APP A.", "type": "APP"},
        {"initials": "JL", "name": "APP J.", "type": "APP"},
    ]
    return pd.DataFrame(default_providers)

def initialize_default_data():
    """Initialize default data if no saved data exists."""
    ensure_data_directory()
    
    # Initialize default providers if no providers file exists
    if not os.path.exists(PROVIDERS_FILE):
        default_providers = get_default_providers()
        save_providers(default_providers, {})
    
    # Initialize default rules if no rules file exists
    if not os.path.exists(RULES_FILE):
        from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY
        from models.data_models import RuleConfig
        
        default_rules = RuleConfig()
        save_rules(default_rules, DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY)

def auto_save_session_state():
    """Automatically save current session state to files."""
    try:
        # Save providers if they exist
        if hasattr(st.session_state, 'providers_df') and not st.session_state.providers_df.empty:
            save_providers(st.session_state.providers_df, st.session_state.get('provider_rules', {}))
        
        # Save rules if they exist
        if hasattr(st.session_state, 'global_rules'):
            save_rules(
                st.session_state.global_rules,
                st.session_state.get('shift_types', []),
                st.session_state.get('shift_capacity', {})
            )
        
        # Save current schedule if it exists
        if hasattr(st.session_state, 'events') and st.session_state.events:
            current_year = st.session_state.get('current_year', datetime.now().year)
            current_month = st.session_state.get('current_month', datetime.now().month)
            save_schedule(current_year, current_month, st.session_state.events)
            
    except Exception as e:
        print(f"Error in auto save: {e}")

def auto_load_session_state():
    """Automatically load data from files into session state."""
    try:
        # Load providers
        providers_df, provider_rules = load_providers()
        if not providers_df.empty:
            st.session_state.providers_df = providers_df
            st.session_state.provider_rules = provider_rules
            st.session_state.providers_loaded = True
        
        # Load rules
        global_rules_dict, shift_types, shift_capacity = load_rules()
        if global_rules_dict:
            from models.data_models import RuleConfig
            st.session_state.global_rules = RuleConfig(**global_rules_dict)
        if shift_types:
            st.session_state.shift_types = shift_types
        if shift_capacity:
            st.session_state.shift_capacity = shift_capacity
        
        # Load current month's schedule
        current_year = st.session_state.get('current_year', datetime.now().year)
        current_month = st.session_state.get('current_month', datetime.now().month)
        events = load_schedule(current_year, current_month)
        if events:
            st.session_state.events = events
            
    except Exception as e:
        print(f"Error in auto load: {e}")
=======
#placeholder
>>>>>>> 688115275780d875a519276c05565fdf07cb61b6
