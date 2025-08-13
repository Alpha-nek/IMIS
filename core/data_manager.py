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

def load_provider_names() -> Dict[str, str]:
    """Load provider full names from the text file."""
    provider_names = {}
    
    # Try to load from the text file
    try:
        if os.path.exists("provider full name.txt"):
            with open("provider full name.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 6:  # Ensure we have enough columns
                            initials = parts[0].strip().upper()
                            provider_type = parts[2].strip()  # MD, NP, PA
                            full_name = parts[5].strip()  # Full name is in column 6
                            
                            if initials and full_name and initials != "Moonlighter":
                                provider_names[initials] = full_name
    except Exception as e:
        print(f"Error loading provider names: {e}")
    
    return provider_names

def get_default_providers() -> pd.DataFrame:
    """Get default provider data with actual full names from the text file."""
    # Load actual provider names
    provider_names = load_provider_names()
    
    # Define the current active providers from IMIS_initials.csv
    # All physicians from the CSV are active
    active_providers = [
        # Physicians from IMIS_initials.csv
        {"initials": "AA", "type": "Physician"},
        {"initials": "AD", "type": "Physician"},
        {"initials": "AM", "type": "Physician"},
        {"initials": "FS", "type": "Physician"},
        {"initials": "JM", "type": "Physician"},
        {"initials": "JT", "type": "Physician"},
        {"initials": "KA", "type": "Physician"},
        {"initials": "LN", "type": "Physician"},
        {"initials": "SM", "type": "Physician"},
        {"initials": "OI", "type": "Physician"},
        {"initials": "NP", "type": "Physician"},
        {"initials": "PR", "type": "Physician"},
        {"initials": "UN", "type": "Physician"},
        {"initials": "DP", "type": "Physician"},
        {"initials": "FY", "type": "Physician"},
        {"initials": "YL", "type": "Physician"},
        {"initials": "RR", "type": "Physician"},
        {"initials": "SD", "type": "Physician"},
        {"initials": "JK", "type": "Physician"},
        {"initials": "NS", "type": "Physician"},
        {"initials": "PD", "type": "Physician"},
        {"initials": "AB", "type": "Physician"},
        {"initials": "KF", "type": "Physician"},
        {"initials": "AL", "type": "Physician"},
        {"initials": "GB", "type": "Physician"},
        {"initials": "KD", "type": "Physician"},
        {"initials": "NG", "type": "Physician"},
        {"initials": "GI", "type": "Physician"},
        {"initials": "VT", "type": "Physician"},
        {"initials": "DI", "type": "Physician"},
        {"initials": "YD", "type": "Physician"},
        {"initials": "HS", "type": "Physician"},
        {"initials": "YA", "type": "Physician"},
        {"initials": "NM", "type": "Physician"},
        {"initials": "EM", "type": "Physician"},
        {"initials": "SS", "type": "Physician"},
        {"initials": "YS", "type": "Physician"},
        {"initials": "HW", "type": "Physician"},
        {"initials": "AH", "type": "Physician"},
        {"initials": "RJ", "type": "Physician"},
        {"initials": "SI", "type": "Physician"},
        {"initials": "FH", "type": "Physician"},
        {"initials": "EB", "type": "Physician"},
        {"initials": "RS", "type": "Physician"},
        {"initials": "RG", "type": "Physician"},
        {"initials": "CJ", "type": "Physician"},
        {"initials": "MS", "type": "Physician"},
        {"initials": "AT", "type": "Physician"},
        {"initials": "YH", "type": "Physician"},
        {"initials": "XL", "type": "Physician"},
        {"initials": "MA", "type": "Physician"},
        {"initials": "LM", "type": "Physician"},
        {"initials": "MQ", "type": "Physician"},
        {"initials": "CM", "type": "Physician"},
        {"initials": "AI", "type": "Physician"},
        # APPs (keeping the current active ones)
        {"initials": "JA", "type": "APP"},
        {"initials": "DN", "type": "APP"},
        {"initials": "KP", "type": "APP"},
        {"initials": "AR", "type": "APP"},
        {"initials": "JL", "type": "APP"},
    ]
    
    # Create provider list with actual names
    default_providers = []
    for provider in active_providers:
        initials = provider["initials"]
        # Check if we have the real name, otherwise use fallback
        if initials in provider_names:
            full_name = provider_names[initials]
        else:
            full_name = f"Dr. {initials}" if provider["type"] == "Physician" else f"APP {initials}"
        
        default_providers.append({
            "initials": initials,
            "name": full_name,
            "type": provider["type"]
        })
    
    return pd.DataFrame(default_providers)

def save_providers(providers_df: pd.DataFrame, provider_rules: Dict) -> None:
    """Save providers and their rules to JSON file."""
    try:
        ensure_data_directory()
        
        data = {
            "providers": providers_df.to_dict('records') if not providers_df.empty else [],
            "provider_rules": provider_rules,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(PROVIDERS_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save providers: {e}")
        raise FileOperationError(f"Failed to save providers: {e}")

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

def save_rules(global_rules: Any, shift_types: List, shift_capacity: Dict, provider_rules: Dict = None) -> None:
    """Save global rules, shift configuration, and provider rules to JSON file."""
    ensure_data_directory()
    
    data = {
        "global_rules": global_rules.dict() if hasattr(global_rules, 'dict') else global_rules.__dict__,
        "shift_types": shift_types,
        "shift_capacity": shift_capacity,
        "provider_rules": provider_rules or {},
        "last_updated": datetime.now().isoformat()
    }
    
    with open(RULES_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_rules() -> tuple[Dict, List, Dict, Dict]:
    """Load global rules, shift configuration, and provider rules from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(RULES_FILE):
        # Return defaults if file doesn't exist
        return {}, [], {}, {}
    
    try:
        with open(RULES_FILE, 'r') as f:
            data = json.load(f)
        
        global_rules = data.get("global_rules", {})
        shift_types = data.get("shift_types", [])
        shift_capacity = data.get("shift_capacity", {})
        provider_rules = data.get("provider_rules", {})
        
        return global_rules, shift_types, shift_capacity, provider_rules
    except Exception as e:
        print(f"Error loading rules: {e}")
        return {}, [], {}, {}

def save_schedule(year: int, month: int, events: List[Any]) -> None:
    """Save schedule for a specific month to JSON file."""
    ensure_data_directory()
    
    # Convert SEvent objects to dictionaries for JSON serialization
    events_dict = []
    for event in events:
        if hasattr(event, 'to_json_event'):
            # It's an SEvent object
            events_dict.append(event.to_json_event())
        elif isinstance(event, dict):
            # It's already a dictionary
            events_dict.append(event)
        else:
            # Convert string representation back to dict if possible
            events_dict.append(str(event))
    
    # Load existing schedules
    schedules = load_all_schedules()
    
    # Update the specific month's schedule
    month_key = f"{year}-{month:02d}"
    schedules[month_key] = {
        "events": events_dict,
        "year": year,
        "month": month,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save all schedules
    with open(SCHEDULES_FILE, 'w') as f:
        json.dump(schedules, f, indent=2)

def load_schedule(year: int, month: int) -> List[Any]:
    """Load schedule for a specific month from JSON file."""
    ensure_data_directory()
    
    if not os.path.exists(SCHEDULES_FILE):
        return []
    
    try:
        with open(SCHEDULES_FILE, 'r') as f:
            schedules = json.load(f)
        
        month_key = f"{year}-{month:02d}"
        if month_key in schedules:
            events_data = schedules[month_key].get("events", [])
            
            # Convert loaded data back to SEvent objects
            events = []
            for event_data in events_data:
                if isinstance(event_data, dict):
                    # Convert datetime strings back to datetime objects
                    if 'start' in event_data and isinstance(event_data['start'], str):
                        event_data['start'] = datetime.fromisoformat(event_data['start'])
                    if 'end' in event_data and isinstance(event_data['end'], str):
                        event_data['end'] = datetime.fromisoformat(event_data['end'])
                    
                    # Create SEvent object
                    from models.data_models import SEvent
                    events.append(SEvent(**event_data))
                else:
                    # Keep as is if it's not a dict
                    events.append(event_data)
            
            return events
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
                st.session_state.get('shift_capacity', {}),
                st.session_state.get('provider_rules', {}) # Pass provider_rules to save_rules
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
        providers_df, provider_rules_from_file = load_providers()
        if not providers_df.empty:
            st.session_state.providers_df = providers_df
            st.session_state.providers_loaded = True
        
        # Load rules
        global_rules_dict, shift_types, shift_capacity, provider_rules_from_rules = load_rules()
        if global_rules_dict:
            from models.data_models import RuleConfig
            st.session_state.global_rules = RuleConfig(**global_rules_dict)
        if shift_types:
            st.session_state.shift_types = shift_types
        if shift_capacity:
            st.session_state.shift_capacity = shift_capacity
        
        # Merge provider rules (rules file takes precedence)
        merged_provider_rules = {**provider_rules_from_file, **provider_rules_from_rules}
        if merged_provider_rules:
            st.session_state.provider_rules = merged_provider_rules
        
        # Load current month's schedule
        current_year = st.session_state.get('current_year', datetime.now().year)
        current_month = st.session_state.get('current_month', datetime.now().month)
        events = load_schedule(current_year, current_month)
        if events:
            st.session_state.events = events
            
    except Exception as e:
        print(f"Error in auto load: {e}")
