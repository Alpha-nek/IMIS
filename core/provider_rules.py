# =============================================================================
# Provider Rules Management
# =============================================================================

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def ensure_default_provider_rules(providers: List[str], provider_rules: Dict) -> Dict:
    """
    Ensure all providers have default rules set.
    """
    from models.constants import APP_PROVIDER_INITIALS
    from core.provider_types import NOCTURNISTS, SENIORS
    
    for provider in providers:
        if provider not in provider_rules:
            provider_rules[provider] = {}
        
        # Set defaults based on provider type
        if provider in APP_PROVIDER_INITIALS:
            # APP providers - no shift limits, only APP shifts
            provider_rules[provider].setdefault("shift_preferences", {
                "APP": True,
                "R12": False,
                "A12": False,
                "A10": False,
                "N12": False,
                "NB": False
            })
            provider_rules[provider].setdefault("max_nights", 0)  # APP providers don't do nights
        elif provider in NOCTURNISTS:
            # Nocturnists - only night shifts
            provider_rules[provider].setdefault("shift_preferences", {
                "APP": False,
                "R12": False,
                "A12": False,
                "A10": False,
                "N12": True,
                "NB": True
            })
            provider_rules[provider].setdefault("max_nights", 15)  # High night limit for nocturnists
        elif provider in SENIORS:
            # Seniors - only rounding shifts
            provider_rules[provider].setdefault("shift_preferences", {
                "APP": False,
                "R12": True,
                "A12": False,
                "A10": False,
                "N12": False,
                "NB": False
            })
            provider_rules[provider].setdefault("max_nights", 0)  # Seniors don't do nights
        else:
            # Regular providers - can do all shifts except APP
            provider_rules[provider].setdefault("shift_preferences", {
                "APP": False,
                "R12": True,
                "A12": True,
                "A10": True,
                "N12": True,
                "NB": False  # Regular providers don't do bridge shifts
            })
            provider_rules[provider].setdefault("max_nights", 3)  # Default 3 nights/month for regular providers
        
        # Set other defaults
        provider_rules[provider].setdefault("fte_percentage", 1.0)
        provider_rules[provider].setdefault("min_rest_days", 1)
        provider_rules[provider].setdefault("unavailable_days_of_week", [])
        provider_rules[provider].setdefault("shift_timing_preference", None)
        provider_rules[provider].setdefault("is_senior", provider in SENIORS)
    
    return provider_rules
