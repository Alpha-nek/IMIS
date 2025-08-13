# =============================================================================
# Provider Rules Management
# =============================================================================

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def ensure_default_provider_rules(providers: List[str], provider_rules: Dict) -> Dict:
    """
    Ensure all providers have default rules set based on clinical requirements.
    """
    from models.constants import APP_PROVIDER_INITIALS
    from core.provider_types import NOCTURNISTS, SENIORS, BRIDGE_QUALIFIED
    
    for provider in providers:
        if provider not in provider_rules:
            provider_rules[provider] = {}
        
        # Set defaults based on provider type and clinical requirements
        if provider in APP_PROVIDER_INITIALS:
            # APP providers - ONLY APP shifts, cannot take any other shifts
            provider_rules[provider].setdefault("shift_preferences", {
                "APP": True,
                "R12": False,
                "A12": False,
                "A10": False,
                "N12": False,
                "NB": False
            })
            provider_rules[provider].setdefault("max_nights", 0)  # APP providers don't do nights
        elif provider in SENIORS:
            # Seniors - ONLY 7am rounding shifts (R12)
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
            # Regular providers and nocturnists
            # Bridge qualification determines NB eligibility
            bridge_qualified = provider in BRIDGE_QUALIFIED
            
            if provider in NOCTURNISTS:
                # Nocturnists - prefer nights but can do day shifts
                provider_rules[provider].setdefault("shift_preferences", {
                    "APP": False,
                    "R12": True,
                    "A12": True,
                    "A10": True,
                    "N12": True,
                    "NB": bridge_qualified  # Only if bridge qualified
                })
                provider_rules[provider].setdefault("max_nights", 10)  # Higher night limit for nocturnists
            else:
                # Regular providers - can do all except APP and NB (unless bridge qualified)
                provider_rules[provider].setdefault("shift_preferences", {
                    "APP": False,
                    "R12": True,
                    "A12": True,
                    "A10": True,
                    "N12": True,
                    "NB": bridge_qualified  # Only if bridge qualified
                })
                provider_rules[provider].setdefault("max_nights", 4)  # Regular night limit
        
        # Set other defaults
        provider_rules[provider].setdefault("fte_percentage", 1.0)
        provider_rules[provider].setdefault("min_rest_days", 1)
        provider_rules[provider].setdefault("unavailable_days_of_week", [])
        provider_rules[provider].setdefault("shift_timing_preference", None)
        provider_rules[provider].setdefault("is_senior", provider in SENIORS)
    
    return provider_rules
