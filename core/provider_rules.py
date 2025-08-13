# =============================================================================
# Provider Rules Management
# =============================================================================

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def ensure_default_provider_rules(providers: List[str], provider_rules: Dict) -> Dict:
    """
    Ensure all providers have default rules initialized.
    REASONABLE DEFAULTS: Day shifts enabled by default, night shifts opt-in.
    """
    try:
        for provider in providers:
            if provider not in provider_rules:
                provider_rules[provider] = {
                    "shift_preferences": {
                        "R12": True,   # Default: Enable day shifts for all providers
                        "A12": True,   # Default: Enable day shifts for all providers
                        "A10": True,   # Default: Enable day shifts for all providers
                        "N12": False,  # Opt-in: Night shifts require explicit preference
                        "NB": False,   # Bridge shifts are not available to regular providers
                        "APP": False   # Opt-in: APP shifts require explicit preference
                    },
                    "vacations": [],
                    "unavailable_dates": []
                }
            elif not isinstance(provider_rules[provider], dict):
                # If provider rules exist but are not a dict, initialize them
                provider_rules[provider] = {
                    "shift_preferences": {
                        "R12": True,   # Default: Enable day shifts for all providers
                        "A12": True,   # Default: Enable day shifts for all providers
                        "A10": True,   # Default: Enable day shifts for all providers
                        "N12": False,  # Opt-in: Night shifts require explicit preference
                        "NB": False,   # Bridge shifts are not available to regular providers
                        "APP": False   # Opt-in: APP shifts require explicit preference
                    },
                    "vacations": [],
                    "unavailable_dates": []
                }
            else:
                # Ensure existing providers have all shift preferences defined
                if "shift_preferences" not in provider_rules[provider]:
                    provider_rules[provider]["shift_preferences"] = {}
                
                shift_prefs = provider_rules[provider]["shift_preferences"]
                # Set defaults for missing preferences
                if "R12" not in shift_prefs:
                    shift_prefs["R12"] = True  # Default to True for day shifts
                if "A12" not in shift_prefs:
                    shift_prefs["A12"] = True  # Default to True for day shifts
                if "A10" not in shift_prefs:
                    shift_prefs["A10"] = True  # Default to True for day shifts
                if "N12" not in shift_prefs:
                    shift_prefs["N12"] = False  # Default to False for night shifts
                if "NB" not in shift_prefs:
                    shift_prefs["NB"] = False  # Default to False for night shifts
                if "APP" not in shift_prefs:
                    shift_prefs["APP"] = False  # Default to False for APP shifts
        
        return provider_rules
    except Exception as e:
        logger.error(f"Error ensuring default provider rules: {e}")
        return provider_rules
