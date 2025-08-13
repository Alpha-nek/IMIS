# =============================================================================
# Provider Types and Configurations
# =============================================================================

from typing import Set, Dict, List
from models.constants import APP_PROVIDER_INITIALS

# Define provider type constants
NOCTURNISTS = {"JT", "OI", "AT", "CM", "YD", "RS"}
SENIORS = {"AA", "AD", "LN", "SM", "FS", "KA", "RR", "AM"}

# Bridge-qualified providers (only these can take NB shifts)
BRIDGE_QUALIFIED = {"RJ", "AT", "EB", "YH", "CM", "MS", "DI"}

def get_provider_type(provider: str) -> str:
    """
    Get the type of a provider.
    """
    if provider in APP_PROVIDER_INITIALS:
        return "APP"
    elif provider in NOCTURNISTS:
        return "NOCTURNIST"
    elif provider in SENIORS:
        return "SENIOR"
    else:
        return "REGULAR"

def get_allowed_shift_types(provider: str) -> List[str]:
    """
    Get the shift types allowed for a provider based on their type and qualifications.
    """
    provider_type = get_provider_type(provider)
    
    if provider_type == "APP":
        # APPs can ONLY take APP shifts
        return ["APP"]
    elif provider_type == "SENIOR":
        # Seniors ONLY take 7am rounding shifts (R12)
        return ["R12"]
    else:
        # Regular providers and nocturnists
        allowed = []
        
        # All non-APP, non-senior providers can do day and night shifts
        if provider_type == "NOCTURNIST" or provider in NOCTURNISTS:
            # Nocturnists prefer nights but can do day shifts if needed
            allowed.extend(["N12", "R12", "A12", "A10"])
        else:
            # Regular providers can do all except APP
            allowed.extend(["R12", "A12", "A10", "N12"])
        
        # Bridge shifts (NB) - only bridge-qualified providers
        if provider in BRIDGE_QUALIFIED:
            allowed.append("NB")
        
        return allowed

def is_bridge_qualified(provider: str) -> bool:
    """
    Check if a provider is qualified to take bridge shifts.
    """
    return provider in BRIDGE_QUALIFIED

def is_senior_provider(provider: str) -> bool:
    """
    Check if a provider is a senior provider.
    """
    return provider in SENIORS

def add_senior_provider(provider: str) -> None:
    """
    Add a provider to the senior providers list.
    """
    SENIORS.add(provider)

def remove_senior_provider(provider: str) -> None:
    """
    Remove a provider from the senior providers list.
    """
    SENIORS.discard(provider)

def get_senior_providers() -> Set[str]:
    """
    Get the current list of senior providers.
    """
    return SENIORS.copy()

def set_senior_providers(providers: List[str]) -> None:
    """
    Set the list of senior providers.
    """
    global SENIORS
    SENIORS = set(providers)
