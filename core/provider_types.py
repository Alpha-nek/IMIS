# =============================================================================
# Provider Types and Configurations
# =============================================================================

from typing import Set, Dict, List
from models.constants import APP_PROVIDER_INITIALS

# Define provider type constants
NOCTURNISTS = {"JT", "OI", "AT", "CM", "YD", "RS"}
SENIORS = {"AA", "AD", "LN", "SM", "FS", "KA", "RR", "AM"}

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
    Get the shift types allowed for a provider based on their type.
    """
    provider_type = get_provider_type(provider)
    
    if provider_type == "APP":
        return ["APP"]
    elif provider_type == "NOCTURNIST":
        return ["N12", "NB"]
    elif provider_type == "SENIOR":
        return ["R12"]  # Seniors only do 7am rounding shifts
    else:
        return ["R12", "A12", "A10", "N12"]  # Regular providers can do all shifts except bridge shifts

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
