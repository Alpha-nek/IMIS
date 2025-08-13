# =============================================================================
# Shift Validation and Rule Checking
# =============================================================================

import logging
from typing import Dict, List, Optional
from datetime import date, timedelta
from models.data_models import RuleConfig, SEvent
from core.utils import is_provider_unavailable_on_date
from core.provider_types import get_allowed_shift_types, get_provider_type

logger = logging.getLogger(__name__)

def validate_shift_type_preference(provider: str, shift_type: str, provider_rules: Dict) -> bool:
    """
    Validate if a provider has preference for a specific shift type.
    LESS RESTRICTIVE: If no preferences are set, allow any shift type.
    """
    try:
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        # If no preferences are set, allow any shift type
        if not shift_preferences:
            return True
        
        # If preferences are set, check if this shift type is preferred
        return shift_preferences.get(shift_type, False)
    except Exception as e:
        logger.error(f"Error validating shift type preference: {e}")
        return True  # Default to allowing the shift if there's an error

def validate_provider_role_shift_type(provider: str, shift_type: str) -> bool:
    """
    Validate if a provider can do a specific shift type based on their role.
    """
    try:
        allowed_types = get_allowed_shift_types(provider)
        return shift_type in allowed_types
    except Exception as e:
        logger.error(f"Error validating provider role shift type: {e}")
        return False

def has_sufficient_rest(provider: str, target_date: date, provider_shifts: List[SEvent], min_rest_days: int = 2) -> bool:
    """
    Check if provider has sufficient rest days before the target date.
    ENFORCES STRICT REST RULES: Must have min_rest_days between any shifts.
    """
    try:
        # Get all shift dates for this provider
        shift_dates = []
        for shift in provider_shifts:
            if hasattr(shift, 'start'):
                shift_date = shift.start.date()
            elif isinstance(shift, dict) and 'start' in shift:
                from datetime import datetime
                shift_date = datetime.fromisoformat(shift['start']).date()
            else:
                continue
            
            shift_dates.append(shift_date)
        
        if not shift_dates:
            return True  # No previous shifts, so rest is sufficient
        
        # Sort dates to find the most recent shift
        shift_dates.sort()
        last_shift_date = shift_dates[-1]
        
        # Calculate days between last shift and target date
        days_between = (target_date - last_shift_date).days
        
        # STRICT ENFORCEMENT: Must have exactly min_rest_days or more
        if days_between < min_rest_days:
            logger.debug(f"Provider {provider} needs {min_rest_days} days rest, but only has {days_between} days")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking sufficient rest: {e}")
        return False

def has_shift_on_date(provider: str, target_date: date, provider_shifts: List[SEvent]) -> bool:
    """
    Check if provider already has a shift on the target date.
    """
    try:
        for shift in provider_shifts:
            if hasattr(shift, 'start'):
                shift_date = shift.start.date()
            elif isinstance(shift, dict) and 'start' in shift:
                from datetime import datetime
                shift_date = datetime.fromisoformat(shift['start']).date()
            else:
                continue
            
            if shift_date == target_date:
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking shift on date: {e}")
        return False

def validate_block_size(block_size: int, max_size: int = 7) -> int:
    """
    Validate and adjust block size to ensure it doesn't exceed maximum.
    """
    return min(block_size, max_size)

def validate_all_hard_rules(provider: str, day: date, shift_type: str, provider_shifts: Dict,
                           provider_rules: Dict, global_rules: RuleConfig, 
                           provider_expected_shifts: Dict) -> bool:
    """
    Validate ALL hard rules for a provider taking a specific shift.
    ULTRA-STRICT: Returns True only if ALL rules are satisfied.
    """
    try:
        # HARD RULE 1: Shift type preference
        if not validate_shift_type_preference(provider, shift_type, provider_rules):
            logger.debug(f"Provider {provider} fails shift type preference check for {shift_type}")
            return False
        
        # HARD RULE 2: Provider role validation
        if not validate_provider_role_shift_type(provider, shift_type):
            logger.debug(f"Provider {provider} fails role validation for {shift_type}")
            return False
        
        # HARD RULE 3: Expected shifts limit - ULTRA-STRICT
        current_shifts = len(provider_shifts.get(provider, []))
        from models.constants import APP_PROVIDER_INITIALS
        if provider in APP_PROVIDER_INITIALS:
            # APP providers: HARD STOP at 12 shifts
            if current_shifts >= 12:
                logger.warning(f"ULTRA-STRICT: APP provider {provider} at {current_shifts} shifts - BLOCKED")
                return False
        else:
            # Regular providers: HARD STOP at expected shifts
            expected_shifts = provider_expected_shifts.get(provider, 15)
            if current_shifts >= expected_shifts:
                logger.warning(f"ULTRA-STRICT: Provider {provider} at {current_shifts} shifts (expected {expected_shifts}) - BLOCKED")
                return False
        
        # HARD RULE 4: Rest requirements
        min_rest_days = 2
        if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
            min_rest_days = max(2, global_rules.min_days_between_shifts)
        
        if not has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
            logger.debug(f"Provider {provider} fails rest requirements check")
            return False
        
        logger.debug(f"Provider {provider} passes ALL hard rules for {shift_type}")
        return True
    except Exception as e:
        logger.error(f"Error validating hard rules: {e}")
        return False
