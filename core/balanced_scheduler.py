# =============================================================================
# Balanced Scheduling Algorithm - Simplified and Effective
# =============================================================================

import logging
import random
from typing import Dict, List, Optional
from datetime import date, timedelta
from models.data_models import SEvent, RuleConfig
from core.utils import count_shifts_on_date, get_adjusted_expected_shifts, is_provider_unavailable_on_date
from core.shift_validation import (
    validate_shift_type_preference, has_sufficient_rest, has_shift_on_date
)
from core.shift_creation import create_shift_event

logger = logging.getLogger(__name__)

def fill_remaining_shifts_balanced(month_days: List[date], providers: List[str], 
                                 shift_capacity: Dict[str, int], provider_rules: Dict, 
                                 global_rules: RuleConfig, provider_shifts: Dict,
                                 year: int = None, month: int = None) -> List[SEvent]:
    """
    SIMPLIFIED AND EFFECTIVE ALGORITHM: Fill shifts day by day with proper distribution.
    
    Algorithm:
    1. For each day, fill each shift type up to capacity
    2. Use smart provider selection based on availability, preferences, and load
    3. Ensure 2-day rest between shifts for the same provider
    4. Distribute shift types evenly across providers
    """
    events = []
    
    # Pre-calculate expected shifts for all providers
    provider_expected_shifts = {}
    for provider in providers:
        from models.constants import APP_PROVIDER_INITIALS
        if provider not in APP_PROVIDER_INITIALS:
            provider_expected_shifts[provider] = get_adjusted_expected_shifts(
                provider, year, month, provider_rules, global_rules
            )
    
    # Track shift type distribution for each provider
    provider_shift_counts = {provider: {"R12": 0, "A12": 0, "A10": 0, "N12": 0, "NB": 0, "APP": 0} 
                           for provider in providers}
    
    # Define shift priority (most important first)
    shift_priority = ["N12", "A12", "A10", "R12", "NB", "APP"]
    
    logger.info(f"Starting simplified algorithm for {len(providers)} providers over {len(month_days)} days")
    logger.info(f"Shift capacity: {shift_capacity}")
    
    # Process each day
    for day in month_days:
        logger.info(f"Processing day: {day}")
        
        # Fill each shift type for this day
        for shift_type in shift_priority:
            if shift_type not in shift_capacity:
                continue
                
            capacity = shift_capacity[shift_type]
            assigned_count = count_shifts_on_date(day, shift_type, provider_shifts)
            remaining_slots = capacity - assigned_count
            
            logger.info(f"  {shift_type}: {assigned_count}/{capacity} slots filled, {remaining_slots} remaining")
            
            # Fill remaining slots
            for slot in range(remaining_slots):
                best_provider = find_best_provider_for_day_shift(
                    day, shift_type, providers, provider_shifts, provider_rules,
                    global_rules, provider_expected_shifts, provider_shift_counts, year, month
                )
                
                if best_provider:
                    event = create_shift_event(best_provider, shift_type, day)
                    events.append(event)
                    provider_shifts[best_provider].append(event)
                    provider_shift_counts[best_provider][shift_type] += 1
                    
                    logger.info(f"    ✅ Assigned {shift_type} to {best_provider}")
                else:
                    logger.warning(f"    ❌ No provider available for {shift_type} on {day}")
    
    # Summary statistics
    providers_assigned = set()
    for event in events:
        provider = event.extendedProps.get("provider")
        if provider:
            providers_assigned.add(provider)
    
    logger.info(f"Algorithm completed:")
    logger.info(f"  Total events created: {len(events)}")
    logger.info(f"  Providers assigned: {len(providers_assigned)}/{len(providers)}")
    logger.info(f"  Unassigned providers: {set(providers) - providers_assigned}")
    
    # Log shift type distribution
    for shift_type in shift_priority:
        total_assigned = sum(counts[shift_type] for counts in provider_shift_counts.values())
        logger.info(f"  {shift_type} shifts assigned: {total_assigned}")
    
    return events

def find_best_provider_for_day_shift(day: date, shift_type: str, providers: List[str],
                                   provider_shifts: Dict, provider_rules: Dict,
                                   global_rules: RuleConfig, provider_expected_shifts: Dict,
                                   provider_shift_counts: Dict, year: int, month: int) -> Optional[str]:
    """
    Find the best provider for a specific shift type on a given day.
    SIMPLIFIED: Focus on availability, rest, and fair distribution.
    """
    from core.provider_types import get_provider_type, get_allowed_shift_types
    
    best_provider = None
    best_score = float('inf')
    available_providers = []
    
    for provider in providers:
        # Check if provider can do this shift type
        allowed_shifts = get_allowed_shift_types(provider)
        if shift_type not in allowed_shifts:
            continue
        
        # Basic availability checks
        if is_provider_unavailable_on_date(provider, day, provider_rules):
            continue
        
        if has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Check rest requirements (1 day minimum between shifts - more flexible)
        min_rest_days = 1
        if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
            min_rest_days = max(1, global_rules.min_days_between_shifts)
        
        if not has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts >= expected_shifts:
            continue
        
        # Provider is available - add to list
        available_providers.append(provider)
        
        # Calculate score (lower is better)
        score = 0
        
        # Factor 1: Current shift load (prefer providers with fewer shifts)
        score += (current_shifts / expected_shifts) * 10
        
        # Factor 2: Shift type distribution (prefer providers who haven't done this type much)
        total_shifts_for_type = provider_shift_counts[provider][shift_type]
        score += total_shifts_for_type * 2
        
        # Factor 3: Provider preference bonus
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        if shift_preferences.get(shift_type, False):
            score -= 3  # Significant bonus for preference
        
        # Factor 4: Random factor to avoid always picking the same provider
        score += random.uniform(0, 1)
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    # Log debugging information
    if not available_providers:
        logger.warning(f"No providers available for {shift_type} on {day}")
        logger.warning(f"Total providers checked: {len(providers)}")
        
        # Log why providers are not available
        for provider in providers[:5]:  # Check first 5 providers
            allowed_shifts = get_allowed_shift_types(provider)
            if shift_type not in allowed_shifts:
                logger.debug(f"  {provider}: shift type {shift_type} not in allowed types {allowed_shifts}")
            elif is_provider_unavailable_on_date(provider, day, provider_rules):
                logger.debug(f"  {provider}: unavailable on {day}")
            elif has_shift_on_date(provider, day, provider_shifts[provider]):
                logger.debug(f"  {provider}: already has shift on {day}")
            elif not has_sufficient_rest(provider, day, provider_shifts[provider], 1):
                logger.debug(f"  {provider}: insufficient rest")
            else:
                current_shifts = len(provider_shifts[provider])
                expected_shifts = provider_expected_shifts.get(provider, 15)
                if current_shifts >= expected_shifts:
                    logger.debug(f"  {provider}: at shift limit ({current_shifts}/{expected_shifts})")
    
    return best_provider

def create_provider_shift_blocks(providers: List[str], provider_expected_shifts: Dict,
                               month_days: List[date], provider_rules: Dict, 
                               global_rules: RuleConfig) -> Dict[str, List[Dict]]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return {}

def determine_shift_type_for_block(provider: str, provider_rules: Dict) -> str:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return "R12"

def assign_blocks_with_proper_spacing(provider_blocks: Dict[str, List[Dict]], month_days: List[date],
                                    shift_capacity: Dict[str, int], provider_rules: Dict,
                                    global_rules: RuleConfig, provider_shifts: Dict,
                                    year: int, month: int) -> List[SEvent]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return []

def find_available_dates_for_block_with_spacing(provider: str, shift_type: str, block_size: int,
                                              month_days: List[date], provider_shifts: Dict,
                                              provider_rules: Dict, global_rules: RuleConfig,
                                              year: int, month: int) -> List[date]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return []

def assign_shift_block(provider: str, shift_type: str, dates: List[date], provider_shifts: Dict) -> List[SEvent]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return []

def fill_remaining_gaps_individual(month_days: List[date], providers: List[str],
                                 shift_capacity: Dict[str, int], provider_rules: Dict,
                                 global_rules: RuleConfig, provider_shifts: Dict,
                                 provider_expected_shifts: Dict, year: int, month: int) -> List[SEvent]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return []

def find_best_provider_for_single_shift(day: date, shift_type: str, providers: List[str],
                                      provider_shifts: Dict, provider_rules: Dict,
                                      global_rules: RuleConfig, provider_expected_shifts: Dict,
                                      year: int, month: int) -> Optional[str]:
    """
    DEPRECATED: This function is no longer used with the simplified algorithm.
    """
    return None
