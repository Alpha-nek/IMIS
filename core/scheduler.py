# =============================================================================
# Core Scheduling Logic for IMIS Scheduler
# =============================================================================

import random
import uuid
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from collections import defaultdict

from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS
from models.data_models import RuleConfig, Provider, SEvent
from core.utils import (
    is_holiday, get_holiday_adjusted_capacity, parse_time, 
    date_range, month_start_end, make_month_days,
    _expand_vacation_dates, is_provider_unavailable_on_date, count_shifts_on_date
)
from core.exceptions import (
    ScheduleGenerationError, ProviderError, RuleValidationError, 
    ShiftAssignmentError, ValidationError, DataValidationError
)
from core.provider_types import NOCTURNISTS, SENIORS, get_provider_type, get_allowed_shift_types
from core.gap_analysis import analyze_admitting_gaps, analyze_rounding_gaps, find_consecutive_gaps
from core.block_assignment import (
    create_admitting_to_rounding_block, 
    create_rounding_block
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def generate_schedule(year: int, month: int, providers: List[str], 
                     shift_types: List[Dict], shift_capacity: Dict[str, int],
                     provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Generate a complete schedule following the ground rules:
    1. APPs only do APP shifts
    2. Providers prefer same shift type in blocks
    3. Rounders start with 1-2 admitting shifts before rounding
    4. Shift blocks of 3-7 shifts
    5. Nocturnists only do night shifts
    6. Optimize shift distribution to fill all slots
    """
    try:
        # Ensure provider_rules is a dictionary
        if not isinstance(provider_rules, dict):
            provider_rules = {}
        
        # Initialize default rules for any missing providers
        provider_rules = _ensure_default_provider_rules(providers, provider_rules)
        
        # Add some randomness for variety
        random.seed(datetime.now().timestamp())
        
        events = assign_advanced(year, month, providers, shift_types, shift_capacity, 
                               provider_rules, global_rules)
        
        return events
    except Exception as e:
        logger.error(f"Error generating schedule: {e}")
        raise ScheduleGenerationError(f"Failed to generate schedule: {e}")

def _ensure_default_provider_rules(providers: List[str], provider_rules: Dict) -> Dict:
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

def assign_advanced(year: int, month: int, providers: List[str], 
                   shift_types: List[Dict], shift_capacity: Dict[str, int],
                   provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    BALANCED assignment algorithm following ground rules.
    Uses a combination of block assignment and greedy filling to ensure good coverage.
    """
    try:
        import streamlit as st
        
        # Validate inputs
        if not providers:
            logger.warning("No providers provided for schedule generation")
            return []
        
        if not isinstance(provider_rules, dict):
            logger.warning("Provider rules is not a dictionary, initializing empty dict")
            provider_rules = {}
        
        # Initialize
        events = []
        month_days = make_month_days(year, month)
        
        # Separate providers by type
        app_providers = [p for p in providers if p in APP_PROVIDER_INITIALS]
        nocturnists = [p for p in providers if p in NOCTURNISTS]
        seniors = [p for p in providers if p in SENIORS]
        regular_providers = [p for p in providers if p not in APP_PROVIDER_INITIALS and p not in NOCTURNISTS and p not in SENIORS]
        
        # Track provider assignments
        provider_shifts = {p: [] for p in providers}
        
        # Step 1: Assign APP shifts first (they have specific rules)
        app_events = assign_app_shifts(month_days, app_providers, shift_capacity, 
                                      provider_rules, global_rules)
        events.extend(app_events)
        
        # Update provider tracking
        for event in app_events:
            provider = event.extendedProps.get("provider")
            if provider:
                provider_shifts[provider].append(event)
        
        # Step 2: Assign night shifts to nocturnists in blocks
        night_events = assign_night_shifts_to_nocturnists(month_days, nocturnists, 
                                                         shift_capacity, provider_rules, 
                                                         global_rules, provider_shifts, year, month)
        events.extend(night_events)
        
        # Step 3: Assign senior providers to rounding shifts only
        senior_events = assign_senior_rounding_shifts(month_days, seniors, shift_capacity,
                                                    provider_rules, global_rules, provider_shifts, year, month)
        events.extend(senior_events)
        
        # Step 4: Fill remaining shifts using a balanced approach
        remaining_events = fill_remaining_shifts_balanced(month_days, regular_providers, shift_capacity,
                                                        provider_rules, global_rules, provider_shifts, year, month)
        events.extend(remaining_events)
        
        # Step 5: Validate and auto-adjust schedule
        events = validate_and_adjust_schedule(events, providers, provider_rules, global_rules, year, month)
        
        return events
    except Exception as e:
        logger.error(f"Error in assign_advanced: {e}")
        raise ScheduleGenerationError(f"Failed to assign shifts: {e}")

def assign_night_shifts_to_nocturnists(month_days: List[date], nocturnists: List[str], 
                                      shift_capacity: Dict[str, int], provider_rules: Dict, 
                                      global_rules: RuleConfig, provider_shifts: Dict, 
                                      year: int, month: int) -> List[SEvent]:
    """
    Assign night shifts to nocturnists in blocks of 3-7 shifts.
    """
    events = []
    night_shift_types = ["N12", "NB"]
    
    # Create night shift blocks for each nocturnist
    for nocturnist in nocturnists:
        # Determine target number of night shifts for this nocturnist (adjusted for vacation)
        from core.utils import get_adjusted_expected_shifts
        target_shifts = get_adjusted_expected_shifts(nocturnist, year, month, provider_rules, global_rules)
        
        # Create blocks of 3-7 shifts (HARD RULE: max 7 shifts per block)
        remaining_shifts = target_shifts
        while remaining_shifts > 0:
            # Determine block size (3-7 shifts, but no more than remaining shifts)
            max_block_size = min(7, remaining_shifts)
            if max_block_size < 3:
                # If remaining shifts is less than 3, use all remaining shifts
                block_size = remaining_shifts
            else:
                block_size = random.randint(3, max_block_size)
            
            # HARD RULE: Validate block size (max 7 shifts)
            block_size = _validate_block_size(block_size, 7)
            
            # Choose night shift type for this block
            shift_type = random.choice(night_shift_types)
            
            # Find available dates for this block
            available_dates = find_available_dates_for_block(nocturnist, shift_type, block_size, 
                                                           provider_shifts, year, month, global_rules, provider_rules)
            
            if len(available_dates) >= block_size:
                # Assign the block
                block_events = assign_night_shift_block(nocturnist, shift_type, available_dates[:block_size], 
                                                      provider_shifts)
                events.extend(block_events)
                remaining_shifts -= block_size
            else:
                # If we can't find enough dates, try with fewer shifts
                if len(available_dates) >= 3:
                    block_events = assign_night_shift_block(nocturnist, shift_type, available_dates, 
                                                          provider_shifts)
                    events.extend(block_events)
                    remaining_shifts -= len(available_dates)
                else:
                    # Skip this nocturnist if we can't assign a proper block
                    break
    
    return events

def assign_night_shift_block(nocturnist: str, shift_type: str, dates: List[date], 
                           provider_shifts: Dict) -> List[SEvent]:
    """
    Assign a block of night shifts to a nocturnist.
    HARD STOP: Never exceed expected shifts.
    """
    events = []
    shift_config = get_shift_config(shift_type)
    
    # HARD STOP: Check if this block would exceed expected shifts
    current_shifts = len(provider_shifts.get(nocturnist, []))
    if current_shifts + len(dates) > 16:  # Nocturnists typically have 16 shifts max
        logger.warning(f"HARD STOP: Nocturnist {nocturnist} would exceed 16 shifts (current: {current_shifts}, block: {len(dates)})")
        return events
    
    for day in dates:
        start_time = datetime.combine(day, parse_time(shift_config["start"]))
        end_time = datetime.combine(day, parse_time(shift_config["end"]))
        
        # Handle overnight shifts
        if shift_config["end"] < shift_config["start"]:
            end_time += timedelta(days=1)
        
        event = SEvent(
            id=str(uuid.uuid4()),
            title=f"{shift_config['label']} - {nocturnist}",
            start=start_time,
            end=end_time,
            backgroundColor=shift_config["color"],
            extendedProps={
                "provider": nocturnist,
                "shift_type": shift_type,
                "shift_label": shift_config["label"]
            }
        )
        
        events.append(event)
        provider_shifts[nocturnist].append(event)
    
    return events

def assign_app_shifts(month_days: List[date], app_providers: List[str], 
                     shift_capacity: Dict[str, int], provider_rules: Dict, 
                     global_rules: RuleConfig) -> List[SEvent]:
    """
    Assign APP shifts following APP-specific rules:
    - Only APPs can do APP shifts
    - 2 slots on weekdays, 1 slot on weekends
    - No min/max shift requirements
    """
    events = []
    
    for day in month_days:
        # Determine APP capacity based on day of week
        is_weekend = day.weekday() >= 5  # Saturday = 5, Sunday = 6
        app_capacity = 1 if is_weekend else 2
        
        # Get available APP providers for this day
        available_apps = [p for p in app_providers 
                         if not is_provider_unavailable_on_date(p, day, provider_rules)]
        
        # Filter out APPs who already have shifts on this day
        available_apps = [p for p in available_apps 
                         if not _has_shift_on_date(p, day, events)]
        
        # Assign APPs up to capacity
        assigned_count = 0
        while assigned_count < app_capacity and available_apps:
            # Select APP (with some randomness)
            app_provider = random.choice(available_apps)
            available_apps.remove(app_provider)
            
            # HARD STOP: Check if this APP would exceed 12 shifts
            current_shifts = len([e for e in events if e.extendedProps.get("provider") == app_provider])
            if current_shifts >= 12:
                logger.warning(f"HARD STOP: APP provider {app_provider} would exceed 12 shifts (current: {current_shifts})")
                continue
            
            # Create APP shift event
            start_time = datetime.combine(day, parse_time("07:00"))
            end_time = datetime.combine(day, parse_time("19:00"))
            
            event = SEvent(
                id=str(uuid.uuid4()),
                title=f"APP Provider - {app_provider}",
                start=start_time,
                end=end_time,
                backgroundColor="#8b5cf6",
                extendedProps={
                    "provider": app_provider,
                    "shift_type": "APP",
                    "shift_label": "APP Provider"
                }
            )
            
            events.append(event)
            assigned_count += 1
    
    return events

def assign_physician_shifts(month_days: List[date], physician_providers: List[str], 
                          shift_capacity: Dict[str, int], provider_rules: Dict, 
                          global_rules: RuleConfig, provider_shifts: Dict, 
                          year: int, month: int) -> List[SEvent]:
    """
    Assign physician shifts following the ground rules:
    - Shift type consistency in blocks
    - Rounders start with 1-2 admitting shifts
    - Blocks of 3-7 shifts
    - Optimize shift distribution
    """
    events = []
    
    # Create shift blocks for each physician
    physician_blocks = create_shift_blocks(month_days, physician_providers, 
                                         shift_capacity, provider_rules, global_rules, year, month)
    
    # Assign shifts based on blocks
    for provider, blocks in physician_blocks.items():
        for block in blocks:
            block_events = assign_shift_block(provider, block, provider_shifts, year, month, global_rules, provider_rules)
            events.extend(block_events)
    
    return events

def fill_remaining_shifts_balanced(month_days: List[date], providers: List[str], 
                                 shift_capacity: Dict[str, int], provider_rules: Dict, 
                                 global_rules: RuleConfig, provider_shifts: Dict,
                                 year: int = None, month: int = None) -> List[SEvent]:
    """
    BALANCED ALGORITHM: Fill remaining shifts with a more flexible approach.
    Prioritizes getting providers assigned while respecting hard rules.
    
    Algorithm:
    1. First pass: Try to fill gaps with blocks where possible (2+ consecutive days)
    2. Second pass: Fill individual gaps with best available providers
    3. Third pass: Fill any remaining gaps with any available provider (relaxed rules)
    """
    events = []
    
    # Pre-calculate expected shifts for all providers
    provider_expected_shifts = {}
    for provider in providers:
        if provider not in APP_PROVIDER_INITIALS:
            from core.utils import get_adjusted_expected_shifts
            provider_expected_shifts[provider] = get_adjusted_expected_shifts(
                provider, year, month, provider_rules, global_rules
            )
    
    # PASS 1: Try to fill consecutive gaps with blocks (2+ days)
    events.extend(_fill_consecutive_gaps_with_blocks(month_days, providers, shift_capacity,
                                                   provider_rules, global_rules, provider_shifts, 
                                                   provider_expected_shifts, year, month))
    
    # PASS 2: Fill individual gaps with best available providers
    events.extend(_fill_individual_gaps(month_days, providers, shift_capacity,
                                      provider_rules, global_rules, provider_shifts,
                                      provider_expected_shifts, year, month))
    
    # PASS 3: Fill any remaining gaps with relaxed rules (if providers are under their expected shifts)
    events.extend(_fill_remaining_gaps_relaxed(month_days, providers, shift_capacity,
                                             provider_rules, global_rules, provider_shifts,
                                             provider_expected_shifts, year, month))
    
    return events

def _fill_consecutive_gaps_with_blocks(month_days: List[date], providers: List[str],
                                     shift_capacity: Dict[str, int], provider_rules: Dict,
                                     global_rules: RuleConfig, provider_shifts: Dict,
                                     provider_expected_shifts: Dict, year: int, month: int) -> List[SEvent]:
    """
    Fill consecutive gaps (2+ days) with blocks to optimize provider assignments.
    """
    events = []
    
    # Find consecutive gaps for each shift type
    for shift_type in ["R12", "A12", "A10", "N12"]:
        if shift_type not in shift_capacity:
            continue
            
        # Find consecutive gaps of 2+ days
        consecutive_gaps = _find_consecutive_gaps_for_shift_type(month_days, shift_type, 
                                                               shift_capacity, provider_shifts, min_consecutive=2)
        
        for gap_group in consecutive_gaps:
            # Find best provider for this gap group
            best_provider = _find_best_provider_for_gap_group_relaxed(gap_group, providers, 
                                                                     provider_shifts, provider_rules,
                                                                     global_rules, provider_expected_shifts, year, month)
            
            if best_provider:
                # Assign the block
                block_events = _assign_shift_block_relaxed(best_provider, gap_group, provider_shifts)
                events.extend(block_events)
                
                # Update provider tracking
                for event in block_events:
                    provider = event.extendedProps.get("provider")
                    if provider:
                        provider_shifts[provider].append(event)
    
    return events

def _fill_individual_gaps(month_days: List[date], providers: List[str],
                         shift_capacity: Dict[str, int], provider_rules: Dict,
                         global_rules: RuleConfig, provider_shifts: Dict,
                         provider_expected_shifts: Dict, year: int, month: int) -> List[SEvent]:
    """
    Fill individual gaps with best available providers.
    """
    events = []
    
    for day in month_days:
        for shift_type in ["R12", "A12", "A10", "N12"]:
            if shift_type not in shift_capacity:
                continue
                
            capacity = shift_capacity[shift_type]
            assigned_count = count_shifts_on_date(day, shift_type, provider_shifts)
            remaining_slots = capacity - assigned_count
            
            for slot in range(remaining_slots):
                # Find best available provider
                best_provider = _find_best_provider_for_single_shift(day, shift_type, providers,
                                                                   provider_shifts, provider_rules,
                                                                   global_rules, provider_expected_shifts, year, month)
                
                if best_provider:
                    event = _create_shift_event(best_provider, shift_type, day)
                    events.append(event)
                    provider_shifts[best_provider].append(event)
    
    return events

def _fill_remaining_gaps_relaxed(month_days: List[date], providers: List[str],
                                shift_capacity: Dict[str, int], provider_rules: Dict,
                                global_rules: RuleConfig, provider_shifts: Dict,
                                provider_expected_shifts: Dict, year: int, month: int) -> List[SEvent]:
    """
    Fill any remaining gaps with relaxed rules for providers under their expected shifts.
    """
    events = []
    
    for day in month_days:
        for shift_type in ["R12", "A12", "A10", "N12"]:
            if shift_type not in shift_capacity:
                continue
                
            capacity = shift_capacity[shift_type]
            assigned_count = count_shifts_on_date(day, shift_type, provider_shifts)
            remaining_slots = capacity - assigned_count
            
            for slot in range(remaining_slots):
                # Find any provider who is under their expected shifts
                best_provider = _find_any_available_provider(day, shift_type, providers,
                                                           provider_shifts, provider_expected_shifts)
                
                if best_provider:
                    event = _create_shift_event(best_provider, shift_type, day)
                    events.append(event)
                    provider_shifts[best_provider].append(event)
    
    return events

def _calculate_provider_score(provider: str, day: date, shift_type: str, provider_shifts: Dict,
                            provider_rules: Dict, global_rules: RuleConfig, 
                            provider_expected_shifts: Dict, year: int = None, month: int = None) -> Optional[float]:
    """
    Calculate a score for a provider taking a specific shift.
    HARD STOP: Returns None if assignment would exceed expected shifts.
    Lower score = better choice.
    """
    # Basic eligibility checks
    if is_provider_unavailable_on_date(provider, day, provider_rules):
        return None
    
    if _has_shift_on_date(provider, day, provider_shifts[provider]):
        return None
    
    # HARD RULE: Check shift type preference
    if not _validate_shift_type_preference(provider, shift_type, provider_rules):
        return None
    
    # HARD RULE: Check provider role
    if not _validate_provider_role_shift_type(provider, shift_type):
        return None
    
    # HARD RULE: Check rest requirements
    min_rest_days = 2
    if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
        min_rest_days = max(2, global_rules.min_days_between_shifts)
    
    if not _has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
        return None
    
    # HARD STOP: Check if assignment would exceed expected shifts
    current_shifts = len(provider_shifts.get(provider, []))
    
    if provider in APP_PROVIDER_INITIALS:
        # APP providers: HARD STOP at 12 shifts
        if current_shifts >= 12:
            logger.debug(f"APP provider {provider} at {current_shifts} shifts - HARD STOP")
            return None
        score = current_shifts  # Lower is better
    else:
        # Regular providers: HARD STOP at expected shifts
        expected_shifts = provider_expected_shifts.get(provider, 15)
        if current_shifts >= expected_shifts:
            logger.debug(f"Provider {provider} at {current_shifts} shifts (expected {expected_shifts}) - HARD STOP")
            return None
        
        # Score based on how close to expected shifts (closer = higher score = worse)
        score = current_shifts / expected_shifts
    
    # Add shift timing preference score
    if year and month:
        from core.utils import calculate_shift_timing_score
        timing_score = calculate_shift_timing_score(provider, day, provider_rules, year, month)
        score += timing_score
    
    return score

def _validate_all_hard_rules(provider: str, day: date, shift_type: str, provider_shifts: Dict,
                           provider_rules: Dict, global_rules: RuleConfig, 
                           provider_expected_shifts: Dict) -> bool:
    """
    Validate ALL hard rules for a provider taking a specific shift.
    ULTRA-STRICT: Returns True only if ALL rules are satisfied.
    """
    # HARD RULE 1: Shift type preference
    if not _validate_shift_type_preference(provider, shift_type, provider_rules):
        logger.debug(f"Provider {provider} fails shift type preference check for {shift_type}")
        return False
    
    # HARD RULE 2: Provider role validation
    if not _validate_provider_role_shift_type(provider, shift_type):
        logger.debug(f"Provider {provider} fails role validation for {shift_type}")
        return False
    
    # HARD RULE 3: Expected shifts limit - ULTRA-STRICT
    current_shifts = len(provider_shifts.get(provider, []))
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
    
    if not _has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
        logger.debug(f"Provider {provider} fails rest requirements check")
        return False
    
    logger.debug(f"Provider {provider} passes ALL hard rules for {shift_type}")
    return True

def _create_shift_event(provider: str, shift_type: str, day: date) -> SEvent:
    """
    Create a shift event for a provider.
    """
    shift_config = get_shift_config(shift_type)
    start_time = datetime.combine(day, parse_time(shift_config["start"]))
    end_time = datetime.combine(day, parse_time(shift_config["end"]))
    
    # Handle overnight shifts
    if shift_config["end"] < shift_config["start"]:
        end_time += timedelta(days=1)
    
    return SEvent(
        id=str(uuid.uuid4()),
        title=f"{shift_config['label']} - {provider}",
        start=start_time,
        end=end_time,
        backgroundColor=shift_config["color"],
        extendedProps={
            "provider": provider,
            "shift_type": shift_type,
            "shift_label": shift_config["label"]
        }
    )

def select_provider_with_fewer_shifts(available_providers: List[str], 
                                    provider_shifts: Dict) -> str:
    """
    Select a provider from the available list, preferring those with fewer shifts.
    """
    if not available_providers:
        return None
    
    # Find provider with minimum shifts
    min_shifts = float('inf')
    selected_provider = available_providers[0]
    
    for provider in available_providers:
        current_shifts = len(provider_shifts.get(provider, []))
        if current_shifts < min_shifts:
            min_shifts = current_shifts
            selected_provider = provider
    
    return selected_provider

def get_available_providers_for_shift(day: date, shift_type: str, providers: List[str], 
                                     provider_shifts: Dict, provider_rules: Dict, 
                                     global_rules: RuleConfig = None) -> List[str]:
    """
    Get available providers for a specific shift type on a given day.
    HARD RULES: Only providers who explicitly prefer this shift type are available.
    """
    available_providers = []
    
    # Get minimum rest days from global rules, default to 2 (HARD RULE)
    min_rest_days = 2
    if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
        min_rest_days = max(2, global_rules.min_days_between_shifts)  # Enforce minimum 2 days
    
    for provider in providers:
        # Skip if provider is unavailable
        if is_provider_unavailable_on_date(provider, day, provider_rules):
            continue
        
        # Skip if provider already has a shift on this day
        if _has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # HARD RULE: Check provider-specific shift preferences
        provider_rule = provider_rules.get(provider, {})
        if not isinstance(provider_rule, dict):
            provider_rule = {}
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        # HARD RULE: Provider must EXPLICITLY prefer this shift type
        # Default is False (not True) - providers must opt-in to shift types
        if not shift_preferences.get(shift_type, False):
            continue
        
        # HARD RULE: Validate that provider role matches shift type
        if not _validate_provider_role_shift_type(provider, shift_type):
            continue
        
        # Check rest requirements
        if not _has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
            continue
        
        available_providers.append(provider)
    
    return available_providers



def create_shift_blocks(month_days: List[date], physician_providers: List[str], 
                       shift_capacity: Dict[str, int], provider_rules: Dict, 
                       global_rules: RuleConfig, year: int, month: int) -> Dict[str, List[Dict]]:
    """
    Create shift blocks for each physician following HARD RULES:
    - Maximum 7 shifts per block
    - 3-day rest between blocks
    - Rounders should end on rounding shifts
    - HARD RULE: Only create blocks for shift types the provider explicitly prefers
    - HARD RULE: Never exceed expected shifts (adjusted for FTE and vacation)
    """
    physician_blocks = {p: [] for p in physician_providers}
    
    # Define shift types for physicians (exclude APP and night shifts for non-nocturnists)
    physician_shift_types = ["R12", "A12", "A10"]
    
    # Ensure we have enough providers to cover all shifts
    if len(physician_providers) == 0:
        return physician_blocks
    
    for provider in physician_providers:
        # Get provider preferences and expected shifts (adjusted for FTE and vacation)
        from core.utils import get_adjusted_expected_shifts
        target_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        
        # HARD RULE: Get provider's shift preferences
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        # HARD RULE: Filter shift types to only those the provider explicitly prefers
        preferred_shift_types = []
        for shift_type in physician_shift_types:
            if shift_preferences.get(shift_type, False):  # Must be explicitly True
                preferred_shift_types.append(shift_type)
        
        # If no preferred shift types, skip this provider (they won't get any blocks)
        if not preferred_shift_types:
            logger.warning(f"Provider {provider} has no preferred shift types - no blocks created")
            continue
        
        # Create blocks with maximum 7 shifts per block (HARD RULE)
        remaining_shifts = target_shifts
        while remaining_shifts > 0:
            # Determine block size (3-7 shifts, but no more than remaining shifts)
            max_block_size = min(7, remaining_shifts)
            # Ensure we have a valid range for random.randint
            if max_block_size < 3:
                # If remaining shifts is less than 3, use all remaining shifts
                block_size = remaining_shifts
            else:
                if max_block_size == 3:
                    block_size = 3
                else:
                    block_size = random.randint(3, max_block_size)
            
            # HARD RULE: Validate block size (max 7 shifts)
            block_size = _validate_block_size(block_size, 7)
            
            # HARD RULE: Only select from preferred shift types
            shift_type = select_shift_type_for_block(provider, preferred_shift_types, 
                                                    provider_rule)
            
            # HARD RULE: Double-check shift type preference
            if not _validate_shift_type_preference(provider, shift_type, provider_rules):
                logger.warning(f"Provider {provider} does not prefer shift type {shift_type}, skipping block")
                continue
            
            # HARD RULE: Validate that provider role matches shift type
            if not _validate_provider_role_shift_type(provider, shift_type):
                logger.warning(f"Provider {provider} role does not match shift type {shift_type}, skipping block")
                continue
            
            # Create block
            block = {
                "shift_type": shift_type,
                "size": block_size,
                "start_date": None,  # Will be determined during assignment
                "dates": []
            }
            
            physician_blocks[provider].append(block)
            remaining_shifts -= block_size
    
    return physician_blocks

def select_shift_type_for_block(provider: str, shift_types: List[str], 
                               provider_rule: Dict) -> str:
    """
    Select appropriate shift type for a block based on provider preferences.
    HARD RULE: Only select from shift types the provider explicitly prefers.
    """
    try:
        # Ensure provider_rule is a dictionary
        if not isinstance(provider_rule, dict):
            provider_rule = {}
        
        # Get provider's shift preferences
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        # HARD RULE: Filter available shift types based on EXPLICIT preferences
        # Default is False - providers must opt-in to shift types
        available_types = []
        for shift_type in shift_types:
            if shift_preferences.get(shift_type, False):  # Must be explicitly True
                available_types.append(shift_type)
        
        # HARD RULE: If no preferred types, return the first available type
        # This should not happen if the calling function filters properly
        if not available_types:
            logger.warning(f"Provider {provider} has no preferred shift types from {shift_types}")
            return shift_types[0] if shift_types else "R12"
        
        # Prefer rounder shifts (R12) as they're most common
        if "R12" in available_types:
            return "R12"
        
        # Otherwise, choose randomly from available types
        return random.choice(available_types)
    except Exception as e:
        logger.warning(f"Error selecting shift type for provider {provider}: {e}")
        # Fallback to R12 if available, otherwise first shift type
        if "R12" in shift_types:
            return "R12"
        return shift_types[0] if shift_types else "R12"

def assign_shift_block(provider: str, block: Dict, provider_shifts: Dict, year: int, month: int,
                      global_rules: RuleConfig = None, provider_rules: Dict = None) -> List[SEvent]:
    """
    Assign a specific shift block to a provider with proper sequence rules.
    HARD STOP: Never exceed expected shifts.
    """
    events = []
    shift_type = block["shift_type"]
    block_size = block["size"]
    
    # HARD STOP: Check if this block would exceed expected shifts
    current_shifts = len(provider_shifts.get(provider, []))
    if provider in APP_PROVIDER_INITIALS:
        if current_shifts + block_size > 12:
            logger.warning(f"HARD STOP: APP provider {provider} would exceed 12 shifts (current: {current_shifts}, block: {block_size})")
            return events
    else:
        # Get expected shifts for regular providers
        from core.utils import get_adjusted_expected_shifts
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, {}, global_rules)
        if current_shifts + block_size > expected_shifts:
            logger.warning(f"HARD STOP: Provider {provider} would exceed {expected_shifts} shifts (current: {current_shifts}, block: {block_size})")
            return events
    
    # Get shift type configuration
    shift_config = get_shift_config(shift_type)
    
    # Find available dates for this block
    available_dates = find_available_dates_for_block(provider, shift_type, block_size, 
                                                   provider_shifts, year, month, global_rules, provider_rules)
    
    if len(available_dates) < block_size:
        # If we can't find enough consecutive dates, take what we can get
        available_dates = available_dates[:block_size]
    
    # Special handling for rounders: start with 1-2 admitting shifts, end with rounding
    if shift_type == "R12" and len(available_dates) >= 3:
        # Start with 1-2 admitting shifts (use consistent type)
        admitting_count = min(2, len(available_dates) - 1)  # Ensure at least 1 rounding shift
        admitting_events = assign_admitting_before_rounding(provider, available_dates[:admitting_count])
        events.extend(admitting_events)
        
        # Assign rounding shifts for the remaining dates
        rounding_dates = available_dates[admitting_count:]
        for day in rounding_dates:
            start_time = datetime.combine(day, parse_time(shift_config["start"]))
            end_time = datetime.combine(day, parse_time(shift_config["end"]))
            
            event = SEvent(
                id=str(uuid.uuid4()),
                title=f"{shift_config['label']} - {provider}",
                start=start_time,
                end=end_time,
                backgroundColor=shift_config["color"],
                extendedProps={
                    "provider": provider,
                    "shift_type": shift_type,
                    "shift_label": shift_config["label"]
                }
            )
            
            events.append(event)
            provider_shifts[provider].append(event)
    else:
        # For non-rounder shifts or insufficient dates, assign normally
        for i, day in enumerate(available_dates):
            if i >= block_size:
                break
                
            # Create shift event
            start_time = datetime.combine(day, parse_time(shift_config["start"]))
            end_time = datetime.combine(day, parse_time(shift_config["end"]))
            
            # Handle overnight shifts
            if shift_config["end"] < shift_config["start"]:
                end_time += timedelta(days=1)
            
            event = SEvent(
                id=str(uuid.uuid4()),
                title=f"{shift_config['label']} - {provider}",
                start=start_time,
                end=end_time,
                backgroundColor=shift_config["color"],
                extendedProps={
                    "provider": provider,
                    "shift_type": shift_type,
                    "shift_label": shift_config["label"]
                }
            )
            
            events.append(event)
            provider_shifts[provider].append(event)
    
    return events

def assign_admitting_before_rounding(provider: str, dates: List[date]) -> List[SEvent]:
    """
    Assign 1-2 admitting shifts before rounding shifts for continuity of care.
    Use consistent admitting shift type (either A12 or A10, not mixed).
    """
    events = []
    
    # Choose one admitting shift type for consistency (either A12 or A10)
    admitting_type = random.choice(["A12", "A10"])  # Pick one type for the entire block
    
    for day in dates:
        shift_config = get_shift_config(admitting_type)
        
        start_time = datetime.combine(day, parse_time(shift_config["start"]))
        end_time = datetime.combine(day, parse_time(shift_config["end"]))
        
        # Handle overnight shifts
        if shift_config["end"] < shift_config["start"]:
            end_time += timedelta(days=1)
        
        event = SEvent(
            id=str(uuid.uuid4()),
            title=f"{shift_config['label']} - {provider}",
            start=start_time,
            end=end_time,
            backgroundColor=shift_config["color"],
            extendedProps={
                "provider": provider,
                "shift_type": admitting_type,
                "shift_label": shift_config["label"]
            }
        )
        
        events.append(event)
    
    return events

def find_available_dates_for_block(provider: str, shift_type: str, block_size: int, 
                                  provider_shifts: Dict, year: int = None, month: int = None,
                                  global_rules: RuleConfig = None, provider_rules: Dict = None) -> List[date]:
    """
    Find available dates for a shift block, ensuring sufficient rest between blocks.
    Prioritizes consecutive dates for better block formation.
    """
    # Get all dates in the specified month
    if year is None or month is None:
        today = date.today()
        year, month = today.year, today.month
    
    # Get minimum rest days from global rules, default to 2 (HARD RULE)
    min_rest_days = 2
    if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
        min_rest_days = max(2, global_rules.min_days_between_shifts)  # Enforce minimum 2 days
    
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)
    
    available_dates = []
    
    # Find all available dates for this provider and shift type
    current_date = month_start
    while current_date <= month_end:
        if not is_provider_unavailable_on_date(provider, current_date, provider_rules):
            if not _has_shift_on_date(provider, current_date, provider_shifts[provider]):
                # Check for sufficient rest requirement
                if _has_sufficient_rest(provider, current_date, provider_shifts[provider], min_rest_days):
                    available_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Sort dates for better consecutive finding
    available_dates.sort()
    
    # Try to find consecutive dates first (preferred)
    consecutive_dates = find_consecutive_dates(available_dates, block_size)
    if consecutive_dates:
        return consecutive_dates
    
    # If no consecutive dates, try to find dates with minimal gaps
    if len(available_dates) >= block_size:
        # Find the best sequence with minimal gaps
        best_sequence = find_best_date_sequence(available_dates, block_size)
        if best_sequence:
            return best_sequence
    
    # Fallback: return any available dates
    return available_dates[:block_size]

def find_best_date_sequence(dates: List[date], count: int) -> List[date]:
    """
    Find the best sequence of dates with minimal gaps between them.
    """
    if len(dates) < count:
        return []
    
    best_sequence = None
    min_total_gap = float('inf')
    
    # Try all possible sequences of 'count' dates
    for i in range(len(dates) - count + 1):
        sequence = dates[i:i+count]
        total_gap = 0
        
        # Calculate total gap between consecutive dates
        for j in range(len(sequence) - 1):
            gap = (sequence[j+1] - sequence[j]).days - 1
            total_gap += gap
        
        if total_gap < min_total_gap:
            min_total_gap = total_gap
            best_sequence = sequence
    
    return best_sequence

def find_consecutive_dates(dates: List[date], count: int) -> List[date]:
    """
    Find consecutive dates in a list.
    """
    if len(dates) < count:
        return []
    
    dates.sort()
    
    for i in range(len(dates) - count + 1):
        consecutive = dates[i:i+count]
        if all((consecutive[j+1] - consecutive[j]).days == 1 
               for j in range(len(consecutive)-1)):
            return consecutive
    
    return []

def get_shift_config(shift_type: str) -> Dict:
    """
    Get shift configuration for a given shift type.
    """
    shift_configs = {
        "R12": {"label": "7am–7pm Rounder", "start": "07:00", "end": "19:00", "color": "#16a34a"},
        "A12": {"label": "7am–7pm Admitter", "start": "07:00", "end": "19:00", "color": "#f59e0b"},
        "A10": {"label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
        "N12": {"label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
        "NB": {"label": "Night Bridge", "start": "23:00", "end": "07:00", "color": "#06b6d4"},
        "APP": {"label": "APP Provider", "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
    }
    
    return shift_configs.get(shift_type, shift_configs["R12"])

def _has_sufficient_rest(provider: str, target_date: date, provider_shifts: List[SEvent], 
                        min_rest_days: int = 2) -> bool:
    """
    Check if provider has sufficient rest before the target date.
    HARD RULE: Default is 2 days rest between shift blocks (unless manually edited).
    """
    # Find the last shift date for this provider
    last_shift_date = None
    for event in provider_shifts:
        if hasattr(event, 'start'):
            event_date = event.start.date()
        elif isinstance(event, dict) and 'start' in event:
            try:
                event_date = datetime.fromisoformat(event['start']).date()
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        if last_shift_date is None or event_date > last_shift_date:
            last_shift_date = event_date
    
    if last_shift_date is None:
        return True  # No previous shifts, so rest requirement is met
    
    # Check if there are sufficient days between last shift and target date
    days_since_last = (target_date - last_shift_date).days
    return days_since_last >= min_rest_days

def _has_shift_on_date(provider: str, day: date, events: List[SEvent]) -> bool:
    """
    Check if provider already has a shift on a specific date.
    """
    for event in events:
        if hasattr(event, 'start'):
            event_date = event.start.date()
        elif isinstance(event, dict) and 'start' in event:
            try:
                event_date = datetime.fromisoformat(event['start']).date()
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        if event_date == day:
            return True
    return False

def _validate_block_size(block_size: int, max_allowed: int = 7) -> int:
    """
    Validate and enforce block size limits.
    HARD RULE: No block should exceed 7 shifts (unless manually edited).
    """
    if block_size > max_allowed:
        logger.warning(f"Block size {block_size} exceeds maximum {max_allowed}, reducing to {max_allowed}")
        return max_allowed
    return block_size

def _validate_shift_type_preference(provider: str, shift_type: str, provider_rules: Dict) -> bool:
    """
    Validate that a provider can be assigned to a specific shift type.
    HARD RULE: Providers can only be assigned to shift types they explicitly prefer.
    """
    provider_rule = provider_rules.get(provider, {})
    if not isinstance(provider_rule, dict):
        logger.warning(f"Provider {provider} has invalid rule format")
        return False
    
    shift_preferences = provider_rule.get("shift_preferences", {})
    if not isinstance(shift_preferences, dict):
        logger.warning(f"Provider {provider} has invalid shift_preferences format")
        return False
    
    # HARD RULE: Provider must EXPLICITLY prefer this shift type
    preference = shift_preferences.get(shift_type, False)
    if not preference:
        logger.debug(f"Provider {provider} does not prefer {shift_type} (preference: {preference})")
        return False
    
    logger.debug(f"Provider {provider} prefers {shift_type} - VALID")
    return True

def _validate_expected_shifts_limit(provider: str, current_shifts: int, expected_shifts: int, 
                                   additional_shifts: int = 0) -> bool:
    """
    Validate that a provider won't exceed their expected shifts.
    HARD RULE: Providers should not exceed expected shifts (unless manually edited).
    """
    total_shifts_after_assignment = current_shifts + additional_shifts + 1
    
    if total_shifts_after_assignment > expected_shifts:
        logger.warning(f"Provider {provider} would exceed expected shifts: {current_shifts} + {additional_shifts} + 1 = {total_shifts_after_assignment} > {expected_shifts}")
        return False
    
    logger.debug(f"Provider {provider} shift count valid: {current_shifts} + {additional_shifts} + 1 = {total_shifts_after_assignment} <= {expected_shifts}")
    return True

def _validate_provider_role_shift_type(provider: str, shift_type: str) -> bool:
    """
    Validate that a provider is assigned to an appropriate shift type based on their role.
    HARD RULE: Providers should only be assigned to shift types appropriate for their role.
    """
    # APP providers should only do APP shifts
    if provider in APP_PROVIDER_INITIALS:
        if shift_type != "APP":
            logger.warning(f"APP provider {provider} cannot be assigned to {shift_type} - only APP shifts allowed")
            return False
        logger.debug(f"APP provider {provider} assigned to APP shift - VALID")
        return True
    
    # Nocturnists should only do night shifts
    if provider in NOCTURNISTS:
        if shift_type not in ["N12", "NB"]:
            logger.warning(f"Nocturnist {provider} cannot be assigned to {shift_type} - only night shifts allowed")
            return False
        logger.debug(f"Nocturnist {provider} assigned to night shift {shift_type} - VALID")
        return True
    
    # Regular physicians can do day shifts (R12, A12, A10) and night shifts if they prefer them
    if shift_type in ["R12", "A12", "A10", "N12", "NB"]:
        logger.debug(f"Regular physician {provider} assigned to {shift_type} - VALID")
        return True
    else:
        logger.warning(f"Regular physician {provider} cannot be assigned to {shift_type} - invalid shift type")
        return False

def _can_provider_take_shift(provider: str, day: date, shift_type: str, 
                           provider_events: List[SEvent], last_shift_date: Optional[date],
                           global_rules: RuleConfig) -> bool:
    """
    Check if a provider can take a specific shift.
    """
    # Check rest days between shifts
    if last_shift_date:
        days_since_last = (day - last_shift_date).days
        if days_since_last < global_rules.min_days_between_shifts:
            return False
    
    # Check if provider already has a shift on this day
    for event in provider_events:
        if hasattr(event, 'start'):
            event_date = event.start.date()
        elif isinstance(event, dict) and 'start' in event:
            try:
                event_date = datetime.fromisoformat(event['start']).date()
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        if event_date == day:
            return False
    
    return True

def validate_rules(events: List[SEvent], providers: List[str], 
                  global_rules: RuleConfig, provider_rules: Dict, 
                  year: int = None, month: int = None) -> Dict[str, Any]:
    """
    ENHANCED VALIDATION: Comprehensive debugging and validation of scheduling rules.
    Checks for:
    1. Expected shift count violations (exceeding or below by more than 1)
    2. Shift type preference violations
    3. Coverage gaps
    4. Rest day violations
    5. Role-based shift type violations
    """
    # Determine year and month from events if not provided
    if year is None or month is None:
        if events:
            # Get year and month from the first event
            first_event = events[0]
            if hasattr(first_event, 'start'):
                year = first_event.start.year
                month = first_event.start.month
            else:
                # Fallback to current date
                from datetime import date as date_class
                today = date_class.today()
                year = today.year
                month = today.month
        else:
            # Fallback to current date
            from datetime import date as date_class
            today = date_class.today()
            year = today.year
            month = today.month
    
    violations = []
    provider_violations = {}
    coverage_gaps = []
    preference_violations = []
    rest_violations = []
    
    # Initialize tracking dictionaries
    provider_shift_counts = {p: 0 for p in providers}
    provider_shift_types = {p: [] for p in providers}  # Track which shift types each provider has
    provider_shift_dates = {p: [] for p in providers}  # Track dates of shifts for rest analysis
    provider_weekend_shifts = {p: 0 for p in providers}
    provider_night_shifts = {p: 0 for p in providers}
    provider_rounder_shifts = {p: 0 for p in providers}
    provider_admitting_shifts = {p: 0 for p in providers}
    
    # Analyze all events
    for event in events:
        provider = event.extendedProps.get("provider")
        shift_type = event.extendedProps.get("shift_type")
        
        if provider and shift_type:
            provider_shift_counts[provider] += 1
            provider_shift_types[provider].append(shift_type)
            
            # Get event date
            if hasattr(event, 'start'):
                event_date = event.start.date()
                provider_shift_dates[provider].append(event_date)
            else:
                continue
            
            # Count weekend shifts
            if event_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                provider_weekend_shifts[provider] += 1
            
            # Count shift types
            if shift_type in ["N12", "NB"]:
                provider_night_shifts[provider] += 1
            elif shift_type == "R12":
                provider_rounder_shifts[provider] += 1
            elif shift_type in ["A12", "A10"]:
                provider_admitting_shifts[provider] += 1
    
    # 1. CHECK EXPECTED SHIFT COUNT VIOLATIONS
    for provider in providers:
        shift_count = provider_shift_counts[provider]
        provider_rule = provider_rules.get(provider, {})
        
        # Skip APP providers for expected shift validation
        if provider in APP_PROVIDER_INITIALS:
            continue
        
        # Get expected shifts (adjusted for vacation)
        from core.utils import get_adjusted_expected_shifts
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        
        provider_violations[provider] = []
        
        # Check if provider exceeds expected shifts or is below by more than 1
        if shift_count > expected_shifts:
            violation = f"❌ {provider}: {shift_count} shifts (EXCEEDS expected {expected_shifts})"
            violations.append(violation)
            provider_violations[provider].append(violation)
        elif shift_count < (expected_shifts - 1):
            violation = f"⚠️ {provider}: {shift_count} shifts (below expected {expected_shifts} by more than 1)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        else:
            # Log good performance
            logger.info(f"✅ {provider}: {shift_count} shifts (within expected range: {expected_shifts})")
    
    # 2. CHECK SHIFT TYPE PREFERENCE VIOLATIONS
    for provider in providers:
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        for shift_type in provider_shift_types[provider]:
            # Check if provider explicitly prefers this shift type
            if not shift_preferences.get(shift_type, False):
                violation = f"❌ {provider}: Assigned {shift_type} but does NOT prefer it"
                preference_violations.append(violation)
                if provider not in provider_violations:
                    provider_violations[provider] = []
                provider_violations[provider].append(violation)
    
    # 3. CHECK REST DAY VIOLATIONS
    min_rest_days = 2
    if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
        min_rest_days = max(2, global_rules.min_days_between_shifts)
    
    for provider in providers:
        dates = sorted(provider_shift_dates[provider])
        for i in range(len(dates) - 1):
            days_between = (dates[i+1] - dates[i]).days
            if days_between < min_rest_days:
                violation = f"❌ {provider}: Only {days_between} days rest between {dates[i]} and {dates[i+1]} (min {min_rest_days} required)"
                rest_violations.append(violation)
                if provider not in provider_violations:
                    provider_violations[provider] = []
                provider_violations[provider].append(violation)
    
    # 4. CHECK ROLE-BASED SHIFT TYPE VIOLATIONS
    for provider in providers:
        for shift_type in provider_shift_types[provider]:
            if not _validate_provider_role_shift_type(provider, shift_type):
                violation = f"❌ {provider}: Assigned {shift_type} but role does not allow it"
                if provider not in provider_violations:
                    provider_violations[provider] = []
                provider_violations[provider].append(violation)
    
    # 5. ANALYZE COVERAGE GAPS
    # Get all dates in the month
    from datetime import date as date_class
    month_start = date_class(year, month, 1)
    if month == 12:
        month_end = date_class(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date_class(year, month + 1, 1) - timedelta(days=1)
    
    # Check each day for coverage gaps
    # Note: shift_capacity is not part of global_rules, so we'll skip coverage gap analysis for now
    # This would need to be passed as a separate parameter to validate_rules
    logger.info("Coverage gap analysis skipped - shift_capacity not available in global_rules")
    
    # 6. CHECK WEEKEND AND NIGHT SHIFT LIMITS (existing logic)
    for provider in providers:
        # Check weekend coverage
        if provider_weekend_shifts[provider] < global_rules.min_weekend_shifts_per_month:
            violation = f"⚠️ {provider}: {provider_weekend_shifts[provider]} weekend shifts (min {global_rules.min_weekend_shifts_per_month} required)"
            violations.append(violation)
            if provider not in provider_violations:
                provider_violations[provider] = []
            provider_violations[provider].append(violation)
        
        if provider_weekend_shifts[provider] > global_rules.max_weekend_shifts_per_month:
            violation = f"❌ {provider}: {provider_weekend_shifts[provider]} weekend shifts (max {global_rules.max_weekend_shifts_per_month} allowed)"
            violations.append(violation)
            if provider not in provider_violations:
                provider_violations[provider] = []
            provider_violations[provider].append(violation)
        
        # Check night shift limits
        if provider_night_shifts[provider] < global_rules.min_night_shifts_per_month:
            violation = f"⚠️ {provider}: {provider_night_shifts[provider]} night shifts (min {global_rules.min_night_shifts_per_month} required)"
            violations.append(violation)
            if provider not in provider_violations:
                provider_violations[provider] = []
            provider_violations[provider].append(violation)
        
        if provider_night_shifts[provider] > global_rules.max_night_shifts_per_month:
            violation = f"❌ {provider}: {provider_night_shifts[provider]} night shifts (max {global_rules.max_night_shifts_per_month} allowed)"
            violations.append(violation)
            if provider not in provider_violations:
                provider_violations[provider] = []
            provider_violations[provider].append(violation)
    
    # Combine all violations
    all_violations = violations + preference_violations + rest_violations
    
    return {
        "is_valid": len(all_violations) == 0,
        "violations": all_violations,
        "provider_violations": provider_violations,
        "coverage_gaps": coverage_gaps,
        "preference_violations": preference_violations,
        "rest_violations": rest_violations,
        "summary": {
            "total_events": len(events),
            "providers_used": len([p for p in providers if provider_shift_counts[p] > 0]),
            "total_violations": len(all_violations),
            "coverage_gaps_count": len(coverage_gaps),
            "preference_violations_count": len(preference_violations),
            "rest_violations_count": len(rest_violations),
            "provider_stats": {
                provider: {
                    "total_shifts": provider_shift_counts[provider],
                    "expected_shifts": get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules) if provider not in APP_PROVIDER_INITIALS else "N/A",
                    "weekend_shifts": provider_weekend_shifts[provider],
                    "night_shifts": provider_night_shifts[provider],
                    "rounder_shifts": provider_rounder_shifts[provider],
                    "admitting_shifts": provider_admitting_shifts[provider],
                    "shift_types": list(set(provider_shift_types[provider]))  # Unique shift types assigned
                } for provider in providers
            }
        }
    }

def assign_senior_rounding_shifts(month_days: List[date], seniors: List[str], 
                                 shift_capacity: Dict[str, int], provider_rules: Dict, 
                                 global_rules: RuleConfig, provider_shifts: Dict, 
                                 year: int, month: int) -> List[SEvent]:
    """
    Assign rounding shifts (R12) to senior providers only.
    Seniors only do 7am-7pm rounding shifts.
    """
    events = []
    
    # Get rounding shift capacity
    rounding_capacity = shift_capacity.get("R12", 0)
    
    for day in month_days:
        # Count how many rounding shifts are already assigned on this day
        assigned_rounding = count_shifts_on_date(day, "R12", provider_shifts)
        remaining_rounding_slots = rounding_capacity - assigned_rounding
        
        if remaining_rounding_slots <= 0:
            continue  # No more rounding slots needed
        
        # Get available senior providers for this day
        available_seniors = []
        for senior in seniors:
            # Check if senior is available
            if is_provider_unavailable_on_date(senior, day, provider_rules):
                continue
            
            # Check if senior already has a shift on this day
            if _has_shift_on_date(senior, day, provider_shifts[senior]):
                continue
            
            # Check rest requirements
            min_rest_days = 2
            if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
                min_rest_days = max(2, global_rules.min_days_between_shifts)
            
            if not _has_sufficient_rest(senior, day, provider_shifts[senior], min_rest_days):
                continue
            
            # Check if senior would exceed expected shifts
            current_shifts = len(provider_shifts[senior])
            from core.utils import get_adjusted_expected_shifts
            expected_shifts = get_adjusted_expected_shifts(senior, year, month, provider_rules, global_rules)
            
            if current_shifts >= expected_shifts:
                continue
            
            available_seniors.append(senior)
        
        # Assign available seniors to rounding shifts
        for i in range(min(remaining_rounding_slots, len(available_seniors))):
            senior = available_seniors[i]
            
            # Create rounding shift event
            start_time = datetime.combine(day, parse_time("07:00"))
            end_time = datetime.combine(day, parse_time("19:00"))
            
            event = SEvent(
                id=str(uuid.uuid4()),
                title=f"7am–7pm Rounder - {senior}",
                start=start_time,
                end=end_time,
                backgroundColor="#16a34a",
                extendedProps={
                    "provider": senior,
                    "shift_type": "R12",
                    "shift_label": "7am–7pm Rounder"
                }
            )
            
            events.append(event)
            provider_shifts[senior].append(event)
            logger.info(f"✅ SENIOR: Assigned R12 to {senior} on {day}")
    
    return events

def analyze_admitting_gaps(month_days: List[date], shift_capacity: Dict[str, int], 
                          provider_shifts: Dict) -> List[Dict]:
    """
    Analyze gaps in admitting shifts (A12, A10) and return gap information.
    """
    gaps = []
    admitting_types = ["A12", "A10"]
    
    for day in month_days:
        for shift_type in admitting_types:
            capacity = shift_capacity.get(shift_type, 0)
            assigned = count_shifts_on_date(day, shift_type, provider_shifts)
            remaining = capacity - assigned
            
            if remaining > 0:
                gaps.append({
                    "day": day,
                    "shift_type": shift_type,
                    "capacity": capacity,
                    "assigned": assigned,
                    "remaining": remaining
                })
    
    return gaps

def analyze_rounding_gaps(month_days: List[date], shift_capacity: Dict[str, int], 
                         provider_shifts: Dict) -> List[Dict]:
    """
    Analyze gaps in rounding shifts (R12) and return gap information.
    """
    gaps = []
    rounding_type = "R12"
    
    for day in month_days:
        capacity = shift_capacity.get(rounding_type, 0)
        assigned = count_shifts_on_date(day, rounding_type, provider_shifts)
        remaining = capacity - assigned
        
        if remaining > 0:
            gaps.append({
                "day": day,
                "shift_type": rounding_type,
                "capacity": capacity,
                "assigned": assigned,
                "remaining": remaining
            })
    
    return gaps

def find_consecutive_gaps(gaps: List[Dict], min_consecutive: int = 3) -> List[List[Dict]]:
    """
    Find consecutive gaps that can be filled with blocks.
    """
    if not gaps:
        return []
    
    # Sort gaps by day
    gaps.sort(key=lambda x: x["day"])
    
    consecutive_groups = []
    current_group = [gaps[0]]
    
    for i in range(1, len(gaps)):
        current_gap = gaps[i]
        last_gap = current_group[-1]
        
        # Check if gaps are consecutive and same shift type
        days_diff = (current_gap["day"] - last_gap["day"]).days
        same_type = current_gap["shift_type"] == last_gap["shift_type"]
        
        if days_diff <= 2 and same_type:  # Allow 1-2 day gaps between shifts
            current_group.append(current_gap)
        else:
            # End current group if it's long enough
            if len(current_group) >= min_consecutive:
                consecutive_groups.append(current_group)
            current_group = [current_gap]
    
    # Add the last group if it's long enough
    if len(current_group) >= min_consecutive:
        consecutive_groups.append(current_group)
    
    return consecutive_groups

# Removed old gap-filling functions - replaced with balanced approach

def create_admitting_to_rounding_block(provider: str, gap_group: List[Dict], 
                                      provider_shifts: Dict, provider_rules: Dict, 
                                      global_rules: RuleConfig, year: int, month: int) -> List[SEvent]:
    """
    Create a block that starts with admitting shifts then transitions to rounding.
    """
    events = []
    
    # Sort gaps by day
    gap_group.sort(key=lambda x: x["day"])
    
    # Start with 1-2 admitting shifts, then transition to rounding
    admitting_count = min(2, len(gap_group))
    
    for i, gap in enumerate(gap_group):
        day = gap["day"]
        shift_type = gap["shift_type"]
        
        # First 1-2 shifts are admitting, rest are rounding
        if i < admitting_count:
            # Use the admitting shift type from the gap
            final_shift_type = shift_type
        else:
            # Transition to rounding
            final_shift_type = "R12"
        
        # Create shift event
        shift_config = get_shift_config(final_shift_type)
        start_time = datetime.combine(day, parse_time(shift_config["start"]))
        end_time = datetime.combine(day, parse_time(shift_config["end"]))
        
        # Handle overnight shifts
        if shift_config["end"] < shift_config["start"]:
            end_time += timedelta(days=1)
        
        event = SEvent(
            id=str(uuid.uuid4()),
            title=f"{shift_config['label']} - {provider}",
            start=start_time,
            end=end_time,
            backgroundColor=shift_config["color"],
            extendedProps={
                "provider": provider,
                "shift_type": final_shift_type,
                "shift_label": shift_config["label"]
            }
        )
        
        events.append(event)
        logger.info(f"✅ ADMITTING BLOCK: Assigned {final_shift_type} to {provider} on {day}")
    
    return events

def create_rounding_block(provider: str, gap_group: List[Dict], 
                         provider_shifts: Dict, provider_rules: Dict, 
                         global_rules: RuleConfig, year: int, month: int) -> List[SEvent]:
    """
    Create a rounding block for the provider.
    """
    events = []
    
    # Sort gaps by day
    gap_group.sort(key=lambda x: x["day"])
    
    for gap in gap_group:
        day = gap["day"]
        
        # Create rounding shift event
        shift_config = get_shift_config("R12")
        start_time = datetime.combine(day, parse_time(shift_config["start"]))
        end_time = datetime.combine(day, parse_time(shift_config["end"]))
        
        event = SEvent(
            id=str(uuid.uuid4()),
            title=f"{shift_config['label']} - {provider}",
            start=start_time,
            end=end_time,
            backgroundColor=shift_config["color"],
            extendedProps={
                "provider": provider,
                "shift_type": "R12",
                "shift_label": shift_config["label"]
            }
        )
        
        events.append(event)
        logger.info(f"✅ ROUNDING BLOCK: Assigned R12 to {provider} on {day}")
    
    return events

def validate_and_adjust_schedule(events: List[SEvent], providers: List[str], 
                                provider_rules: Dict, global_rules: RuleConfig, 
                                year: int, month: int) -> List[SEvent]:
    """
    Validate the schedule and automatically adjust violations.
    Returns the adjusted schedule.
    """
    logger.info("🔍 Starting schedule validation and auto-adjustment...")
    
    # Convert events to provider_shifts format for validation
    provider_shifts = {p: [] for p in providers}
    for event in events:
        provider = event.extendedProps.get("provider")
        if provider and provider in provider_shifts:
            provider_shifts[provider].append(event)
    
    violations = []
    adjusted_events = events.copy()
    
    # Check each provider for violations
    for provider in providers:
        provider_events = provider_shifts[provider]
        
        # 1. Check expected shifts limit
        current_shifts = len(provider_events)
        from core.utils import get_adjusted_expected_shifts
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        
        if current_shifts > expected_shifts:
            violation = {
                "provider": provider,
                "type": "exceeded_expected_shifts",
                "current": current_shifts,
                "expected": expected_shifts,
                "excess": current_shifts - expected_shifts
            }
            violations.append(violation)
            
            # Auto-adjust: Remove excess shifts (remove from end of month first)
            provider_events.sort(key=lambda x: x.start)
            excess_events = provider_events[expected_shifts:]
            for event in excess_events:
                if event in adjusted_events:
                    adjusted_events.remove(event)
                    logger.warning(f"🔄 AUTO-ADJUST: Removed excess shift for {provider} on {event.start.date()}")
        
        # 2. Check shift type preferences
        for event in provider_events:
            shift_type = event.extendedProps.get("shift_type")
            if not _validate_shift_type_preference(provider, shift_type, provider_rules):
                violation = {
                    "provider": provider,
                    "type": "shift_type_preference",
                    "shift_type": shift_type,
                    "date": event.start.date()
                }
                violations.append(violation)
                
                # Auto-adjust: Remove the shift
                if event in adjusted_events:
                    adjusted_events.remove(event)
                    logger.warning(f"🔄 AUTO-ADJUST: Removed {shift_type} shift for {provider} on {event.start.date()} (not preferred)")
        
        # 3. Check rest requirements
        provider_events = [e for e in adjusted_events if e.extendedProps.get("provider") == provider]
        provider_events.sort(key=lambda x: x.start)
        
        min_rest_days = 2
        if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
            min_rest_days = max(2, global_rules.min_days_between_shifts)
        
        for i in range(len(provider_events) - 1):
            current_event = provider_events[i]
            next_event = provider_events[i + 1]
            
            days_between = (next_event.start.date() - current_event.start.date()).days
            if days_between < min_rest_days:
                violation = {
                    "provider": provider,
                    "type": "insufficient_rest",
                    "current_event": current_event.start.date(),
                    "next_event": next_event.start.date(),
                    "days_between": days_between,
                    "required": min_rest_days
                }
                violations.append(violation)
                
                # Auto-adjust: Remove the later shift
                if next_event in adjusted_events:
                    adjusted_events.remove(next_event)
                    logger.warning(f"🔄 AUTO-ADJUST: Removed shift for {provider} on {next_event.start.date()} (insufficient rest)")
        
        # 4. Check role-based shift type violations
        for event in provider_events:
            shift_type = event.extendedProps.get("shift_type")
            if not _validate_provider_role_shift_type(provider, shift_type):
                violation = {
                    "provider": provider,
                    "type": "role_shift_type",
                    "shift_type": shift_type,
                    "date": event.start.date()
                }
                violations.append(violation)
                
                # Auto-adjust: Remove the shift
                if event in adjusted_events:
                    adjusted_events.remove(event)
                    logger.warning(f"🔄 AUTO-ADJUST: Removed {shift_type} shift for {provider} on {event.start.date()} (role violation)")
    
    # Log summary
    if violations:
        logger.warning(f"⚠️ Found {len(violations)} violations, auto-adjusted schedule")
        for violation in violations:
            logger.warning(f"  - {violation['provider']}: {violation['type']}")
    else:
        logger.info("✅ No violations found in schedule")
    
    return adjusted_events

def _find_consecutive_gaps_for_shift_type(month_days: List[date], shift_type: str,
                                        shift_capacity: Dict[str, int], provider_shifts: Dict,
                                        min_consecutive: int = 2) -> List[List[Dict]]:
    """
    Find consecutive gaps for a specific shift type.
    """
    gaps = []
    
    for day in month_days:
        capacity = shift_capacity.get(shift_type, 0)
        assigned = count_shifts_on_date(day, shift_type, provider_shifts)
        remaining = capacity - assigned
        
        if remaining > 0:
            gaps.append({
                "day": day,
                "shift_type": shift_type,
                "remaining": remaining
            })
    
    # Group consecutive gaps
    consecutive_groups = []
    if gaps:
        current_group = [gaps[0]]
        
        for i in range(1, len(gaps)):
            current_gap = gaps[i]
            last_gap = current_group[-1]
            
            # Check if gaps are consecutive (allow 1 day gap)
            days_diff = (current_gap["day"] - last_gap["day"]).days
            
            if days_diff <= 1:
                current_group.append(current_gap)
            else:
                if len(current_group) >= min_consecutive:
                    consecutive_groups.append(current_group)
                current_group = [current_gap]
        
        # Add the last group if it's long enough
        if len(current_group) >= min_consecutive:
            consecutive_groups.append(current_group)
    
    return consecutive_groups

def _find_best_provider_for_gap_group_relaxed(gap_group: List[Dict], providers: List[str],
                                             provider_shifts: Dict, provider_rules: Dict,
                                             global_rules: RuleConfig, provider_expected_shifts: Dict,
                                             year: int, month: int) -> Optional[str]:
    """
    Find the best provider for a gap group with relaxed rules.
    """
    best_provider = None
    best_score = float('inf')
    
    for provider in providers:
        # Check if provider can take at least 70% of the shifts in the gap group
        available_shifts = 0
        total_shifts = len(gap_group)
        
        for gap in gap_group:
            day = gap["day"]
            shift_type = gap["shift_type"]
            
            # Basic availability checks
            if is_provider_unavailable_on_date(provider, day, provider_rules):
                continue
            
            if _has_shift_on_date(provider, day, provider_shifts[provider]):
                continue
            
            # Check shift type preference (relaxed)
            if not _validate_shift_type_preference(provider, shift_type, provider_rules):
                continue
            
            # Check rest requirements (relaxed - allow 1 day rest)
            if not _has_sufficient_rest(provider, day, provider_shifts[provider], 1):
                continue
            
            available_shifts += 1
        
        # Provider must be available for at least 70% of shifts
        if available_shifts < total_shifts * 0.7:
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts + available_shifts > expected_shifts:
            continue
        
        # Calculate score (lower is better)
        score = current_shifts / expected_shifts
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    return best_provider

def _find_best_provider_for_single_shift(day: date, shift_type: str, providers: List[str],
                                       provider_shifts: Dict, provider_rules: Dict,
                                       global_rules: RuleConfig, provider_expected_shifts: Dict,
                                       year: int, month: int) -> Optional[str]:
    """
    Find the best provider for a single shift.
    """
    best_provider = None
    best_score = float('inf')
    
    for provider in providers:
        # Basic availability checks
        if is_provider_unavailable_on_date(provider, day, provider_rules):
            continue
        
        if _has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Check shift type preference
        if not _validate_shift_type_preference(provider, shift_type, provider_rules):
            continue
        
        # Check rest requirements (1 day minimum)
        if not _has_sufficient_rest(provider, day, provider_shifts[provider], 1):
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts >= expected_shifts:
            continue
        
        # Calculate score (lower is better)
        score = current_shifts / expected_shifts
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    return best_provider

def _find_any_available_provider(day: date, shift_type: str, providers: List[str],
                               provider_shifts: Dict, provider_expected_shifts: Dict) -> Optional[str]:
    """
    Find any available provider for a shift (relaxed rules).
    Only checks if provider is under their expected shifts.
    """
    for provider in providers:
        # Check if provider is under their expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts < expected_shifts:
            # Check if they don't already have a shift on this day
            if not _has_shift_on_date(provider, day, provider_shifts[provider]):
                return provider
    
    return None

def _assign_shift_block_relaxed(provider: str, gap_group: List[Dict], provider_shifts: Dict) -> List[SEvent]:
    """
    Assign a block of shifts to a provider with relaxed rules.
    """
    events = []
    
    for gap in gap_group:
        day = gap["day"]
        shift_type = gap["shift_type"]
        
        # Check if provider can take this shift
        if _has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Create shift event
        event = _create_shift_event(provider, shift_type, day)
        events.append(event)
    
    return events
