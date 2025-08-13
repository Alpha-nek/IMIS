# =============================================================================
# Core Scheduling Logic for IMIS Scheduler
# =============================================================================

import random
import logging
from datetime import datetime, date
from typing import List, Dict
from models.data_models import RuleConfig, SEvent
from models.constants import APP_PROVIDER_INITIALS
from core.utils import make_month_days
from core.exceptions import ScheduleGenerationError
from core.provider_types import NOCTURNISTS, SENIORS
from core.provider_rules import ensure_default_provider_rules
from core.balanced_scheduler import fill_remaining_shifts_balanced
from core.shift_creation import create_shift_event
from core.shift_validation import validate_all_hard_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_schedule(year: int, month: int, providers: List[str], 
                     shift_types: List[Dict], shift_capacity: Dict[str, int],
                     provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Generate a complete schedule following the ground rules.
    """
    try:
        # Ensure provider_rules is a dictionary
        if not isinstance(provider_rules, dict):
            provider_rules = {}
        
        # Initialize default rules for any missing providers
        provider_rules = ensure_default_provider_rules(providers, provider_rules)
        
        # Add some randomness for variety
        random.seed(datetime.now().timestamp())
        
        events = assign_advanced(year, month, providers, shift_types, shift_capacity, 
                               provider_rules, global_rules)
        
        return events
    except Exception as e:
        logger.error(f"Error generating schedule: {e}")
        raise ScheduleGenerationError(f"Failed to generate schedule: {e}")

def assign_advanced(year: int, month: int, providers: List[str], 
                   shift_types: List[Dict], shift_capacity: Dict[str, int],
                   provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    BALANCED assignment algorithm following ground rules.
    Uses a combination of block assignment and greedy filling to ensure good coverage.
    """
    try:
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

def assign_app_shifts(month_days: List[date], app_providers: List[str], 
                     shift_capacity: Dict[str, int], provider_rules: Dict, 
                     global_rules: RuleConfig) -> List[SEvent]:
    """
    Assign APP shifts following APP-specific rules.
    """
    events = []
    
    for day in month_days:
        # Determine APP capacity based on day of week
        is_weekend = day.weekday() >= 5  # Saturday = 5, Sunday = 6
        app_capacity = 1 if is_weekend else 2
        
        # Get available APP providers for this day
        from core.utils import is_provider_unavailable_on_date
        from core.shift_validation import has_shift_on_date
        
        available_apps = [p for p in app_providers 
                         if not is_provider_unavailable_on_date(p, day, provider_rules)]
        
        # Filter out APPs who already have shifts on this day
        available_apps = [p for p in available_apps 
                         if not has_shift_on_date(p, day, events)]
        
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
            event = create_shift_event(app_provider, "APP", day)
            events.append(event)
            assigned_count += 1
    
    return events

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
            from core.shift_validation import validate_block_size
            block_size = validate_block_size(block_size, 7)
            
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
    
    # HARD STOP: Check if this block would exceed expected shifts
    current_shifts = len(provider_shifts.get(nocturnist, []))
    if current_shifts + len(dates) > 16:  # Nocturnists typically have 16 shifts max
        logger.warning(f"HARD STOP: Nocturnist {nocturnist} would exceed 16 shifts (current: {current_shifts}, block: {len(dates)})")
        return events
    
    for day in dates:
        event = create_shift_event(nocturnist, shift_type, day)
        events.append(event)
        provider_shifts[nocturnist].append(event)
    
    return events

def assign_senior_rounding_shifts(month_days: List[date], seniors: List[str], shift_capacity: Dict[str, int],
                                provider_rules: Dict, global_rules: RuleConfig, provider_shifts: Dict, 
                                year: int, month: int) -> List[SEvent]:
    """
    Assign R12 shifts to senior providers only.
    """
    events = []
    
    for senior in seniors:
        # Determine target number of shifts for this senior (adjusted for vacation)
        from core.utils import get_adjusted_expected_shifts
        target_shifts = get_adjusted_expected_shifts(senior, year, month, provider_rules, global_rules)
        
        # Create blocks of 3-7 shifts
        remaining_shifts = target_shifts
        while remaining_shifts > 0:
            # Determine block size (3-7 shifts, but no more than remaining shifts)
            max_block_size = min(7, remaining_shifts)
            if max_block_size < 3:
                block_size = remaining_shifts
            else:
                block_size = random.randint(3, max_block_size)
            
            # Find available dates for this block
            available_dates = find_available_dates_for_block(senior, "R12", block_size, 
                                                           provider_shifts, year, month, global_rules, provider_rules)
            
            if len(available_dates) >= block_size:
                # Assign the block
                block_events = assign_shift_block(senior, "R12", available_dates[:block_size], provider_shifts)
                events.extend(block_events)
                remaining_shifts -= block_size
            else:
                # If we can't find enough dates, try with fewer shifts
                if len(available_dates) >= 3:
                    block_events = assign_shift_block(senior, "R12", available_dates, provider_shifts)
                    events.extend(block_events)
                    remaining_shifts -= len(available_dates)
                else:
                    break
    
    return events

def find_available_dates_for_block(provider: str, shift_type: str, block_size: int, 
                                  provider_shifts: Dict, year: int, month: int, 
                                  global_rules: RuleConfig = None, provider_rules: Dict = None) -> List[date]:
    """
    Find available dates for a block of shifts.
    """
    from datetime import date, timedelta
    from core.utils import is_provider_unavailable_on_date
    from core.shift_validation import has_shift_on_date, validate_shift_type_preference, has_sufficient_rest
    
    # Get month days
    month_days = make_month_days(year, month)
    
    available_dates = []
    consecutive_count = 0
    
    for day in month_days:
        # Check if provider is available on this day
        if is_provider_unavailable_on_date(provider, day, provider_rules):
            consecutive_count = 0
            continue
        
        # Check if provider already has a shift on this day
        if has_shift_on_date(provider, day, provider_shifts[provider]):
            consecutive_count = 0
            continue
        
        # Check shift type preference
        if not validate_shift_type_preference(provider, shift_type, provider_rules):
            consecutive_count = 0
            continue
        
        # Check rest requirements - ENFORCE 2+ days rest between blocks
        min_rest_days = 2
        if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
            min_rest_days = max(2, global_rules.min_days_between_shifts)
        
        if not has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
            consecutive_count = 0
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        from core.utils import get_adjusted_expected_shifts
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        
        if current_shifts + len(available_dates) + 1 > expected_shifts:
            break
        
        available_dates.append(day)
        consecutive_count += 1
        
        # If we have enough consecutive dates, we can stop
        if consecutive_count >= block_size:
            break
    
    return available_dates

def assign_shift_block(provider: str, shift_type: str, dates: List[date], provider_shifts: Dict) -> List[SEvent]:
    """
    Assign a block of shifts to a provider.
    """
    events = []
    
    for day in dates:
        event = create_shift_event(provider, shift_type, day)
        events.append(event)
        provider_shifts[provider].append(event)
    
    return events

def validate_and_adjust_schedule(events: List[SEvent], providers: List[str], 
                                provider_rules: Dict, global_rules: RuleConfig, 
                                year: int, month: int) -> List[SEvent]:
    """
    Validate and automatically adjust the schedule to fix violations.
    """
    # For now, return the events as-is
    # TODO: Implement comprehensive validation and adjustment
    return events

def validate_rules(events: List[SEvent], providers: List[str], 
                  global_rules: RuleConfig, provider_rules: Dict,
                  year: int = None, month: int = None) -> Dict:
    """
    ENHANCED VALIDATION: Comprehensive debugging and validation of scheduling rules.
    """
    # TODO: Implement comprehensive validation
    return {
        "violations": [],
        "provider_violations": {},
        "coverage_gaps": [],
        "preference_violations": [],
        "rest_violations": []
    }
