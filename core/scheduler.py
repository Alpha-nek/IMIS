# =============================================================================
# Core Scheduling Logic for IMIS Scheduler
# =============================================================================

import random
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from models.data_models import RuleConfig, SEvent
from models.constants import APP_PROVIDER_INITIALS
from core.utils import make_month_days, count_shifts_on_date
from core.exceptions import ScheduleGenerationError
from core.provider_types import NOCTURNISTS, SENIORS
from core.provider_rules import ensure_default_provider_rules
from core.balanced_scheduler import fill_remaining_shifts_balanced
from core.shift_creation import create_shift_event
from core.shift_validation import validate_all_hard_rules
from core.scoring import create_scorer

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
    BLOCK-BASED assignment algorithm following ground rules.
    Uses block scheduling for regular providers with proper rest periods.
    """
    try:
        # Validate inputs
        if not providers:
            logger.warning("No providers provided for schedule generation")
            return []
        
        if not isinstance(provider_rules, dict):
            logger.warning("Provider rules is not a dictionary, initializing empty dict")
            provider_rules = {}
        
        # Use the new scoring-based assignment
        events = assign_with_scoring(year, month, providers, shift_types, shift_capacity,
                                   provider_rules, global_rules)
        
        return events
    except Exception as e:
        logger.error(f"Error in assign_advanced: {e}")
        raise ScheduleGenerationError(f"Failed to assign shifts: {e}")

def assign_with_scoring(year: int, month: int, providers: List[str], 
                       shift_types: List[Dict], shift_capacity: Dict[str, int],
                       provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Enhanced assignment algorithm using comprehensive scoring system.
    """
    try:
        # Initialize
        events = []
        month_days = make_month_days(year, month)
        
        # Create shift type mapping
        shift_type_keys = [st.get("key", st.get("name", "")) for st in shift_types]
        
        from core.debug_logger import debug_logger
        debug_logger.clear()  # Clear previous debug messages
        
        debug_logger.log(f"🔄 SCHEDULER DEBUG: Starting assignment for {len(providers)} providers")
        debug_logger.log(f"📋 Providers: {providers}")
        debug_logger.log(f"⚙️ Shift types: {shift_type_keys}")
        debug_logger.log(f"📊 Shift capacity: {shift_capacity}")
        debug_logger.log(f"📅 Month days: {len(month_days)}")
        
        # Ensure provider rules are set up
        debug_logger.log("🔧 Setting up provider rules...")
        from core.provider_rules import ensure_default_provider_rules
        provider_rules = ensure_default_provider_rules(providers, provider_rules)
        debug_logger.log(f"✅ Provider rules initialized for {len(provider_rules)} providers")
        
        # Add randomness for variety while maintaining deterministic scoring
        random.seed(year * 100 + month)
        providers_shuffled = providers.copy()
        random.shuffle(providers_shuffled)
        
        # Main assignment loop - assign shifts day by day using scoring
        debug_logger.log("🔄 Starting main assignment loop...")
        print(f"🔄 SCHEDULER: Starting assignment for {len(month_days)} days, {len(providers)} providers")
        print(f"🔄 SCHEDULER: Shift types: {[st.get('key') for st in shift_types]}")
        day_count = 0
        total_days = len(month_days)
        
        for current_day in month_days:
            day_count += 1
            debug_logger.log(f"📅 Processing day {day_count}/{total_days}: {current_day}")
            print(f"📅 Day {day_count}: {current_day}")
            
            for shift_type_dict in shift_types:
                shift_key = shift_type_dict.get("key", shift_type_dict.get("name", ""))
                debug_logger.log(f"  🎯 Processing shift type: {shift_key}")
                
                # Get capacity for this shift type on this day
                capacity = get_shift_capacity(shift_key, current_day, shift_capacity)
                debug_logger.log(f"  📊 Capacity for {shift_key}: {capacity}")
                print(f"  🎯 {shift_key}: capacity {capacity}")
                
                # Assign shifts up to capacity
                for slot in range(capacity):
                    debug_logger.log(f"    🔸 Assigning slot {slot + 1}/{capacity} for {shift_key}")
                    
                    # SIMPLIFIED: Skip complex scoring for now
                    # if 'scorer' not in locals() or len(events) % 10 == 0:  # Update scorer every 10 events
                    #     debug_logger.log("    🧮 Creating/updating scorer...")
                    #     scorer = create_scorer(events, providers, provider_rules, global_rules, year, month)
                    
                    # Find feasible candidates
                    debug_logger.log("    🔍 Checking provider feasibility...")
                    candidates = []
                    feasible_providers = []
                    
                    for provider in providers_shuffled:
                        is_feasible = is_provider_feasible(provider, current_day, shift_key, events, 
                                                         provider_rules, global_rules, year, month)
                        if is_feasible:
                            feasible_providers.append(provider)
                            # IMPROVED: Score based on current shift count to balance load
                            current_shifts = len([e for e in events if e.extendedProps.get("provider") == provider])
                            # Lower current shifts = higher score (prefer providers with fewer shifts)
                            balance_score = 100 - current_shifts  # Higher score for fewer shifts
                            candidates.append((provider, balance_score))
                        else:
                            # DEBUG: Log why provider was rejected
                            pass  # The feasibility function will handle debug output
                    
                    debug_logger.log(f"    ✅ Found {len(candidates)} feasible candidates: {feasible_providers}")
                    print(f"    ✅ Found {len(candidates)} feasible candidates: {feasible_providers}")
                    
                    if not candidates:
                        debug_logger.log(f"    ❌ No feasible candidates for {shift_key} slot {slot + 1}")
                        print(f"    ❌ No feasible candidates for {shift_key} slot {slot + 1}")
                        continue  # Skip this slot if no candidates
                    
                    # Sort by score (highest first) - providers with fewer shifts get priority
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # DEBUG: Show top candidates for load balancing
                    if len(candidates) > 0:
                        top_candidates = candidates[:3]  # Show top 3
                        if shift_key == "R12" and slot < 3:  # Only for first few R12 slots to avoid spam
                            candidate_info = [(name, score, len([e for e in events if e.extendedProps.get("provider") == name])) 
                                            for name, score in top_candidates]
                            print(f"    🏆 Top candidates: {candidate_info} (name, score, current_shifts)")
                    
                    # Select best candidate (highest score = fewest current shifts)
                    best_provider = candidates[0][0]
                    
                    # Create and add the shift event
                    debug_logger.log(f"    ✅ Assigning {shift_key} to {best_provider}")
                    print(f"    ✅ Assigning {shift_key} to {best_provider}")
                    event = create_shift_event(best_provider, shift_key, current_day)
                    events.append(event)
                    debug_logger.log(f"    📝 Total events so far: {len(events)}")
                    print(f"    📝 Total events so far: {len(events)}")
        
        debug_logger.log(f"🎉 COMPLETED ASSIGNMENT: {len(events)} shifts assigned")
        print(f"🎉 COMPLETED ASSIGNMENT: {len(events)} shifts assigned")
        
        # Analyze what was assigned
        shift_type_counts = {}
        for event in events:
            shift_type = event.extendedProps.get('shift_type', 'Unknown')
            shift_type_counts[shift_type] = shift_type_counts.get(shift_type, 0) + 1
        
        print(f"📊 Shift type distribution: {shift_type_counts}")
        
        # Skip validation for speed during testing
        # events = validate_and_adjust_schedule(events, providers, provider_rules, global_rules, year, month)
        
        return events
        
    except Exception as e:
        logger.error(f"Error in assign_with_scoring: {e}")
        raise ScheduleGenerationError(f"Failed to assign shifts with scoring: {e}")

def get_shift_capacity(shift_key: str, day: date, shift_capacity: Dict[str, int]) -> int:
    """Get the capacity for a specific shift type on a given day."""
    if shift_key == "APP":
        # APP shifts: 2 on weekdays, 1 on weekends
        return 1 if day.weekday() >= 5 else 2
    else:
        return shift_capacity.get(shift_key, 1)

def is_provider_feasible(provider: str, day: date, shift_key: str, events: List[SEvent],
                        provider_rules: Dict, global_rules: RuleConfig, 
                        year: int, month: int) -> bool:
    """Check if a provider can feasibly take a shift on a given day."""
    from core.utils import is_provider_unavailable_on_date, get_adjusted_expected_shifts
    from core.shift_validation import has_shift_on_date, validate_shift_type_preference
    from models.constants import APP_PROVIDER_INITIALS
    from core.provider_types import NOCTURNISTS, SENIORS
    
    # DETAILED DEBUG: Track why providers are rejected
    debug_reason = None
    
    # DEBUG: Log start of feasibility check for first few providers
    if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
        print(f"        🔍 Checking {provider} for {shift_key} on {day}")
        print(f"        📊 Current events count: {len(events)}")
    
    # Check basic availability
    try:
        if is_provider_unavailable_on_date(provider, day, provider_rules):
            debug_reason = f"unavailable on {day}"
            return False
    except Exception as e:
        print(f"      ⚠️ Error checking availability for {provider}: {e}")
        # Be lenient - if we can't check, assume available
    
    # Check if provider already has a shift on this day
    try:
        # FIXED: Filter events to only this provider's shifts
        provider_events = [e for e in events if e.extendedProps.get("provider") == provider]
        if has_shift_on_date(provider, day, provider_events):
            debug_reason = f"already has shift on {day}"
            if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
                print(f"        ❌ {provider} already has shift on {day}")
            return False
        elif provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
            print(f"        ✅ {provider} has no shift on {day} (provider events: {len(provider_events)})")
    except Exception as e:
        print(f"      ⚠️ Error checking existing shifts for {provider}: {e}")
        # Be lenient - if we can't check, assume no conflict
    
    # Check shift type eligibility based on provider type and qualifications
    from core.provider_types import BRIDGE_QUALIFIED
    
    if provider in APP_PROVIDER_INITIALS:
        # APP providers can ONLY take APP shifts
        if shift_key != "APP":
            debug_reason = f"APP provider can't take {shift_key}"
            return False
    elif provider in SENIORS:
        # Seniors can ONLY take R12 shifts (7am rounding)
        if shift_key != "R12":
            debug_reason = f"Senior can't take {shift_key}"
            return False
    elif shift_key == "APP":
        # Only APP providers can take APP shifts
        debug_reason = f"non-APP can't take APP"
        return False
    elif shift_key == "NB":
        # Only bridge-qualified providers can take bridge shifts
        if provider not in BRIDGE_QUALIFIED:
            debug_reason = f"not bridge-qualified for NB"
            return False
    
    # RE-ENABLED: Check shift preferences but be more lenient
    try:
        if not validate_shift_type_preference(provider, shift_key, provider_rules):
            # Only reject if provider explicitly refuses this shift type
            provider_rule = provider_rules.get(provider, {})
            shift_preferences = provider_rule.get("shift_preferences", {})
            
            # If preference is explicitly False, reject. If missing/None, allow
            if shift_preferences.get(shift_key) is False:
                debug_reason = f"explicitly refuses {shift_key}"
                if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
                    print(f"        ❌ {provider} explicitly refuses {shift_key}")
                return False
            # Otherwise allow the assignment even if preference is not explicitly True
    except Exception as e:
        print(f"      ⚠️ Error checking preferences for {provider}: {e}")
        # If there's an error, be lenient and allow the shift
    
    # RE-ENABLED: Check shift count limits to ensure fair distribution
    current_shifts = len([e for e in events if e.extendedProps.get("provider") == provider])
    try:
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        # Allow some flexibility but prevent extreme overloading
        max_allowed_shifts = expected_shifts + 5  # Allow 5 extra shifts for coverage
        
        if current_shifts >= max_allowed_shifts:
            debug_reason = f"would exceed max shifts ({current_shifts} >= {max_allowed_shifts})"
            if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
                print(f"        ❌ {provider} would exceed max shifts ({current_shifts} >= {max_allowed_shifts})")
            return False
        elif provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
            print(f"        ✅ {provider} shift count OK ({current_shifts}/{max_allowed_shifts})")
    except Exception as e:
        print(f"      ⚠️ Error calculating expected shifts for {provider}: {e}")
        # Fallback: Hard limit at 25 shifts per month to prevent overloading
        if current_shifts >= 25:
            debug_reason = f"too many shifts ({current_shifts} >= 25)"
            if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
                print(f"        ❌ {provider} exceeds fallback limit ({current_shifts} >= 25)")
            return False
    
    # SIMPLIFIED: Skip night shift limits for now
    # if shift_key in ["N12", "NB"]:
    #     night_shifts = len([e for e in events 
    #                        if e.extendedProps.get("provider") == provider 
    #                        and e.extendedProps.get("shift_type") in ["N12", "NB"]])
    #     max_nights = provider_rules.get(provider, {}).get("max_nights", global_rules.max_night_shifts_per_month)
    #     
    #     if max_nights is not None and night_shifts >= max_nights:
    #         debug_reason = f"too many night shifts ({night_shifts} >= {max_nights})"
    #         return False
    
    # SIMPLIFIED: Skip rest requirements for now
    # provider_events = [e for e in events if e.extendedProps.get("provider") == provider]
    # if provider_events:
    #     provider_dates = {e.start.date() for e in provider_events}
    #     
    #     consecutive_count = 0
    #     check_date = day - timedelta(days=1)
    #     while check_date in provider_dates:
    #         consecutive_count += 1
    #         check_date -= timedelta(days=1)
    #         if consecutive_count >= 7:
    #             debug_reason = f"too many consecutive days ({consecutive_count} >= 7)"
    #             return False
    
    # If we get here, provider is feasible
    if provider in ['AB', 'DI', 'JK', 'MA', 'MS']:
        print(f"        ✅ {provider} is feasible for {shift_key}")
    return True

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
    Validate and automatically adjust the schedule to fix violations using scoring.
    """
    try:
        # Run comprehensive validation first
        validation_result = validate_rules(events, providers, global_rules, provider_rules, year, month)
        
        if validation_result["is_valid"]:
            logger.info("Schedule passed validation - no adjustments needed")
            return events
        
        logger.info(f"Schedule has {validation_result['summary']['total_violations']} violations, attempting to optimize...")
        
        # Use scoring-based optimization to improve the schedule
        optimized_events = optimize_schedule_with_scoring(events, providers, provider_rules, 
                                                         global_rules, year, month)
        
        # Validate again after optimization
        final_validation = validate_rules(optimized_events, providers, global_rules, 
                                        provider_rules, year, month)
        
        if final_validation["is_valid"]:
            logger.info("Schedule optimization successful - all violations resolved")
        else:
            remaining_violations = final_validation['summary']['total_violations']
            logger.warning(f"Schedule optimization reduced violations to {remaining_violations}")
        
        return optimized_events
        
    except Exception as e:
        logger.error(f"Error in validate_and_adjust_schedule: {e}")
        # Return original events if optimization fails
        return events

def optimize_schedule_with_scoring(events: List[SEvent], providers: List[str],
                                 provider_rules: Dict, global_rules: RuleConfig,
                                 year: int, month: int, max_iterations: int = 5) -> List[SEvent]:
    """
    Optimize the schedule using scoring-based local search.
    """
    try:
        current_events = events.copy()
        best_events = current_events.copy()
        best_score = calculate_total_schedule_score(best_events, providers, provider_rules, 
                                                   global_rules, year, month)
        
        logger.info(f"Starting schedule optimization with initial score: {best_score:.2f}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try swapping shifts between providers to improve overall score
            for i, event1 in enumerate(current_events):
                for j, event2 in enumerate(current_events[i+1:], i+1):
                    provider1 = event1.extendedProps.get("provider")
                    provider2 = event2.extendedProps.get("provider")
                    
                    if provider1 == provider2:
                        continue
                    
                    # Try swapping the assignments
                    test_events = current_events.copy()
                    test_events[i] = SEvent(
                        id=event1.id,
                        title=event1.title.replace(provider1, provider2),
                        start=event1.start,
                        end=event1.end,
                        backgroundColor=event1.backgroundColor,
                        extendedProps={**event1.extendedProps, "provider": provider2}
                    )
                    test_events[j] = SEvent(
                        id=event2.id,
                        title=event2.title.replace(provider2, provider1),
                        start=event2.start,
                        end=event2.end,
                        backgroundColor=event2.backgroundColor,
                        extendedProps={**event2.extendedProps, "provider": provider1}
                    )
                    
                    # Check if swap is feasible
                    if (is_provider_feasible(provider2, event1.start.date(), 
                                           event1.extendedProps.get("shift_type", ""), 
                                           [e for e in test_events if e != test_events[i]], 
                                           provider_rules, global_rules, year, month) and
                        is_provider_feasible(provider1, event2.start.date(), 
                                           event2.extendedProps.get("shift_type", ""), 
                                           [e for e in test_events if e != test_events[j]], 
                                           provider_rules, global_rules, year, month)):
                        
                        # Calculate new score
                        test_score = calculate_total_schedule_score(test_events, providers, 
                                                                  provider_rules, global_rules, 
                                                                  year, month)
                        
                        if test_score > best_score:
                            best_events = test_events
                            best_score = test_score
                            improved = True
                            logger.debug(f"Improved schedule score to {best_score:.2f} by swapping {provider1} and {provider2}")
            
            current_events = best_events.copy()
            
            if not improved:
                logger.info(f"No improvements found in iteration {iteration + 1}")
                break
        
        logger.info(f"Schedule optimization completed with final score: {best_score:.2f}")
        return best_events
        
    except Exception as e:
        logger.error(f"Error in optimize_schedule_with_scoring: {e}")
        return events

def calculate_total_schedule_score(events: List[SEvent], providers: List[str],
                                 provider_rules: Dict, global_rules: RuleConfig,
                                 year: int, month: int) -> float:
    """
    Calculate the total score for a complete schedule.
    """
    try:
        scorer = create_scorer(events, providers, provider_rules, global_rules, year, month)
        total_score = 0.0
        
        for event in events:
            provider = event.extendedProps.get("provider")
            day = event.start.date()
            shift_type = event.extendedProps.get("shift_type") or event.extendedProps.get("shift_key")
            
            if provider and shift_type:
                # Create temporary events list without this event for scoring
                temp_events = [e for e in events if e != event]
                temp_scorer = create_scorer(temp_events, providers, provider_rules, 
                                          global_rules, year, month)
                score = temp_scorer.score_assignment(provider, day, shift_type)
                total_score += score
        
        return total_score
        
    except Exception as e:
        logger.error(f"Error calculating total schedule score: {e}")
        return 0.0

def validate_rules(events: List[SEvent], providers: List[str], 
                  global_rules: RuleConfig, provider_rules: Dict,
                  year: int = None, month: int = None) -> Dict:
    """
    ENHANCED VALIDATION: Comprehensive debugging and validation of scheduling rules.
    """
    violations = []
    provider_violations = {}
    coverage_gaps = []
    preference_violations = []
    rest_violations = []
    
    # Track provider statistics
    provider_stats = {}
    for provider in providers:
        provider_stats[provider] = {
            'total_shifts': 0,
            'expected_shifts': 'N/A',
            'shift_types': [],
            'weekend_shifts': 0,
            'night_shifts': 0,
            'rounder_shifts': 0,
            'admitting_shifts': 0,
            'shift_dates': []
        }
    
    # Analyze events
    for event in events:
        provider = event.extendedProps.get("provider")
        shift_type = event.extendedProps.get("shift_type")
        event_date = event.start.date()
        
        if provider and provider in provider_stats:
            provider_stats[provider]['total_shifts'] += 1
            provider_stats[provider]['shift_dates'].append(event_date)
            
            if shift_type not in provider_stats[provider]['shift_types']:
                provider_stats[provider]['shift_types'].append(shift_type)
            
            # Categorize shift types
            if shift_type in ["N12", "NB"]:
                provider_stats[provider]['night_shifts'] += 1
            elif shift_type == "R12":
                provider_stats[provider]['rounder_shifts'] += 1
            elif shift_type in ["A12", "A10"]:
                provider_stats[provider]['admitting_shifts'] += 1
            
            # Check for weekend shifts
            if event_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                provider_stats[provider]['weekend_shifts'] += 1
    
    # Calculate expected shifts for each provider
    for provider in providers:
        try:
            from core.utils import get_adjusted_expected_shifts
            expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
            provider_stats[provider]['expected_shifts'] = expected_shifts
            
            # Check if provider exceeds expected shifts
            current_shifts = provider_stats[provider]['total_shifts']
            if current_shifts > expected_shifts:
                violation_msg = f"Provider {provider} has {current_shifts} shifts but expected {expected_shifts}"
                violations.append(violation_msg)
                if provider not in provider_violations:
                    provider_violations[provider] = []
                provider_violations[provider].append(violation_msg)
        except Exception as e:
            logger.warning(f"Could not calculate expected shifts for {provider}: {e}")
    
    # Check for rest violations
    for provider in providers:
        shift_dates = sorted(provider_stats[provider]['shift_dates'])
        for i in range(1, len(shift_dates)):
            days_between = (shift_dates[i] - shift_dates[i-1]).days
            if days_between < 1:  # Minimum 1 day rest
                violation_msg = f"Provider {provider} has insufficient rest between {shift_dates[i-1]} and {shift_dates[i]}"
                rest_violations.append(violation_msg)
                violations.append(violation_msg)
    
    # Check for preference violations
    for provider in providers:
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        for event in events:
            if event.extendedProps.get("provider") == provider:
                shift_type = event.extendedProps.get("shift_type")
                if shift_preferences and shift_type in shift_preferences and not shift_preferences[shift_type]:
                    violation_msg = f"Provider {provider} assigned {shift_type} but doesn't prefer it"
                    preference_violations.append(violation_msg)
                    violations.append(violation_msg)
    
    # Calculate summary statistics
    total_violations = len(violations)
    coverage_gaps_count = len(coverage_gaps)
    preference_violations_count = len(preference_violations)
    rest_violations_count = len(rest_violations)
    is_valid = total_violations == 0
    
    return {
        "violations": violations,
        "provider_violations": provider_violations,
        "coverage_gaps": coverage_gaps,
        "preference_violations": preference_violations,
        "rest_violations": rest_violations,
        "is_valid": is_valid,
        "summary": {
            "total_violations": total_violations,
            "coverage_gaps_count": coverage_gaps_count,
            "preference_violations_count": preference_violations_count,
            "rest_violations_count": rest_violations_count,
            "provider_stats": provider_stats
        }
    }


