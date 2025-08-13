# =============================================================================
# Balanced Scheduling Algorithm - Day-by-Day with Smart Block Building
# =============================================================================

import logging
import random
from typing import Dict, List, Optional, Set
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
    DAY-BY-DAY ALGORITHM with SMART BLOCK BUILDING: Assign shifts day by day with sophisticated scoring.
    
    Algorithm:
    1. Process each day and shift type
    2. Use smart scoring to naturally build blocks (4-7 days preferred)
    3. Enforce shift type consistency within blocks
    4. Ensure proper rest periods and provider limits
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
    
    logger.info(f"Starting day-by-day algorithm for {len(providers)} providers")
    logger.info(f"Provider expected shifts: {provider_expected_shifts}")
    
    # Track provider statistics
    provider_counts = {p: 0 for p in providers}
    provider_nights = {p: 0 for p in providers}
    provider_days = {p: set() for p in providers}  # Track assigned days
    
    # Get block size preferences
    min_block_size = getattr(global_rules, 'min_block_size', 3)
    max_block_size = getattr(global_rules, 'max_block_size', 7)
    
    # Define shift priority (most important first)
    shift_priority = ["N12", "A12", "A10", "R12", "NB"]
    
    # Add randomness to provider selection
    providers_shuffled = providers.copy()
    random.shuffle(providers_shuffled)
    
    # Process each day
    for current_day in month_days:
        logger.info(f"Processing day: {current_day}")
        
        # Process each shift type
        for shift_key in shift_priority:
            if shift_key not in shift_capacity:
                continue
                
            capacity = shift_capacity[shift_key]
            assigned_count = count_shifts_on_date(current_day, shift_key, provider_shifts)
            remaining_slots = capacity - assigned_count
            
            logger.info(f"  {shift_key}: {assigned_count}/{capacity} slots filled, {remaining_slots} remaining")
            
            # Fill remaining slots
            for slot in range(remaining_slots):
                candidates = [prov for prov in providers_shuffled if is_provider_available(
                    prov, current_day, shift_key, provider_shifts, provider_rules, 
                    global_rules, provider_expected_shifts, provider_counts, provider_nights, year, month
                )]
                
                if not candidates:
                    logger.warning(f"    ❌ No provider available for {shift_key} on {current_day}")
                    continue
                
                # Score candidates and select best
                if len(candidates) > 1:
                    scores = [(prov, calculate_provider_score(
                        prov, current_day, shift_key, provider_shifts, provider_rules,
                        global_rules, provider_expected_shifts, provider_counts, 
                        provider_nights, provider_days, min_block_size, max_block_size
                    )) for prov in candidates]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add randomness when scores are close
                    if len(scores) >= 2 and scores[0][1] > 0 and (scores[0][1] - scores[1][1]) / scores[0][1] < 0.1:
                        best_provider = random.choice(scores[:2])[0]
                    else:
                        best_provider = scores[0][0]
                else:
                    best_provider = candidates[0]
                
                # Assign the shift
                event = create_shift_event(best_provider, shift_key, current_day)
                events.append(event)
                provider_shifts[best_provider].append(event)
                provider_counts[best_provider] += 1
                provider_days[best_provider].add(current_day)
                
                if shift_key == "N12":
                    provider_nights[best_provider] += 1
                
                logger.info(f"    ✅ Assigned {shift_key} to {best_provider}")
    
    return events

def is_provider_available(provider: str, day: date, shift_key: str, provider_shifts: Dict,
                         provider_rules: Dict, global_rules: RuleConfig, 
                         provider_expected_shifts: Dict, provider_counts: Dict,
                         provider_nights: Dict, year: int, month: int) -> bool:
    """
    Check if a provider is available for a specific shift on a given day.
    """
    from core.provider_types import get_allowed_shift_types
    
    # Check if provider can do this shift type
    allowed_shifts = get_allowed_shift_types(provider)
    if shift_key not in allowed_shifts:
        return False
    
    # Check if provider is unavailable on this date
    if is_provider_unavailable_on_date(provider, day, provider_rules):
        return False
    
    # Check if provider already has a shift on this day
    if has_shift_on_date(provider, day, provider_shifts[provider]):
        return False
    
    # Check shift type preference
    if not validate_shift_type_preference(provider, shift_key, provider_rules):
        return False
    
    # Check rest requirements
    min_rest_days = 1
    if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
        min_rest_days = max(1, global_rules.min_days_between_shifts)
    
    if not has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
        return False
    
    # Check if assignment would exceed expected shifts
    current_shifts = provider_counts[provider]
    expected_shifts = provider_expected_shifts.get(provider, 15)
    if current_shifts >= expected_shifts:
        return False
    
    # Check night shift limits for regular providers
    if shift_key == "N12":
        max_nights = provider_rules.get(provider, {}).get("max_nights", 3)  # Default 3 nights/month
        if provider_nights[provider] >= max_nights:
            return False
    
    return True

def calculate_provider_score(provider: str, day: date, shift_key: str, provider_shifts: Dict,
                           provider_rules: Dict, global_rules: RuleConfig,
                           provider_expected_shifts: Dict, provider_counts: Dict,
                           provider_nights: Dict, provider_days: Dict[str, Set[date]],
                           min_block_size: int, max_block_size: int) -> float:
    """
    Calculate a score for assigning a shift to a provider.
    Higher scores are better.
    """
    score = 0.0
    
    # Get provider's assigned days
    days_set = provider_days[provider]
    current_shifts = provider_counts[provider]
    expected_shifts = provider_expected_shifts.get(provider, 15)
    
    # 1. Load balancing - prefer providers with fewer shifts
    if current_shifts < expected_shifts:
        score += 4.0
        # Bonus for building up to minimum block size
        left_run = left_run_length(days_set, day)
        if left_run > 0:
            score += 2.0
        if left_run < min_block_size:
            score += 4.0
    
    # 2. Block size optimization - prefer 4-7 day blocks, avoid 1-2 day stretches
    left_run = left_run_length(days_set, day)
    right_run = right_run_length(days_set, day)
    total_block_len = left_run + 1 + right_run
    
    # Penalty for standalone days or very short stretches
    if left_run == 0 and right_run == 0:
        score -= 6.0  # Strong penalty for standalone days
    elif total_block_len <= 2:
        score -= 3.0  # Penalty for short stretches
    elif 4 <= total_block_len <= 7:
        score += 2.0  # Bonus for optimal block size
    
    # 3. Shift type consistency within blocks - ENFORCE STRICT CONSISTENCY
    if days_set:
        # Classify shift types
        night_shifts = {"N12", "NB"}
        day_shifts = {"R12", "A12", "A10"}
        
        current_shift_type = "night" if shift_key in night_shifts else "day"
        
        # Check existing block for shift types
        block_start = day - timedelta(days=left_run)
        block_end = day + timedelta(days=right_run)
        
        block_shift_types = set()
        for event in provider_shifts[provider]:
            event_date = event.start.date()
            if block_start <= event_date <= block_end:
                block_shift_types.add(event.extendedProps.get("shift_type"))
        
        block_has_nights = any(s in night_shifts for s in block_shift_types)
        block_has_days = any(s in day_shifts for s in block_shift_types)
        
        # STRICT CONSISTENCY ENFORCEMENT
        if block_has_nights and block_has_days:
            # Mixed block - strong penalty for adding different type
            if (current_shift_type == "night" and not block_has_nights) or \
               (current_shift_type == "day" and not block_has_days):
                score -= 5.0
        elif block_has_nights and current_shift_type == "day":
            # Adding day shift to night block - very strong penalty
            score -= 8.0
        elif block_has_days and current_shift_type == "night":
            # Adding night shift to day block - very strong penalty
            score -= 8.0
        else:
            # Consistent block - strong bonus
            score += 3.0
            
        # Bonus for extending existing consistent blocks
        if left_run > 0 or right_run > 0:
            if (current_shift_type == "night" and block_has_nights and not block_has_days) or \
               (current_shift_type == "day" and block_has_days and not block_has_nights):
                score += 2.0
    
    # 4. Gentle load balancing
    score += max(0, 20 - current_shifts) * 0.01
    
    # 5. Soft penalty if this hits max block size
    if max_block_size and total_block_len == max_block_size:
        score -= 0.2
    
    # 6. Weekend incentive
    weekend_required = provider_rules.get(provider, {}).get("require_weekend", False)
    if day.weekday() >= 5 and weekend_required and provider_weekend_count(provider, provider_shifts) == 0:
        score += 3.0
    
    # 7. Day/night ratio preference
    try:
        provider_rule = provider_rules.get(provider, {})
        ratio = provider_rule.get("day_night_ratio", None)  # percent of day shifts
        if ratio is not None:
            desired_night_frac = max(0.0, (100.0 - float(ratio)) / 100.0)
            cur_nights = provider_nights[provider]
            est_total = current_shifts + 1
            est_nights = cur_nights + (1 if shift_key == "N12" else 0)
            est_night_frac = est_nights / max(1, est_total)
            
            # Penalize if assigning night would push above desired fraction
            if shift_key == "N12" and est_night_frac > desired_night_frac + 0.05:
                score -= 2.0
            # Small bonus if assigning day reduces night fraction toward target
            if shift_key != "N12" and est_night_frac < desired_night_frac - 0.10:
                score += 0.5
    except Exception:
        pass
    
    return score

def left_run_length(days_set: Set[date], day: date) -> int:
    """Calculate the length of consecutive days to the left of the given day."""
    run = 0
    current = day - timedelta(days=1)
    while current in days_set:
        run += 1
        current -= timedelta(days=1)
    return run

def right_run_length(days_set: Set[date], day: date) -> int:
    """Calculate the length of consecutive days to the right of the given day."""
    run = 0
    current = day + timedelta(days=1)
    while current in days_set:
        run += 1
        current += timedelta(days=1)
    return run

def provider_weekend_count(provider: str, provider_shifts: Dict) -> int:
    """Count weekend shifts for a provider."""
    count = 0
    for event in provider_shifts[provider]:
        if event.start.weekday() >= 5:  # Saturday = 5, Sunday = 6
            count += 1
    return count
