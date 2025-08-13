# =============================================================================
# Balanced Scheduling Algorithm - Block-Based Approach
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
    BLOCK-BASED ALGORITHM: Create optimized shift blocks with proper rest periods.
    
    Algorithm:
    1. Calculate expected shifts for each provider
    2. Create shift blocks (3-7 shifts) for each provider
    3. Assign blocks with proper spacing (2+ days rest between blocks)
    4. Fill any remaining gaps with individual shifts only if necessary
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
    
    # Step 1: Create shift blocks for each provider
    provider_blocks = create_provider_shift_blocks(providers, provider_expected_shifts, 
                                                 month_days, provider_rules, global_rules)
    
    # Step 2: Assign blocks with proper spacing (2+ days rest between blocks)
    events.extend(assign_blocks_with_proper_spacing(provider_blocks, month_days, shift_capacity,
                                                  provider_rules, global_rules, provider_shifts, year, month))
    
    # Step 3: Fill any remaining gaps with individual shifts (only if absolutely necessary)
    events.extend(fill_remaining_gaps_individual(month_days, providers, shift_capacity,
                                               provider_rules, global_rules, provider_shifts,
                                               provider_expected_shifts, year, month))
    
    return events

def create_provider_shift_blocks(providers: List[str], provider_expected_shifts: Dict,
                               month_days: List[date], provider_rules: Dict, 
                               global_rules: RuleConfig) -> Dict[str, List[Dict]]:
    """
    Create shift blocks for each provider based on their expected shifts.
    Prioritize larger blocks for better continuity.
    """
    provider_blocks = {}
    
    for provider in providers:
        expected_shifts = provider_expected_shifts.get(provider, 15)
        current_shifts = 0
        blocks = []
        
        # Create blocks of 3-7 shifts until we reach expected shifts
        while current_shifts < expected_shifts:
            remaining_shifts = expected_shifts - current_shifts
            
            # Determine block size (3-7 shifts, but no more than remaining shifts)
            if remaining_shifts < 3:
                # If less than 3 shifts remain, create a small block
                block_size = remaining_shifts
            else:
                # Prefer larger blocks (5-7 shifts) for better continuity
                max_block_size = min(7, remaining_shifts)
                if max_block_size >= 5:
                    # 70% chance of larger blocks (5-7), 30% chance of smaller blocks (3-4)
                    if random.random() < 0.7:
                        block_size = random.randint(5, max_block_size)
                    else:
                        block_size = random.randint(3, min(4, max_block_size))
                else:
                    block_size = random.randint(3, max_block_size)
            
            # Determine shift type for this block
            shift_type = determine_shift_type_for_block(provider, provider_rules)
            
            blocks.append({
                "provider": provider,
                "shift_type": shift_type,
                "size": block_size,
                "assigned": False
            })
            
            current_shifts += block_size
        
        provider_blocks[provider] = blocks
    
    return provider_blocks

def determine_shift_type_for_block(provider: str, provider_rules: Dict) -> str:
    """
    Determine the best shift type for a provider's block.
    BALANCED APPROACH: Distribute shift types across providers.
    """
    from core.provider_types import get_provider_type, get_allowed_shift_types
    
    # Get provider's type and allowed shift types
    provider_type = get_provider_type(provider)
    allowed_shift_types = get_allowed_shift_types(provider)
    
    # Get provider's shift preferences
    provider_rule = provider_rules.get(provider, {})
    shift_preferences = provider_rule.get("shift_preferences", {})
    
    # For regular providers, use a balanced approach
    if provider_type == "REGULAR":
        # Check if provider has explicit preferences
        preferred_shifts = [shift for shift, preferred in shift_preferences.items() 
                          if preferred and shift in allowed_shift_types]
        
        if preferred_shifts:
            # If they have preferences, use them
            return random.choice(preferred_shifts)
        else:
            # BALANCED DISTRIBUTION: Distribute shift types evenly
            # Use a weighted random selection to ensure good distribution
            shift_weights = {
                "R12": 0.4,  # 40% rounding shifts
                "A12": 0.25, # 25% 7am admitting
                "A10": 0.2,  # 20% 10am admitting  
                "N12": 0.15  # 15% night shifts
            }
            
            # Filter to only allowed shift types
            available_shifts = [shift for shift in allowed_shift_types if shift in shift_weights]
            if available_shifts:
                # Normalize weights for available shifts
                total_weight = sum(shift_weights[shift] for shift in available_shifts)
                if total_weight > 0:
                    normalized_weights = [shift_weights[shift] / total_weight for shift in available_shifts]
                    return random.choices(available_shifts, weights=normalized_weights)[0]
            
            # Fallback to random selection from allowed types
            return random.choice(allowed_shift_types)
    
    # For other provider types, use their specific rules
    elif provider_type == "SENIOR":
        return "R12"  # Seniors only do rounding
    elif provider_type == "NOCTURNIST":
        return random.choice(["N12", "NB"])  # Nocturnists do night shifts
    elif provider_type == "APP":
        return "APP"  # APPs do APP shifts
    else:
        # Default fallback
        return "R12"

def assign_blocks_with_proper_spacing(provider_blocks: Dict[str, List[Dict]], month_days: List[date],
                                    shift_capacity: Dict[str, int], provider_rules: Dict,
                                    global_rules: RuleConfig, provider_shifts: Dict,
                                    year: int, month: int) -> List[SEvent]:
    """
    Assign blocks with proper spacing to ensure 2+ days rest between blocks.
    """
    events = []
    
    # Sort providers by number of blocks (assign those with more blocks first)
    sorted_providers = sorted(provider_blocks.keys(), 
                            key=lambda p: len(provider_blocks[p]), reverse=True)
    
    for provider in sorted_providers:
        blocks = provider_blocks[provider]
        
        for block in blocks:
            if block["assigned"]:
                continue
            
            # Find available dates for this block with proper spacing
            available_dates = find_available_dates_for_block_with_spacing(
                provider, block["shift_type"], block["size"], month_days,
                provider_shifts, provider_rules, global_rules, year, month
            )
            
            if len(available_dates) >= block["size"]:
                # Assign the block
                block_events = assign_shift_block(provider, block["shift_type"], 
                                                available_dates[:block["size"]], provider_shifts)
                events.extend(block_events)
                block["assigned"] = True
                
                logger.info(f"✅ Assigned {block['size']} {block['shift_type']} shifts to {provider} starting {available_dates[0]}")
    
    return events

def find_available_dates_for_block_with_spacing(provider: str, shift_type: str, block_size: int,
                                              month_days: List[date], provider_shifts: Dict,
                                              provider_rules: Dict, global_rules: RuleConfig,
                                              year: int, month: int) -> List[date]:
    """
    Find available dates for a block with proper spacing (2+ days rest between blocks).
    """
    from core.utils import is_provider_unavailable_on_date
    from core.shift_validation import has_shift_on_date, validate_shift_type_preference, has_sufficient_rest
    
    # Get month days
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

def fill_remaining_gaps_individual(month_days: List[date], providers: List[str],
                                 shift_capacity: Dict[str, int], provider_rules: Dict,
                                 global_rules: RuleConfig, provider_shifts: Dict,
                                 provider_expected_shifts: Dict, year: int, month: int) -> List[SEvent]:
    """
    Fill any remaining gaps with individual shifts (only if absolutely necessary).
    This should be minimal with proper block assignment.
    IMPROVED: Better coverage of all shift types.
    """
    events = []
    
    # Define priority order for shift types (most important first)
    shift_priority = ["N12", "A12", "A10", "R12", "NB", "APP"]
    
    for day in month_days:
        for shift_type in shift_priority:
            if shift_type not in shift_capacity:
                continue
                
            capacity = shift_capacity[shift_type]
            assigned_count = count_shifts_on_date(day, shift_type, provider_shifts)
            remaining_slots = capacity - assigned_count
            
            for slot in range(remaining_slots):
                # Find best available provider for this shift type
                best_provider = find_best_provider_for_single_shift(day, shift_type, providers,
                                                                  provider_shifts, provider_rules,
                                                                  global_rules, provider_expected_shifts, year, month)
                
                if best_provider:
                    event = create_shift_event(best_provider, shift_type, day)
                    events.append(event)
                    provider_shifts[best_provider].append(event)
                    logger.info(f"✅ Gap fill: Assigned {shift_type} to {best_provider} on {day}")
    
    return events

def find_best_provider_for_single_shift(day: date, shift_type: str, providers: List[str],
                                      provider_shifts: Dict, provider_rules: Dict,
                                      global_rules: RuleConfig, provider_expected_shifts: Dict,
                                      year: int, month: int) -> Optional[str]:
    """
    Find the best provider for a single shift.
    IMPROVED: Better provider selection for different shift types.
    """
    from core.provider_types import get_provider_type, get_allowed_shift_types
    
    best_provider = None
    best_score = float('inf')
    
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
        
        # Check shift type preference
        if not validate_shift_type_preference(provider, shift_type, provider_rules):
            continue
        
        # Check rest requirements (1 day minimum for individual shifts)
        if not has_sufficient_rest(provider, day, provider_shifts[provider], 1):
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts >= expected_shifts:
            continue
        
        # Calculate score (lower is better)
        # Prefer providers with fewer shifts
        score = current_shifts / expected_shifts
        
        # Bonus for providers who prefer this shift type
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        if shift_preferences.get(shift_type, False):
            score -= 0.1  # Small bonus for preference
        
        # Bonus for providers who haven't done this shift type recently
        recent_shifts = [s for s in provider_shifts[provider] if hasattr(s, 'extendedProps')]
        recent_shift_types = [s.extendedProps.get("shift_type") for s in recent_shifts[-3:]]  # Last 3 shifts
        if shift_type not in recent_shift_types:
            score -= 0.05  # Small bonus for variety
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    return best_provider
