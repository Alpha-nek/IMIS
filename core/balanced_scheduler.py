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
    BLOCK-BASED ALGORITHM: Create shift blocks for regular providers.
    
    Algorithm:
    1. Calculate expected shifts for each regular provider
    2. Create shift blocks (3-7 shifts) with admitting → rounding pattern
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
    
    logger.info(f"Starting block-based algorithm for {len(providers)} providers")
    logger.info(f"Provider expected shifts: {provider_expected_shifts}")
    
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
    BLOCK PATTERN: 1-2 admitting shifts → rounding shifts
    """
    provider_blocks = {}
    
    for provider in providers:
        expected_shifts = provider_expected_shifts.get(provider, 15)
        current_shifts = 0
        blocks = []
        
        logger.info(f"Creating blocks for {provider} (expected: {expected_shifts} shifts)")
        
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
            
            # Create block with admitting → rounding pattern
            block = create_admitting_to_rounding_block(provider, block_size, provider_rules)
            blocks.append(block)
            
            current_shifts += block_size
            logger.info(f"  Created {block_size}-shift block: {block['shift_sequence']}")
        
        provider_blocks[provider] = blocks
    
    return provider_blocks

def create_admitting_to_rounding_block(provider: str, block_size: int, provider_rules: Dict) -> Dict:
    """
    Create a block with diverse shift pattern for regular providers.
    Pattern: 1-2 admitting shifts → rounding shifts → night shifts (if needed)
    """
    from core.provider_types import get_allowed_shift_types
    
    allowed_shifts = get_allowed_shift_types(provider)
    admitting_shifts = [shift for shift in ["A12", "A10"] if shift in allowed_shifts]
    rounding_shifts = [shift for shift in ["R12"] if shift in allowed_shifts]
    night_shifts = [shift for shift in ["N12"] if shift in allowed_shifts]
    
    # Determine shift types (prefer A12, fallback to A10)
    admitting_shift = "A12" if "A12" in admitting_shifts else "A10" if "A10" in admitting_shifts else "R12"
    rounding_shift = "R12" if "R12" in rounding_shifts else admitting_shift
    night_shift = "N12" if "N12" in night_shifts else rounding_shift
    
    # Create shift sequence based on block size
    shift_sequence = []
    
    if block_size >= 5:
        # Large block: 1-2 admitting → 2-3 rounding → 1 night
        num_admitting = min(2, block_size - 3)  # At least 2 rounding + 1 night
        num_rounding = block_size - num_admitting - 1  # Leave room for 1 night
        num_night = 1
        
        for i in range(num_admitting):
            shift_sequence.append(admitting_shift)
        for i in range(num_rounding):
            shift_sequence.append(rounding_shift)
        for i in range(num_night):
            shift_sequence.append(night_shift)
            
    elif block_size >= 3:
        # Medium block: 1 admitting → 1-2 rounding → 1 night (if space)
        num_admitting = 1
        num_rounding = block_size - num_admitting - 1 if block_size > 3 else block_size - num_admitting
        num_night = block_size - num_admitting - num_rounding
        
        for i in range(num_admitting):
            shift_sequence.append(admitting_shift)
        for i in range(num_rounding):
            shift_sequence.append(rounding_shift)
        for i in range(num_night):
            shift_sequence.append(night_shift)
    else:
        # Small block: All rounding or admitting
        for i in range(block_size):
            shift_sequence.append(rounding_shift)
    
    return {
        "provider": provider,
        "shift_sequence": shift_sequence,
        "size": block_size,
        "assigned": False
    }

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
    
    logger.info(f"Assigning blocks for providers: {sorted_providers}")
    
    for provider in sorted_providers:
        blocks = provider_blocks[provider]
        
        for block in blocks:
            if block["assigned"]:
                continue
            
            # Find available dates for this block with proper spacing
            available_dates = find_available_dates_for_block_with_spacing(
                provider, block["shift_sequence"], month_days,
                provider_shifts, provider_rules, global_rules, year, month
            )
            
            if len(available_dates) >= block["size"]:
                # Assign the block
                block_events = assign_shift_block(provider, block["shift_sequence"], 
                                                available_dates[:block["size"]], provider_shifts)
                events.extend(block_events)
                block["assigned"] = True
                
                logger.info(f"✅ Assigned {block['size']}-shift block to {provider} starting {available_dates[0]}")
                logger.info(f"   Shift sequence: {block['shift_sequence']}")
            else:
                logger.warning(f"❌ Could not assign {block['size']}-shift block to {provider} (only {len(available_dates)} dates available)")
    
    return events

def find_available_dates_for_block_with_spacing(provider: str, shift_sequence: List[str], 
                                              month_days: List[date], provider_shifts: Dict,
                                              provider_rules: Dict, global_rules: RuleConfig,
                                              year: int, month: int) -> List[date]:
    """
    Find available dates for a block with proper spacing (2+ days rest between blocks).
    """
    from core.utils import is_provider_unavailable_on_date
    from core.shift_validation import has_shift_on_date, validate_shift_type_preference, has_sufficient_rest
    
    available_dates = []
    
    for start_idx in range(len(month_days) - len(shift_sequence) + 1):
        # Check if we can place the block starting at this date
        block_dates = month_days[start_idx:start_idx + len(shift_sequence)]
        can_place_block = True
        
        for i, (day, shift_type) in enumerate(zip(block_dates, shift_sequence)):
            # Check if provider is available on this day
            if is_provider_unavailable_on_date(provider, day, provider_rules):
                can_place_block = False
                break
            
            # Check if provider already has a shift on this day
            if has_shift_on_date(provider, day, provider_shifts[provider]):
                can_place_block = False
                break
            
            # Check shift type preference
            if not validate_shift_type_preference(provider, shift_type, provider_rules):
                can_place_block = False
                break
            
            # Check rest requirements - ENFORCE 2+ days rest between blocks
            min_rest_days = 2
            if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
                min_rest_days = max(2, global_rules.min_days_between_shifts)
            
            if not has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
                can_place_block = False
                break
        
        if can_place_block:
            available_dates = block_dates
            break
    
    return available_dates

def assign_shift_block(provider: str, shift_sequence: List[str], dates: List[date], 
                      provider_shifts: Dict) -> List[SEvent]:
    """
    Assign a block of shifts to a provider.
    """
    events = []
    
    for day, shift_type in zip(dates, shift_sequence):
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
    Find the best provider for a single shift (for gap filling only).
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
        
        # Check rest requirements (1 day minimum for individual shifts)
        if not has_sufficient_rest(provider, day, provider_shifts[provider], 1):
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = provider_expected_shifts.get(provider, 15)
        
        if current_shifts >= expected_shifts:
            continue
        
        # Calculate score (lower is better)
        score = current_shifts / expected_shifts
        
        # Bonus for providers who prefer this shift type
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        if shift_preferences.get(shift_type, False):
            score -= 0.1  # Small bonus for preference
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    return best_provider
