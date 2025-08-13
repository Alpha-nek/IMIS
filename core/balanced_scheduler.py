# =============================================================================
# Balanced Scheduling Algorithm
# =============================================================================

import logging
from typing import Dict, List, Optional
from datetime import date
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
        from models.constants import APP_PROVIDER_INITIALS
        if provider not in APP_PROVIDER_INITIALS:
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
                    event = create_shift_event(best_provider, shift_type, day)
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
                    event = create_shift_event(best_provider, shift_type, day)
                    events.append(event)
                    provider_shifts[best_provider].append(event)
    
    return events

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
            
            if has_shift_on_date(provider, day, provider_shifts[provider]):
                continue
            
            # Check shift type preference (relaxed)
            if not validate_shift_type_preference(provider, shift_type, provider_rules):
                continue
            
            # Check rest requirements (relaxed - allow 1 day rest)
            if not has_sufficient_rest(provider, day, provider_shifts[provider], 1):
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
        
        if has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Check shift type preference
        if not validate_shift_type_preference(provider, shift_type, provider_rules):
            continue
        
        # Check rest requirements (1 day minimum)
        if not has_sufficient_rest(provider, day, provider_shifts[provider], 1):
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
            if not has_shift_on_date(provider, day, provider_shifts[provider]):
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
        if has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Create shift event
        event = create_shift_event(provider, shift_type, day)
        events.append(event)
    
    return events
