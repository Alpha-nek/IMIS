# =============================================================================
# Core Scheduling Logic for IMIS Scheduler
# =============================================================================

import random
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from collections import defaultdict

from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS
from models.data_models import RuleConfig, Provider, SEvent
from core.utils import (
    is_holiday, get_holiday_adjusted_capacity, parse_time, 
    date_range, month_start_end, make_month_days,
    _expand_vacation_dates, is_provider_unavailable_on_date
)

# Define nocturnists (night shift only providers)
NOCTURNISTS = {"JT", "OI", "AT", "CM", "YD", "RS"}

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
    # Add some randomness for variety
    random.seed(datetime.now().timestamp())
    
    events = assign_advanced(year, month, providers, shift_types, shift_capacity, 
                           provider_rules, global_rules)
    
    return events

def assign_advanced(year: int, month: int, providers: List[str], 
                   shift_types: List[Dict], shift_capacity: Dict[str, int],
                   provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Advanced assignment algorithm following ground rules.
    """
    import streamlit as st
    
    # Initialize
    events = []
    month_days = make_month_days(year, month)
    
    # Separate providers by type
    app_providers = [p for p in providers if p in APP_PROVIDER_INITIALS]
    nocturnists = [p for p in providers if p in NOCTURNISTS]
    physician_providers = [p for p in providers if p not in APP_PROVIDER_INITIALS and p not in NOCTURNISTS]
    
    # Track provider assignments
    provider_shifts = {p: [] for p in providers}
    provider_blocks = {p: [] for p in providers}  # Track current shift blocks
    provider_last_shift = {p: None for p in providers}
    
    # Step 1: Assign APP shifts first (they have specific rules)
    app_events = assign_app_shifts(month_days, app_providers, shift_capacity, 
                                  provider_rules, global_rules)
    events.extend(app_events)
    
    # Update provider tracking
    for event in app_events:
        provider = event.extendedProps.get("provider")
        if provider:
            provider_shifts[provider].append(event)
            provider_last_shift[provider] = event.start.date()
    
    # Step 2: Assign night shifts to nocturnists first
    night_events = assign_night_shifts_to_nocturnists(month_days, nocturnists, 
                                                     shift_capacity, provider_rules, 
                                                     global_rules, provider_shifts)
    events.extend(night_events)
    
    # Step 3: Assign remaining physician shifts in blocks
    physician_events = assign_physician_shifts(month_days, physician_providers, 
                                             shift_capacity, provider_rules, 
                                             global_rules, provider_shifts)
    events.extend(physician_events)
    
    # Step 4: Fill remaining shifts with any available providers
    remaining_events = fill_remaining_shifts(month_days, providers, shift_capacity, 
                                           provider_rules, global_rules, provider_shifts)
    events.extend(remaining_events)
    
    return events

def assign_night_shifts_to_nocturnists(month_days: List[date], nocturnists: List[str], 
                                      shift_capacity: Dict[str, int], provider_rules: Dict, 
                                      global_rules: RuleConfig, provider_shifts: Dict) -> List[SEvent]:
    """
    Assign night shifts to nocturnists first to ensure proper coverage.
    """
    events = []
    night_shift_types = ["N12", "NB"]
    
    for day in month_days:
        for shift_type in night_shift_types:
            capacity = shift_capacity.get(shift_type, 0)
            if capacity <= 0:
                continue
            
            # Get available nocturnists for this day
            available_nocturnists = [p for p in nocturnists 
                                   if not is_provider_unavailable_on_date(p, day)]
            
            # Filter out nocturnists who already have shifts on this day
            available_nocturnists = [p for p in available_nocturnists 
                                   if not _has_shift_on_date(p, day, provider_shifts[p])]
            
            # Assign nocturnists up to capacity
            assigned_count = 0
            while assigned_count < capacity and available_nocturnists:
                # Select nocturnist (with some randomness)
                nocturnist = random.choice(available_nocturnists)
                available_nocturnists.remove(nocturnist)
                
                # Create night shift event
                shift_config = get_shift_config(shift_type)
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
                assigned_count += 1
    
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
                         if not is_provider_unavailable_on_date(p, day)]
        
        # Filter out APPs who already have shifts on this day
        available_apps = [p for p in available_apps 
                         if not _has_shift_on_date(p, day, events)]
        
        # Assign APPs up to capacity
        assigned_count = 0
        while assigned_count < app_capacity and available_apps:
            # Select APP (with some randomness)
            app_provider = random.choice(available_apps)
            available_apps.remove(app_provider)
            
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
                          global_rules: RuleConfig, provider_shifts: Dict) -> List[SEvent]:
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
                                         shift_capacity, provider_rules, global_rules)
    
    # Assign shifts based on blocks
    for provider, blocks in physician_blocks.items():
        for block in blocks:
            block_events = assign_shift_block(provider, block, provider_shifts)
            events.extend(block_events)
    
    return events

def fill_remaining_shifts(month_days: List[date], providers: List[str], 
                         shift_capacity: Dict[str, int], provider_rules: Dict, 
                         global_rules: RuleConfig, provider_shifts: Dict) -> List[SEvent]:
    """
    Fill any remaining shifts with available providers to optimize coverage.
    """
    events = []
    
    # Define shift types and their priorities for filling
    shift_priorities = [
        ("N12", 4),  # Night shifts - high priority
        ("NB", 1),   # Night bridge - high priority
        ("R12", 13), # Rounder shifts - medium priority
        ("A12", 1),  # Admitter shifts - medium priority
        ("A10", 2),  # Admitter shifts - medium priority
    ]
    
    for day in month_days:
        for shift_type, capacity in shift_priorities:
            # Check if this shift type needs more coverage
            current_coverage = count_shifts_on_date(day, shift_type, provider_shifts)
            if current_coverage >= capacity:
                continue
            
            # Find available providers for this shift type
            available_providers = get_available_providers_for_shift(day, shift_type, 
                                                                  providers, provider_shifts, 
                                                                  provider_rules)
            
            # Assign providers to fill remaining slots
            remaining_slots = capacity - current_coverage
            assigned_count = 0
            
            while assigned_count < remaining_slots and available_providers:
                provider = random.choice(available_providers)
                available_providers.remove(provider)
                
                # Create shift event
                shift_config = get_shift_config(shift_type)
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
                assigned_count += 1
    
    return events

def get_available_providers_for_shift(day: date, shift_type: str, providers: List[str], 
                                     provider_shifts: Dict, provider_rules: Dict) -> List[str]:
    """
    Get available providers for a specific shift type on a given day.
    """
    available_providers = []
    
    for provider in providers:
        # Skip if provider is unavailable
        if is_provider_unavailable_on_date(provider, day):
            continue
        
        # Skip if provider already has a shift on this day
        if _has_shift_on_date(provider, day, provider_shifts[provider]):
            continue
        
        # Check provider-specific restrictions
        provider_rule = provider_rules.get(provider, {})
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        # Skip if provider doesn't prefer this shift type
        if not shift_preferences.get(shift_type, True):
            continue
        
        # Check rest requirements
        if not _has_sufficient_rest(provider, day, provider_shifts[provider]):
            continue
        
        available_providers.append(provider)
    
    return available_providers

def count_shifts_on_date(day: date, shift_type: str, provider_shifts: Dict) -> int:
    """
    Count how many shifts of a specific type are already assigned on a given day.
    """
    count = 0
    for provider, shifts in provider_shifts.items():
        for shift in shifts:
            if hasattr(shift, 'start'):
                shift_date = shift.start.date()
                shift_type_actual = shift.extendedProps.get("shift_type")
            elif isinstance(shift, dict) and 'start' in shift:
                try:
                    shift_date = datetime.fromisoformat(shift['start']).date()
                    shift_type_actual = shift.get('extendedProps', {}).get("shift_type")
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            if shift_date == day and shift_type_actual == shift_type:
                count += 1
    
    return count

def create_shift_blocks(month_days: List[date], physician_providers: List[str], 
                       shift_capacity: Dict[str, int], provider_rules: Dict, 
                       global_rules: RuleConfig) -> Dict[str, List[Dict]]:
    """
    Create shift blocks for each physician following the rules:
    - Maximum 7 shifts per block
    - 3-day rest between blocks
    - Rounders should end on rounding shifts
    """
    physician_blocks = {p: [] for p in physician_providers}
    
    # Define shift types for physicians (exclude APP and night shifts for non-nocturnists)
    physician_shift_types = ["R12", "A12", "A10"]
    
    for provider in physician_providers:
        # Get provider preferences
        provider_rule = provider_rules.get(provider, {})
        min_shifts = provider_rule.get("min_shifts", 8)
        max_shifts = provider_rule.get("max_shifts", 16)
        
        # Determine target number of shifts for this provider
        target_shifts = random.randint(min_shifts, max_shifts)
        
        # Create blocks with maximum 7 shifts per block
        remaining_shifts = target_shifts
        while remaining_shifts > 0:
            # Determine block size (3-7 shifts, but no more than remaining shifts)
            max_block_size = min(7, remaining_shifts)
            # Ensure we have a valid range for random.randint
            if max_block_size < 3:
                # If remaining shifts is less than 3, use all remaining shifts
                block_size = remaining_shifts
            else:
                block_size = random.randint(3, max_block_size)
            
            # Determine shift type for this block
            shift_type = select_shift_type_for_block(provider, physician_shift_types, 
                                                    provider_rule)
            
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
    """
    # Get provider's shift preferences
    shift_preferences = provider_rule.get("shift_preferences", {})
    
    # Filter available shift types based on preferences
    available_types = []
    for shift_type in shift_types:
        if shift_preferences.get(shift_type, True):  # Default to True if not specified
            available_types.append(shift_type)
    
    if not available_types:
        available_types = shift_types  # Fallback to all types
    
    # Prefer rounder shifts (R12) as they're most common
    if "R12" in available_types:
        return "R12"
    
    # Otherwise, choose randomly from available types
    return random.choice(available_types)

def assign_shift_block(provider: str, block: Dict, provider_shifts: Dict) -> List[SEvent]:
    """
    Assign a specific shift block to a provider with proper sequence rules.
    """
    events = []
    shift_type = block["shift_type"]
    block_size = block["size"]
    
    # Get shift type configuration
    shift_config = get_shift_config(shift_type)
    
    # Find available dates for this block
    available_dates = find_available_dates_for_block(provider, shift_type, block_size, 
                                                   provider_shifts)
    
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
                                  provider_shifts: Dict) -> List[date]:
    """
    Find available dates for a shift block, ensuring 3-day rest between blocks.
    """
    # Get all dates in the current month
    today = date.today()
    month_start = today.replace(day=1)
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)
    month_end = next_month - timedelta(days=1)
    
    available_dates = []
    
    # Find all available dates for this provider and shift type
    current_date = month_start
    while current_date <= month_end:
        if not is_provider_unavailable_on_date(provider, current_date):
            if not _has_shift_on_date(provider, current_date, provider_shifts[provider]):
                # Check for 3-day rest requirement
                if _has_sufficient_rest(provider, current_date, provider_shifts[provider]):
                    available_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Try to find consecutive dates first
    consecutive_dates = find_consecutive_dates(available_dates, block_size)
    if consecutive_dates:
        return consecutive_dates
    
    # If no consecutive dates, return any available dates
    return available_dates[:block_size]

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

def _has_sufficient_rest(provider: str, target_date: date, provider_shifts: List[SEvent]) -> bool:
    """
    Check if provider has at least 3 days of rest before the target date.
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
    
    # Check if there are at least 3 days between last shift and target date
    days_since_last = (target_date - last_shift_date).days
    return days_since_last >= 3

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
                  global_rules: RuleConfig, provider_rules: Dict) -> Dict[str, Any]:
    """
    Validate scheduling rules and return violations.
    """
    violations = []
    provider_violations = {}
    
    # Count shifts per provider
    provider_shift_counts = {p: 0 for p in providers}
    provider_weekend_shifts = {p: 0 for p in providers}
    provider_night_shifts = {p: 0 for p in providers}
    
    for event in events:
        provider = event.extendedProps.get("provider")
        if provider:
            provider_shift_counts[provider] += 1
            
            # Count weekend shifts
            if hasattr(event, 'start'):
                event_date = event.start.date()
            elif isinstance(event, dict) and 'start' in event:
                try:
                    event_date = datetime.fromisoformat(event['start']).date()
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            if event_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                provider_weekend_shifts[provider] += 1
            
            # Count night shifts
            shift_type = event.extendedProps.get("shift_type")
            if shift_type in ["N12", "NB"]:
                provider_night_shifts[provider] += 1
    
    # Check violations
    for provider in providers:
        shift_count = provider_shift_counts[provider]
        provider_rule = provider_rules.get(provider, {})
        
        # Skip APP providers for min/max shift validation
        if provider in APP_PROVIDER_INITIALS:
            continue
        
        # Get min/max from provider-specific rules or global rules
        min_shifts = provider_rule.get("min_shifts", global_rules.min_shifts_per_month)
        max_shifts = provider_rule.get("max_shifts", global_rules.max_shifts_per_month)
        
        provider_violations[provider] = []
        
        if shift_count < min_shifts:
            violation = f"{provider}: {shift_count} shifts (min {min_shifts} required)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        
        if shift_count > max_shifts:
            violation = f"{provider}: {shift_count} shifts (max {max_shifts} allowed)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        
        # Check weekend coverage
        if provider_weekend_shifts[provider] < global_rules.min_weekend_shifts_per_month:
            violation = f"{provider}: {provider_weekend_shifts[provider]} weekend shifts (min {global_rules.min_weekend_shifts_per_month} required)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        
        if provider_weekend_shifts[provider] > global_rules.max_weekend_shifts_per_month:
            violation = f"{provider}: {provider_weekend_shifts[provider]} weekend shifts (max {global_rules.max_weekend_shifts_per_month} allowed)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        
        # Check night shift limits
        if provider_night_shifts[provider] < global_rules.min_night_shifts_per_month:
            violation = f"{provider}: {provider_night_shifts[provider]} night shifts (min {global_rules.min_night_shifts_per_month} required)"
            violations.append(violation)
            provider_violations[provider].append(violation)
        
        if provider_night_shifts[provider] > global_rules.max_night_shifts_per_month:
            violation = f"{provider}: {provider_night_shifts[provider]} night shifts (max {global_rules.max_night_shifts_per_month} allowed)"
            violations.append(violation)
            provider_violations[provider].append(violation)
    
    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
        "provider_violations": provider_violations
    }
