# inital file
# =============================================================================
# Core Scheduling Logic for IMIS Scheduler
# =============================================================================

import random
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd

from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS
from models.data_models import RuleConfig, Provider, SEvent
from core.utils import (
    is_holiday, get_holiday_adjusted_capacity, parse_time, 
    date_range, month_start_end, make_month_days,
    _expand_vacation_dates, is_provider_unavailable_on_date
)

def assign_greedy(year: int, month: int, providers: List[str], 
                 shift_types: List[Dict], shift_capacity: Dict[str, int],
                 provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Greedy algorithm to assign shifts to providers.
    """
    import streamlit as st
    
    # Initialize
    events = []
    month_days = make_month_days(year, month)
    
    # Create provider availability tracking
    provider_shifts = {p: [] for p in providers}
    provider_last_shift = {p: None for p in providers}
    
    # Separate APP providers
    app_providers = [p for p in providers if p in APP_PROVIDER_INITIALS]
    regular_providers = [p for p in providers if p not in APP_PROVIDER_INITIALS]
    
    # Process each day
    for day in month_days:
        # Check if provider is unavailable
        available_providers = [p for p in providers if not is_provider_unavailable_on_date(p, day)]
        
        # Separate available providers by type
        available_app_providers = [p for p in available_providers if p in APP_PROVIDER_INITIALS]
        available_regular_providers = [p for p in available_providers if p not in APP_PROVIDER_INITIALS]
        
        # Assign shifts for this day
        for shift_type in shift_types:
            shift_key = shift_type["key"]
            capacity = get_holiday_adjusted_capacity(shift_capacity.get(shift_key, 1), day)
            
            # Determine which providers can take this shift
            if shift_key == "APP":
                eligible_providers = available_app_providers
            else:
                eligible_providers = available_regular_providers
            
            # Filter providers based on rest days and other rules
            eligible_providers = [
                p for p in eligible_providers
                if _can_provider_take_shift(p, day, shift_key, provider_shifts[p], 
                                          provider_last_shift[p], global_rules)
            ]
            
            # Assign shifts up to capacity
            assigned_count = 0
            while assigned_count < capacity and eligible_providers:
                # Select provider (with some randomness for variety)
                provider = random.choice(eligible_providers)
                eligible_providers.remove(provider)
                
                # Create shift event
                start_time = datetime.combine(day, parse_time(shift_type["start"]))
                end_time = datetime.combine(day, parse_time(shift_type["end"]))
                
                # Handle overnight shifts
                if shift_type["end"] < shift_type["start"]:
                    end_time += timedelta(days=1)
                
                event = SEvent(
                    id=str(uuid.uuid4()),
                    title=f"{shift_type['label']} - {provider}",
                    start=start_time,
                    end=end_time,
                    backgroundColor=shift_type["color"],
                    extendedProps={
                        "provider": provider,
                        "shift_type": shift_key,
                        "shift_label": shift_type["label"]
                    }
                )
                
                events.append(event)
                provider_shifts[provider].append(event)
                provider_last_shift[provider] = day
                assigned_count += 1
    
    return events

def _can_provider_take_shift(provider: str, day: date, shift_type: str, 
                           provider_events: List[SEvent], last_shift_date: Optional[date],
                           global_rules: RuleConfig) -> bool:
    """
    Check if a provider can take a specific shift.
    """
    # Check rest days between shifts
    if last_shift_date:
        days_since_last = (day - last_shift_date).days
        if days_since_last < global_rules.min_rest_days_between_shifts:
            return False
    
    # Check if provider already has a shift on this day
    for event in provider_events:
        if event.start.date() == day:
            return False
    
    return True

def validate_rules(events: List[SEvent], providers: List[str], 
                  global_rules: RuleConfig, provider_rules: Dict) -> Dict[str, Any]:
    """
    Validate scheduling rules and return violations.
    """
    violations = {
        "min_shifts": [],
        "max_shifts": [],
        "rest_days": [],
        "weekend_coverage": [],
        "night_shifts": []
    }
    
    # Count shifts per provider
    provider_shift_counts = {p: 0 for p in providers}
    provider_weekend_shifts = {p: 0 for p in providers}
    provider_night_shifts = {p: 0 for p in providers}
    
    for event in events:
        provider = event.extendedProps.get("provider")
        if provider:
            provider_shift_counts[provider] += 1
            
            # Count weekend shifts
            if event.start.weekday() >= 5:  # Saturday = 5, Sunday = 6
                provider_weekend_shifts[provider] += 1
            
            # Count night shifts
            shift_type = event.extendedProps.get("shift_type")
            if shift_type in ["N12", "NB"]:
                provider_night_shifts[provider] += 1
    
    # Check violations
    for provider in providers:
        shift_count = provider_shift_counts[provider]
        provider_rule = provider_rules.get(provider, {})
        
        # Get min/max from provider-specific rules or global rules
        min_shifts = provider_rule.get("min_shifts", global_rules.min_shifts_per_provider)
        max_shifts = provider_rule.get("max_shifts", global_rules.max_shifts_per_provider)
        
        if shift_count < min_shifts:
            violations["min_shifts"].append({
                "provider": provider,
                "current": shift_count,
                "required": min_shifts
            })
        
        if shift_count > max_shifts:
            violations["max_shifts"].append({
                "provider": provider,
                "current": shift_count,
                "allowed": max_shifts
            })
        
        # Check weekend coverage
        if global_rules.require_at_least_one_weekend and provider_weekend_shifts[provider] == 0:
            violations["weekend_coverage"].append(provider)
        
        # Check night shift limits
        if global_rules.max_nights_per_provider and provider_night_shifts[provider] > global_rules.max_nights_per_provider:
            violations["night_shifts"].append({
                "provider": provider,
                "current": provider_night_shifts[provider],
                "allowed": global_rules.max_nights_per_provider
            })
    
    return violations

def generate_schedule(year: int, month: int, providers: List[str], 
                     shift_types: List[Dict], shift_capacity: Dict[str, int],
                     provider_rules: Dict, global_rules: RuleConfig) -> List[SEvent]:
    """
    Generate a complete schedule for the specified month.
    """
    # Add some randomness for variety
    random.seed(datetime.now().timestamp())
    
    events = assign_greedy(year, month, providers, shift_types, shift_capacity, 
                          provider_rules, global_rules)
    
    return events
