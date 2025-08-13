# =============================================================================
# Block Assignment and Provider Selection
# =============================================================================

from typing import List, Dict, Optional
from datetime import datetime, timedelta, date
import uuid
import logging

from models.data_models import SEvent, RuleConfig
from core.utils import (
    parse_time, get_adjusted_expected_shifts, is_provider_unavailable_on_date
)
from core.provider_types import get_allowed_shift_types

logger = logging.getLogger(__name__)

def find_best_provider_for_gap_group(gap_group: List[Dict], regular_providers: List[str], 
                                    provider_shifts: Dict, provider_rules: Dict, 
                                    global_rules: RuleConfig, year: int, month: int) -> Optional[str]:
    """
    Find the best provider for a gap group based on:
    1. Current shift count (fewer = better)
    2. Availability for all dates in the gap
    3. Shift type preferences
    4. Rest requirements
    """
    best_provider = None
    best_score = float('inf')
    
    for provider in regular_providers:
        # Check if provider is available for all dates in the gap
        available_for_all = True
        for gap in gap_group:
            day = gap["day"]
            shift_type = gap["shift_type"]
            
            # Check availability
            if is_provider_unavailable_on_date(provider, day, provider_rules):
                available_for_all = False
                break
            
            # Check if already has shift on this day
            if _has_shift_on_date(provider, day, provider_shifts[provider]):
                available_for_all = False
                break
            
            # Check shift type preference
            if not _validate_shift_type_preference(provider, shift_type, provider_rules):
                available_for_all = False
                break
        
        if not available_for_all:
            continue
        
        # Check rest requirements
        min_rest_days = 2
        if global_rules and hasattr(global_rules, 'min_days_between_shifts'):
            min_rest_days = max(2, global_rules.min_days_between_shifts)
        
        rest_ok = True
        for gap in gap_group:
            day = gap["day"]
            if not _has_sufficient_rest(provider, day, provider_shifts[provider], min_rest_days):
                rest_ok = False
                break
        
        if not rest_ok:
            continue
        
        # Check if assignment would exceed expected shifts
        current_shifts = len(provider_shifts[provider])
        expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
        
        if current_shifts + len(gap_group) > expected_shifts:
            continue
        
        # Calculate score (lower is better)
        score = current_shifts / expected_shifts  # Prefer providers with fewer shifts
        
        if score < best_score:
            best_score = score
            best_provider = provider
    
    return best_provider

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

# Helper functions (these will be moved from scheduler.py)
def _has_shift_on_date(provider: str, target_date: date, provider_shifts: List[SEvent]) -> bool:
    """Check if provider has a shift on the target date."""
    for event in provider_shifts:
        if event.start.date() == target_date:
            return True
    return False

def _has_sufficient_rest(provider: str, target_date: date, provider_shifts: List[SEvent], 
                        min_rest_days: int = 2) -> bool:
    """Check if provider has sufficient rest before the target date."""
    # Find the last shift date for this provider
    last_shift_date = None
    for event in provider_shifts:
        if event.start.date() < target_date:
            if last_shift_date is None or event.start.date() > last_shift_date:
                last_shift_date = event.start.date()
    
    if last_shift_date is None:
        return True  # No previous shifts
    
    days_since_last = (target_date - last_shift_date).days
    return days_since_last >= min_rest_days

def _validate_shift_type_preference(provider: str, shift_type: str, provider_rules: Dict) -> bool:
    """Validate if provider prefers this shift type."""
    if provider not in provider_rules:
        return True  # Default to allowed if no rules
    
    shift_preferences = provider_rules[provider].get("shift_preferences", {})
    return shift_preferences.get(shift_type, False)

def get_shift_config(shift_type: str) -> Dict:
    """Get shift configuration for a given shift type."""
    shift_configs = {
        "R12": {"label": "7am–7pm Rounder", "start": "07:00", "end": "19:00", "color": "#16a34a"},
        "A12": {"label": "7am–7pm Admitter", "start": "07:00", "end": "19:00", "color": "#f59e0b"},
        "A10": {"label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
        "N12": {"label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
        "NB": {"label": "Night Bridge", "start": "23:00", "end": "07:00", "color": "#06b6d4"},
        "APP": {"label": "APP Provider", "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
    }
    
    return shift_configs.get(shift_type, shift_configs["R12"])
