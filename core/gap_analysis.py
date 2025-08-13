# =============================================================================
# Gap Analysis and Block Optimization
# =============================================================================

from typing import List, Dict, Optional
from datetime import date
from core.utils import count_shifts_on_date

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
