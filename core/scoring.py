# =============================================================================
# Scoring System for IMIS Scheduler
# =============================================================================

import logging
from datetime import date, timedelta
from typing import List, Dict, Set, Optional
from models.data_models import SEvent, RuleConfig
from core.provider_types import NOCTURNISTS, SENIORS
from models.constants import APP_PROVIDER_INITIALS

logger = logging.getLogger(__name__)

class ScheduleScorer:
    """
    Comprehensive scoring system for provider shift assignments.
    
    This scoring system optimizes for:
    1. Block-based scheduling (3-7 consecutive shifts)
    2. Proper rest periods between blocks (2+ days)
    3. No day-to-night transitions without rest
    4. Shift type consistency within blocks
    5. Provider preferences (shift types, timing, day/night ratio)
    6. Fair distribution of weekend and night shifts
    7. Work-life balance through front/back-loaded schedules
    """
    
    def __init__(self, events: List[SEvent], providers: List[str], 
                 provider_rules: Dict, global_rules: RuleConfig,
                 year: int, month: int):
        print(f"      🧮 Initializing scorer with {len(events)} events, {len(providers)} providers")
        self.events = events
        self.providers = providers
        self.provider_rules = provider_rules
        self.global_rules = global_rules
        self.year = year
        self.month = month
        
        # Pre-calculate provider statistics for efficiency
        print(f"      📊 Calculating provider statistics...")
        self._provider_stats = self._calculate_provider_stats()
        print(f"      ✅ Scorer initialized")
    
    def _calculate_provider_stats(self) -> Dict[str, Dict]:
        """Pre-calculate provider statistics for efficient scoring."""
        stats = {}
        
        for provider in self.providers:
            provider_upper = provider.upper()
            stats[provider_upper] = {
                'total_shifts': 0,
                'night_shifts': 0,
                'weekend_shifts': 0,
                'shift_dates': set(),
                'shift_types': [],
                'last_shift_type': None,
                'consecutive_days': 0,
                'blocks': []  # List of (start_date, end_date, shift_type) tuples
            }
            
            # Analyze existing events
            provider_events = [e for e in self.events 
                             if (e.extendedProps.get("provider") or "").upper() == provider_upper]
            
            shift_dates = sorted([e.start.date() for e in provider_events])
            
            for event in provider_events:
                shift_date = event.start.date()
                shift_type = event.extendedProps.get("shift_type") or event.extendedProps.get("shift_key")
                
                stats[provider_upper]['total_shifts'] += 1
                stats[provider_upper]['shift_dates'].add(shift_date)
                stats[provider_upper]['shift_types'].append(shift_type)
                
                # Count night shifts
                if shift_type in ["N12", "NB"]:
                    stats[provider_upper]['night_shifts'] += 1
                
                # Count weekend shifts
                if shift_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    stats[provider_upper]['weekend_shifts'] += 1
            
            # Calculate blocks
            stats[provider_upper]['blocks'] = self._calculate_blocks(provider_events)
        
        return stats
    
    def _calculate_blocks(self, provider_events: List[SEvent]) -> List[tuple]:
        """Calculate existing blocks for a provider."""
        if not provider_events:
            return []
        
        # Sort events by date
        events_sorted = sorted(provider_events, key=lambda e: e.start.date())
        blocks = []
        current_block_start = events_sorted[0].start.date()
        current_block_end = current_block_start
        current_shift_type = events_sorted[0].extendedProps.get("shift_type") or events_sorted[0].extendedProps.get("shift_key")
        
        for i in range(1, len(events_sorted)):
            event_date = events_sorted[i].start.date()
            shift_type = events_sorted[i].extendedProps.get("shift_type") or events_sorted[i].extendedProps.get("shift_key")
            
            # Check if this continues the current block
            if (event_date - current_block_end).days == 1 and shift_type == current_shift_type:
                current_block_end = event_date
            else:
                # End current block and start new one
                blocks.append((current_block_start, current_block_end, current_shift_type))
                current_block_start = event_date
                current_block_end = event_date
                current_shift_type = shift_type
        
        # Add the last block
        blocks.append((current_block_start, current_block_end, current_shift_type))
        
        return blocks
    
    def score_assignment(self, provider: str, day: date, shift_type: str) -> float:
        """
        Calculate the score for assigning a specific shift to a provider on a given day.
        
        Higher scores indicate better assignments.
        """
        provider_upper = provider.upper()
        score = 0.0
        
        # Get provider statistics
        stats = self._provider_stats.get(provider_upper, {})
        provider_rule = self.provider_rules.get(provider, {})
        
        # 1. Provider Type Specific Scoring
        if provider in APP_PROVIDER_INITIALS:
            score += self._score_app_provider(provider, day, shift_type, stats, provider_rule)
        elif provider in NOCTURNISTS:
            score += self._score_nocturnist(provider, day, shift_type, stats, provider_rule)
        elif provider in SENIORS:
            score += self._score_senior(provider, day, shift_type, stats, provider_rule)
        else:
            score += self._score_regular_provider(provider, day, shift_type, stats, provider_rule)
        
        # 2. Universal Scoring Components (apply to all providers)
        score += self._score_block_consistency(provider, day, shift_type, stats)
        score += self._score_rest_requirements(provider, day, stats)
        score += self._score_shift_preferences(provider, shift_type, provider_rule)
        score += self._score_timing_preferences(provider, day, provider_rule)
        score += self._score_workload_balance(provider, stats, provider_rule)
        score += self._score_weekend_distribution(provider, day, stats, provider_rule)
        
        return score
    
    def _score_app_provider(self, provider: str, day: date, shift_type: str, 
                           stats: Dict, provider_rule: Dict) -> float:
        """Scoring specific to APP providers."""
        score = 0.0
        
        # APP providers should only take APP shifts
        if shift_type != "APP":
            return -1000.0  # Strong penalty for wrong shift type
        
        # Weekend coverage priority
        if self._is_weekend(day):
            score += 8.0  # High priority for weekend coverage
        
        # Block consistency for APP providers
        existing_dates = stats.get('shift_dates', set())
        left_run = self._left_run_length(existing_dates, day)
        right_run = self._right_run_length(existing_dates, day)
        
        if left_run > 0:
            score += 3.0  # Bonus for continuing a block
        
        if left_run < self.global_rules.min_block_size:
            score += 2.0  # Bonus for building up to minimum block size
        
        # Avoid standalone days
        if left_run == 0 and right_run == 0:
            score -= 2.0  # Penalty for standalone days
        
        # Prefer optimal block size (4-7 days)
        total_block_len = left_run + 1 + right_run
        if 4 <= total_block_len <= 7:
            score += 3.0
        elif total_block_len > 7:
            score -= 1.0
        
        return score
    
    def _score_nocturnist(self, provider: str, day: date, shift_type: str, 
                         stats: Dict, provider_rule: Dict) -> float:
        """Scoring specific to nocturnists."""
        score = 0.0
        
        # Nocturnists should only take night shifts
        if shift_type not in ["N12", "NB"]:
            return -1000.0  # Strong penalty for non-night shifts
        
        # Encourage night shift coverage
        score += 5.0  # Base bonus for taking night shifts
        
        # Block-based scheduling for nocturnists (prefer 3-7 consecutive nights)
        existing_dates = stats.get('shift_dates', set())
        left_run = self._left_run_length(existing_dates, day)
        right_run = self._right_run_length(existing_dates, day)
        total_block_len = left_run + 1 + right_run
        
        if 3 <= total_block_len <= 7:
            score += 4.0  # Optimal block size
        elif total_block_len > 7:
            score -= 2.0  # Penalty for very long blocks
        elif total_block_len < 3 and left_run > 0:
            score += 2.0  # Building up to minimum block
        
        return score
    
    def _score_senior(self, provider: str, day: date, shift_type: str, 
                     stats: Dict, provider_rule: Dict) -> float:
        """Scoring specific to senior providers."""
        score = 0.0
        
        # Seniors should only take R12 shifts
        if shift_type != "R12":
            return -1000.0  # Strong penalty for non-rounding shifts
        
        # Encourage rounding shift coverage
        score += 3.0  # Base bonus for taking rounding shifts
        
        # Block consistency (seniors also work in blocks)
        existing_dates = stats.get('shift_dates', set())
        left_run = self._left_run_length(existing_dates, day)
        total_block_len = left_run + 1 + self._right_run_length(existing_dates, day)
        
        if 3 <= total_block_len <= 5:
            score += 2.0  # Optimal block size for seniors
        
        return score
    
    def _score_regular_provider(self, provider: str, day: date, shift_type: str, 
                               stats: Dict, provider_rule: Dict) -> float:
        """Scoring specific to regular providers."""
        score = 0.0
        
        # Regular providers can't take APP shifts
        if shift_type == "APP":
            return -1000.0
        
        # Get expected shifts for workload balancing
        expected_shifts = self._get_expected_shifts(provider, provider_rule)
        current_shifts = stats.get('total_shifts', 0)
        
        # Encourage providers who are below their target
        if current_shifts < expected_shifts:
            score += 4.0
            
            # Additional bonus for block building when below target
            existing_dates = stats.get('shift_dates', set())
            left_run = self._left_run_length(existing_dates, day)
            if left_run > 0:
                score += 2.0  # Continuing a block
            if left_run < self.global_rules.min_block_size:
                score += 4.0  # Building up to minimum block
        
        # Day/night ratio preference
        if shift_type == "N12":
            night_ratio = self._calculate_night_ratio(provider, stats, provider_rule)
            desired_night_ratio = (100.0 - provider_rule.get("day_percentage", 80.0)) / 100.0
            
            if night_ratio > desired_night_ratio + 0.05:
                score -= 2.0  # Too many nights
            elif night_ratio < desired_night_ratio - 0.05:
                score += 1.0  # Good to add more nights
        
        return score
    
    def _score_block_consistency(self, provider: str, day: date, shift_type: str, 
                                stats: Dict) -> float:
        """Score based on block consistency and shift type mixing."""
        score = 0.0
        
        existing_dates = stats.get('shift_dates', set())
        if not existing_dates:
            return score
        
        # Check if this assignment would create or extend a block
        left_run = self._left_run_length(existing_dates, day)
        right_run = self._right_run_length(existing_dates, day)
        
        if left_run == 0 and right_run == 0:
            # Standalone day - strong penalty unless it's strategic
            score -= 6.0
        elif left_run + right_run + 1 <= 2:
            # Very short block - moderate penalty
            score -= 3.0
        elif 4 <= left_run + right_run + 1 <= 7:
            # Optimal block length - bonus
            score += 2.0
        
        # Check shift type consistency within blocks
        if left_run > 0 or right_run > 0:
            block_start = day - timedelta(days=left_run)
            block_end = day + timedelta(days=right_run)
            
            # Get shift types in the existing block
            block_shift_types = set()
            for event in self.events:
                if (event.extendedProps.get("provider") or "").upper() == provider.upper():
                    event_date = event.start.date()
                    if block_start <= event_date <= block_end:
                        existing_shift_type = event.extendedProps.get("shift_type") or event.extendedProps.get("shift_key")
                        block_shift_types.add(existing_shift_type)
            
            # Classify shift types
            night_shifts = {"N12", "NB"}
            day_shifts = {"R12", "A12", "A10"}
            
            current_is_night = shift_type in night_shifts
            block_has_nights = any(s in night_shifts for s in block_shift_types)
            block_has_days = any(s in day_shifts for s in block_shift_types)
            
            # Strong preference for shift consistency within blocks
            if block_has_nights and block_has_days:
                # Mixed block - penalty for adding different type
                score -= 5.0
            elif block_has_nights and not current_is_night:
                # Adding day shift to night block - very strong penalty
                score -= 8.0
            elif block_has_days and current_is_night:
                # Adding night shift to day block - very strong penalty
                score -= 8.0
            else:
                # Consistent block - strong bonus
                score += 3.0
                
                # Additional bonus for extending consistent blocks
                if left_run > 0 or right_run > 0:
                    score += 2.0
        
        return score
    
    def _score_rest_requirements(self, provider: str, day: date, stats: Dict) -> float:
        """Score based on rest requirements between shifts and blocks."""
        score = 0.0
        
        existing_dates = stats.get('shift_dates', set())
        if not existing_dates:
            return score
        
        # Check minimum rest between shifts
        min_rest_days = self.provider_rules.get(provider, {}).get("min_rest_days", 
                                                                 self.global_rules.min_days_between_shifts)
        
        # Check for adequate rest before this shift
        for existing_date in existing_dates:
            days_diff = abs((day - existing_date).days)
            if days_diff == 1:
                # Adjacent day - check if it's part of a block or violates rest
                left_run = self._left_run_length(existing_dates, day)
                if left_run == 0:  # Not continuing a block
                    score -= 10.0  # Strong penalty for inadequate rest
            elif 1 < days_diff <= min_rest_days:
                score -= 5.0  # Penalty for insufficient rest
        
        # Bonus for good rest patterns
        closest_shift = min(existing_dates, key=lambda d: abs((day - d).days))
        days_to_closest = abs((day - closest_shift).days)
        
        if days_to_closest >= min_rest_days + 1:
            score += 1.0  # Bonus for good rest
        
        return score
    
    def _score_shift_preferences(self, provider: str, shift_type: str, 
                                provider_rule: Dict) -> float:
        """Score based on provider's shift type preferences."""
        shift_preferences = provider_rule.get("shift_preferences", {})
        
        if shift_type in shift_preferences:
            if shift_preferences[shift_type]:
                return 2.0  # Bonus for preferred shift type
            else:
                return -5.0  # Penalty for non-preferred shift type
        
        return 0.0  # Neutral if no preference specified
    
    def _score_timing_preferences(self, provider: str, day: date, provider_rule: Dict) -> float:
        """Score based on front-loaded vs back-loaded preferences."""
        timing_pref = provider_rule.get("shift_timing_preference")
        if not timing_pref:
            return 0.0
        
        score = 0.0
        day_of_month = day.day
        month_length = self._get_month_length(day.year, day.month)
        
        if timing_pref == "front_loaded":
            if day_of_month <= month_length // 2:
                score += 1.5  # Bonus for first half
            else:
                score -= 0.5  # Small penalty for second half
        elif timing_pref == "back_loaded":
            if day_of_month > month_length // 2:
                score += 1.5  # Bonus for second half
            else:
                score -= 0.5  # Small penalty for first half
        
        return score
    
    def _score_workload_balance(self, provider: str, stats: Dict, provider_rule: Dict) -> float:
        """Score based on workload balancing across providers."""
        score = 0.0
        
        current_shifts = stats.get('total_shifts', 0)
        expected_shifts = self._get_expected_shifts(provider, provider_rule)
        
        # Gentle load balancing
        score += max(0, expected_shifts - current_shifts) * 0.5
        
        # Penalty for exceeding expected shifts
        if current_shifts >= expected_shifts:
            score -= (current_shifts - expected_shifts) * 2.0
        
        return score
    
    def _score_weekend_distribution(self, provider: str, day: date, stats: Dict, 
                                   provider_rule: Dict) -> float:
        """Score based on fair weekend distribution."""
        if not self._is_weekend(day):
            return 0.0
        
        score = 0.0
        weekend_shifts = stats.get('weekend_shifts', 0)
        
        # Encourage weekend coverage if provider has few weekend shifts
        if weekend_shifts == 0:
            score += 3.0  # Strong bonus for first weekend
        elif weekend_shifts < 2:
            score += 1.0  # Moderate bonus for second weekend
        elif weekend_shifts >= 4:
            score -= 2.0  # Penalty for too many weekends
        
        return score
    
    def _left_run_length(self, dates: Set[date], target_date: date) -> int:
        """Calculate length of consecutive days to the left of target date."""
        run = 0
        current = target_date - timedelta(days=1)
        while current in dates:
            run += 1
            current -= timedelta(days=1)
        return run
    
    def _right_run_length(self, dates: Set[date], target_date: date) -> int:
        """Calculate length of consecutive days to the right of target date."""
        run = 0
        current = target_date + timedelta(days=1)
        while current in dates:
            run += 1
            current += timedelta(days=1)
        return run
    
    def _is_weekend(self, day: date) -> bool:
        """Check if a day is weekend (Saturday or Sunday)."""
        return day.weekday() >= 5
    
    def _get_expected_shifts(self, provider: str, provider_rule: Dict) -> int:
        """Calculate expected number of shifts for a provider."""
        base_shifts = self.global_rules.expected_shifts_per_month
        fte = provider_rule.get("fte_percentage", 1.0)
        return int(base_shifts * fte)
    
    def _calculate_night_ratio(self, provider: str, stats: Dict, provider_rule: Dict) -> float:
        """Calculate current night shift ratio for a provider."""
        total_shifts = stats.get('total_shifts', 0)
        night_shifts = stats.get('night_shifts', 0)
        
        if total_shifts == 0:
            return 0.0
        
        return night_shifts / total_shifts
    
    def _get_month_length(self, year: int, month: int) -> int:
        """Get the number of days in a month."""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        current_month = date(year, month, 1)
        return (next_month - current_month).days

def create_scorer(events: List[SEvent], providers: List[str], 
                 provider_rules: Dict, global_rules: RuleConfig,
                 year: int, month: int) -> ScheduleScorer:
    """Factory function to create a ScheduleScorer instance."""
    return ScheduleScorer(events, providers, provider_rules, global_rules, year, month)
