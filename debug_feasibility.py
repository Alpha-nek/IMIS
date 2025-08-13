#!/usr/bin/env python3
"""
Debug feasibility checking to find out why all providers are unfeasible.
"""

import sys
sys.path.append('.')

from datetime import date
from core.scheduler import is_provider_feasible
from models.data_models import RuleConfig
from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY
from core.provider_rules import ensure_default_provider_rules

def debug_feasibility():
    """Debug why providers are unfeasible."""
    
    # Test parameters with clinical provider types
    year, month = 2024, 1
    providers = [
        'JA',  # APP provider (from APP_PROVIDER_INITIALS)
        'AA',  # Senior provider (from SENIORS)
        'RJ',  # Bridge-qualified regular provider (from BRIDGE_QUALIFIED)
        'AB',  # Regular provider (not in any special category)
        'JT'   # Nocturnist (from NOCTURNISTS)
    ]
    provider_rules = {}
    global_rules = RuleConfig()
    events = []  # Empty events list
    test_day = date(2024, 1, 15)
    
    # Ensure provider rules are set up
    provider_rules = ensure_default_provider_rules(providers, provider_rules)
    
    print("Debugging provider feasibility...")
    print(f"Test day: {test_day}")
    print(f"Global rules: {global_rules}")
    print(f"Providers: {providers}")
    
    # Test each shift type
    for shift_type in ["R12", "A12", "A10", "N12", "NB", "APP"]:
        print(f"\n--- Testing shift type: {shift_type} ---")
        
        for provider in providers:
            print(f"\nProvider: {provider}")
            
            # Check each feasibility condition step by step
            try:
                from core.utils import is_provider_unavailable_on_date
                unavailable = is_provider_unavailable_on_date(provider, test_day, provider_rules)
                print(f"  1. Unavailable on date: {unavailable}")
                
                from core.shift_validation import has_shift_on_date
                has_shift = has_shift_on_date(provider, test_day, events)
                print(f"  2. Already has shift: {has_shift}")
                
                from models.constants import APP_PROVIDER_INITIALS
                from core.provider_types import NOCTURNISTS, SENIORS
                
                # Check provider type eligibility
                if provider in APP_PROVIDER_INITIALS:
                    type_check = shift_type == "APP"
                    print(f"  3. APP provider type check: {type_check} (is APP: {shift_type == 'APP'})")
                elif provider in NOCTURNISTS:
                    type_check = shift_type in ["N12", "NB"]
                    print(f"  3. Nocturnist type check: {type_check} (is night: {shift_type in ['N12', 'NB']})")
                elif provider in SENIORS:
                    type_check = shift_type == "R12"
                    print(f"  3. Senior type check: {type_check} (is R12: {shift_type == 'R12'})")
                else:
                    type_check = shift_type != "APP"
                    print(f"  3. Regular provider type check: {type_check} (not APP: {shift_type != 'APP'})")
                
                # Check shift preferences
                from core.shift_validation import validate_shift_type_preference
                pref_check = validate_shift_type_preference(provider, shift_type, provider_rules)
                provider_rule = provider_rules.get(provider, {})
                shift_preferences = provider_rule.get("shift_preferences", {})
                print(f"  4. Shift preference check: {pref_check}")
                print(f"     Preferences: {shift_preferences}")
                
                # Check expected shifts
                from core.utils import get_adjusted_expected_shifts
                current_shifts = len([e for e in events if e.extendedProps.get("provider") == provider])
                expected_shifts = get_adjusted_expected_shifts(provider, year, month, provider_rules, global_rules)
                exceeds_expected = current_shifts >= expected_shifts
                print(f"  5. Exceeds expected shifts: {exceeds_expected} ({current_shifts} >= {expected_shifts})")
                
                # Overall feasibility
                feasible = is_provider_feasible(provider, test_day, shift_type, events, 
                                              provider_rules, global_rules, year, month)
                print(f"  ➤ OVERALL FEASIBLE: {feasible}")
                
            except Exception as e:
                print(f"  ❌ Error checking {provider} for {shift_type}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    debug_feasibility()
