#!/usr/bin/env python3
"""
Test script to validate shift count violations and optimize the scheduling algorithm.
This script will help identify why providers are exceeding the 15/16 shift limit.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, date
from typing import List, Dict, Any
import pandas as pd

from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY, APP_PROVIDER_INITIALS
from models.data_models import RuleConfig, SEvent
from core.scheduler import generate_schedule, validate_rules
from core.data_manager import load_providers, load_rules
from core.utils import make_month_days

def test_shift_count_validation(year: int = 2024, month: int = 12):
    """
    Test the shift count validation to identify why providers exceed limits.
    """
    print(f"=== Testing Shift Count Validation for {month}/{year} ===")
    
    # Load data
    providers_df, _ = load_providers()
    if providers_df.empty:
        # Use default providers from CSV
        import pandas as pd
        providers_df = pd.read_csv('IMIS_initials.csv')
    providers = providers_df['initials'].tolist()
    
    # Load rules
    global_rules_dict, shift_types, shift_capacity, provider_rules = load_rules()
    
    # Convert global_rules_dict to RuleConfig object
    from models.data_models import RuleConfig
    global_rules = RuleConfig(**global_rules_dict) if global_rules_dict else RuleConfig()
    
    # Get month days
    month_days = make_month_days(year, month)
    days_in_month = len(month_days)
    expected_min_shifts = 15 if days_in_month == 30 else 16
    
    print(f"Month has {days_in_month} days, expected minimum shifts: {expected_min_shifts}")
    print(f"Total providers: {len(providers)}")
    print(f"APP providers: {len([p for p in providers if p in APP_PROVIDER_INITIALS])}")
    print(f"Physician providers: {len([p for p in providers if p not in APP_PROVIDER_INITIALS])}")
    
    # Generate schedule
    print("\n=== Generating Schedule ===")
    events = generate_schedule(year, month, providers, shift_types, DEFAULT_SHIFT_CAPACITY, 
                             provider_rules, global_rules)
    
    print(f"Generated {len(events)} events")
    
    # Validate rules
    print("\n=== Validating Rules ===")
    validation_result = validate_rules(events, providers, global_rules, provider_rules)
    
    # Analyze results
    print(f"Schedule valid: {validation_result['is_valid']}")
    print(f"Total violations: {validation_result['summary']['total_violations']}")
    
    # Detailed analysis
    print("\n=== Detailed Provider Analysis ===")
    provider_stats = validation_result['summary']['provider_stats']
    
    violations_by_type = {
        'below_min': [],
        'above_max': [],
        'weekend_violations': [],
        'night_violations': []
    }
    
    for provider, stats in provider_stats.items():
        total_shifts = stats['total_shifts']
        provider_rule = provider_rules.get(provider, {})
        
        # Skip APP providers for min/max validation
        if provider in APP_PROVIDER_INITIALS:
            print(f"{provider} (APP): {total_shifts} shifts - APP providers have different rules")
            continue
        
        # Get min/max from provider-specific rules or global rules
        min_shifts = provider_rule.get("min_shifts", global_rules.min_shifts_per_month)
        max_shifts = provider_rule.get("max_shifts", global_rules.max_shifts_per_month)
        
        print(f"{provider}: {total_shifts} shifts (min: {min_shifts}, max: {max_shifts})")
        
        if total_shifts < min_shifts:
            violations_by_type['below_min'].append((provider, total_shifts, min_shifts))
        elif total_shifts > max_shifts:
            violations_by_type['above_max'].append((provider, total_shifts, max_shifts))
    
    # Summary of violations
    print("\n=== Violation Summary ===")
    print(f"Providers below minimum: {len(violations_by_type['below_min'])}")
    for provider, actual, expected in violations_by_type['below_min']:
        print(f"  {provider}: {actual} shifts (need {expected})")
    
    print(f"\nProviders above maximum: {len(violations_by_type['above_max'])}")
    for provider, actual, expected in violations_by_type['above_max']:
        print(f"  {provider}: {actual} shifts (max {expected})")
    
    # Analyze shift distribution
    print("\n=== Shift Distribution Analysis ===")
    shift_type_counts = {}
    for event in events:
        shift_type = event.extendedProps.get("shift_type")
        if shift_type:
            shift_type_counts[shift_type] = shift_type_counts.get(shift_type, 0) + 1
    
    print("Shift type distribution:")
    for shift_type, count in shift_type_counts.items():
        print(f"  {shift_type}: {count} shifts")
    
    # Calculate total available shifts
    total_available = 0
    for day in month_days:
        for shift_type, capacity in DEFAULT_SHIFT_CAPACITY.items():
            total_available += capacity
    
    print(f"\nTotal available shifts: {total_available}")
    print(f"Total assigned shifts: {len(events)}")
    print(f"Coverage: {len(events)/total_available*100:.1f}%")
    
    return validation_result, events

def analyze_scheduler_logic():
    """
    Analyze the scheduler logic to identify potential issues.
    """
    print("\n=== Scheduler Logic Analysis ===")
    
    # Load data
    providers_df, _ = load_providers()
    if providers_df.empty:
        # Use default providers from CSV
        import pandas as pd
        providers_df = pd.read_csv('IMIS_initials.csv')
    providers = providers_df['initials'].tolist()
    global_rules_dict, shift_types, shift_capacity, provider_rules = load_rules()
    
    # Convert global_rules_dict to RuleConfig object
    from models.data_models import RuleConfig
    global_rules = RuleConfig(**global_rules_dict) if global_rules_dict else RuleConfig()
    
    # Analyze provider rules
    print("Provider-specific rules:")
    if isinstance(provider_rules, dict):
        for provider, rules in provider_rules.items():
            if provider in providers:  # Only show current providers
                min_shifts = rules.get("min_shifts", "default")
                max_shifts = rules.get("max_shifts", "default")
                print(f"  {provider}: min={min_shifts}, max={max_shifts}")
    else:
        print(f"  Provider rules is not a dict: {type(provider_rules)}")
    
    # Analyze global rules
    print(f"\nGlobal rules:")
    print(f"  min_shifts_per_month: {global_rules.min_shifts_per_month}")
    print(f"  max_shifts_per_month: {global_rules.max_shifts_per_month}")
    print(f"  min_days_between_shifts: {global_rules.min_days_between_shifts}")
    
    # Analyze shift capacity
    print(f"\nShift capacity:")
    for shift_type, capacity in DEFAULT_SHIFT_CAPACITY.items():
        print(f"  {shift_type}: {capacity} slots per day")

def test_multiple_months():
    """
    Test the algorithm across multiple months to identify patterns.
    """
    print("\n=== Testing Multiple Months ===")
    
    test_months = [
        (2024, 11),  # 30 days
        (2024, 12),  # 31 days
        (2025, 1),   # 31 days
        (2025, 2),   # 28 days (leap year)
    ]
    
    for year, month in test_months:
        print(f"\n--- Testing {month}/{year} ---")
        try:
            validation_result, events = test_shift_count_validation(year, month)
            
            # Quick summary
            violations = validation_result['violations']
            above_max_violations = [v for v in violations if "max" in v.lower()]
            below_min_violations = [v for v in violations if "min" in v.lower()]
            
            print(f"  Above max violations: {len(above_max_violations)}")
            print(f"  Below min violations: {len(below_min_violations)}")
            
        except Exception as e:
            print(f"  Error testing {month}/{year}: {e}")

if __name__ == "__main__":
    print("Starting shift validation tests...")
    
    # Analyze scheduler logic first
    analyze_scheduler_logic()
    
    # Test current month
    current_date = datetime.now()
    test_shift_count_validation(current_date.year, current_date.month)
    
    # Test multiple months
    test_multiple_months()
    
    print("\n=== Test Complete ===")
