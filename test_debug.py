#!/usr/bin/env python3
"""
Simple test script to trigger the scheduler and see debug output
"""

import sys
sys.path.append('.')

from core.scheduler import generate_schedule
from models.data_models import RuleConfig
from models.constants import APP_PROVIDER_INITIALS
from core.provider_types import NOCTURNISTS, SENIORS

def test_scheduler():
    print("🧪 TESTING SCHEDULER WITH DEBUG OUTPUT")
    print("=" * 50)
    
    # Test setup
    year = 2024
    month = 1
    
    shift_types = [
        {'key': 'R12', 'name': '7am–7pm Rounder'},
        {'key': 'A12', 'name': '7am–7pm Admitter'}, 
        {'key': 'A10', 'name': '10am–10pm Admitter'},
        {'key': 'N12', 'name': '7pm–7am (Night)'},
        {'key': 'NB', 'name': 'Night Bridge'},
        {'key': 'APP', 'name': 'APP Provider'}
    ]
    
    shift_capacity = {
        'R12': 2, 
        'A12': 2, 
        'A10': 1, 
        'N12': 1, 
        'NB': 1, 
        'APP': 2
    }
    
    # Define test providers - smaller set for easier debugging
    regular_providers = ['DR', 'GH']  # Just 2 regular providers
    app_providers = list(APP_PROVIDER_INITIALS)[:2]  # First 2 APP providers  
    nocturnists = list(NOCTURNISTS)[:1]  # Just 1 nocturnist
    seniors = list(SENIORS)[:1]  # Just 1 senior
    
    all_providers = regular_providers + app_providers + nocturnists + seniors
    
    print(f"🏥 Testing with {len(all_providers)} providers:")
    print(f"  📋 Regular: {regular_providers}")
    print(f"  🍎 APP: {app_providers}")
    print(f"  🌙 Nocturnists: {nocturnists}")
    print(f"  👨‍⚕️ Seniors: {seniors}")
    print()
    
    # Simple rules
    global_rules = RuleConfig(
        min_days_between_shifts=1,
        max_consecutive_shifts=7,
        max_night_shifts_per_month=10,
        target_shifts_per_month=8
    )
    
    print(f"🔄 Generating schedule for {year}-{month:02d}...")
    print("=" * 50)
    
    try:
        events = generate_schedule(year, month, all_providers, shift_types, shift_capacity, {}, global_rules)
        
        print("=" * 50)
        print(f"✅ SUCCESS: Generated {len(events)} shifts")
        
        # Analyze results
        shift_type_counts = {}
        provider_counts = {}
        
        for event in events:
            shift_type = event.extendedProps.get('shift_type', 'Unknown')
            provider = event.extendedProps.get('provider', 'Unknown')
            
            shift_type_counts[shift_type] = shift_type_counts.get(shift_type, 0) + 1
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        print(f"\n📊 SHIFT TYPE DISTRIBUTION:")
        for shift_type, count in sorted(shift_type_counts.items()):
            print(f"  {shift_type}: {count} shifts")
        
        print(f"\n👥 PROVIDER DISTRIBUTION:")
        for provider, count in sorted(provider_counts.items()):
            print(f"  {provider}: {count} shifts")
        
        # Check if all shift types have assignments
        missing_types = [st['key'] for st in shift_types if st['key'] not in shift_type_counts]
        if missing_types:
            print(f"\n❌ MISSING SHIFT TYPES: {missing_types}")
        else:
            print(f"\n✅ All shift types have assignments!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scheduler()
