#!/usr/bin/env python3
"""
Debug test for the IMIS scheduler to identify why only one shift is being assigned.
"""

import sys
import logging
from datetime import date

# Add the current directory to Python path
sys.path.append('.')

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.scheduler import generate_schedule
from models.data_models import RuleConfig
from models.constants import DEFAULT_SHIFT_TYPES, DEFAULT_SHIFT_CAPACITY

def test_scheduler():
    """Test the scheduler with sample data."""
    
    # Test parameters
    year, month = 2024, 1
    providers = ['AB', 'CD', 'EF', 'GH', 'IJ']  # 5 providers should be enough
    shift_types = DEFAULT_SHIFT_TYPES
    shift_capacity = DEFAULT_SHIFT_CAPACITY
    provider_rules = {}
    global_rules = RuleConfig()
    
    print(f"Testing schedule generation for {year}-{month:02d}")
    print(f"Providers: {providers}")
    print(f"Shift types: {[st['key'] for st in shift_types]}")
    print(f"Shift capacity: {shift_capacity}")
    print("-" * 60)
    
    try:
        events = generate_schedule(year, month, providers, shift_types, 
                                 shift_capacity, provider_rules, global_rules)
        
        print(f"\n✅ Generated {len(events)} events")
        
        if events:
            print("\nFirst 10 events:")
            for i, event in enumerate(events[:10]):
                provider = event.extendedProps.get("provider")
                shift_type = event.extendedProps.get("shift_type")
                day = event.start.date()
                print(f"  {i+1:2d}. {provider} - {shift_type} on {day}")
            
            # Analyze by shift type
            shift_counts = {}
            for event in events:
                shift_type = event.extendedProps.get("shift_type")
                shift_counts[shift_type] = shift_counts.get(shift_type, 0) + 1
            
            print(f"\nShift distribution:")
            for shift_type, count in shift_counts.items():
                expected = shift_capacity.get(shift_type, 1) * 31  # 31 days in January
                if shift_type == "APP":
                    # APP has variable capacity
                    expected = "Variable (1-2 per day)"
                print(f"  {shift_type}: {count} assigned (expected ~{expected})")
        else:
            print("❌ No events generated!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scheduler()
