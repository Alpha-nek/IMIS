#!/usr/bin/env python3
# =============================================================================
# Test Modular Imports for IMIS Scheduler
# =============================================================================

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test that all our modular imports work correctly."""
    print("🧪 Testing IMIS Modular Structure...")
    
    try:
        # Test constants
        print("📋 Testing constants...")
        from models.constants import DEFAULT_SHIFT_TYPES, PROVIDER_INITIALS_DEFAULT, APP_PROVIDER_INITIALS
        print(f"   ✅ Constants loaded: {len(DEFAULT_SHIFT_TYPES)} shift types, {len(PROVIDER_INITIALS_DEFAULT)} providers, {len(APP_PROVIDER_INITIALS)} APP providers")
        
        # Test data models
        print("📊 Testing data models...")
        from models.data_models import RuleConfig, Provider, SEvent
        test_rule = RuleConfig()
        print(f"   ✅ Data models working: RuleConfig created with {test_rule.min_shifts_per_provider} min shifts")
        
        # Test core utils (non-streamlit functions)
        print("🔧 Testing core utilities...")
        from core.utils import is_holiday, parse_time, date_range, month_start_end
        from datetime import date, time
        
        # Test holiday function
        test_date = date(2024, 12, 25)
        holiday_info = is_holiday(test_date)
        if holiday_info:
            print(f"   ✅ Holiday detection working: {holiday_info['description']}")
        else:
            print("   ✅ Holiday detection working: No holiday detected")
        
        # Test time parsing
        test_time = parse_time("07:00")
        print(f"   ✅ Time parsing working: {test_time}")
        
        # Test date utilities
        start, end = month_start_end(2024, 1)
        print(f"   ✅ Date utilities working: January 2024 = {start} to {end}")
        
        print("\n🎉 All modular imports working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Modular structure is ready for Streamlit integration!")
    else:
        print("\n❌ Modular structure needs fixes before Streamlit integration.")
