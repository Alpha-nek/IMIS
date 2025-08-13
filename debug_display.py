#!/usr/bin/env python3
"""
Add this code to your Streamlit app to display the debug output.
"""

import streamlit as st
from core.debug_logger import debug_logger

def display_scheduler_debug():
    """
    Add this function to your Streamlit app to show the scheduler debug output.
    
    Usage in your app:
    ```python
    from debug_display import display_scheduler_debug
    
    # After generating schedule
    display_scheduler_debug()
    ```
    """
    
    debug_messages = debug_logger.get_log_text()
    
    if debug_messages:
        with st.expander("🔍 Scheduler Debug Output", expanded=True):
            st.text_area(
                "Debug Log", 
                debug_messages, 
                height=400,
                help="This shows the step-by-step progress of the scheduling algorithm"
            )
            
            # Add a button to clear the debug log
            if st.button("Clear Debug Log"):
                debug_logger.clear()
                st.rerun()
    else:
        st.info("No debug output available. Generate a schedule to see debug information.")

# Example usage - add this to your main Streamlit app file
def example_usage():
    """Example of how to integrate this into your app."""
    st.title("Schedule Generation with Debug")
    
    if st.button("Generate Schedule"):
        # Your existing schedule generation code here
        # events = generate_schedule(...)
        st.success("Schedule generated!")
        
        # Display debug output
        display_scheduler_debug()

if __name__ == "__main__":
    example_usage()
