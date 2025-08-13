# =============================================================================
# Calendar UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, date
from typing import List, Dict, Any, Tuple
import json

from models.data_models import SEvent

def render_calendar(events: List[Any], height: int = 600) -> None:
    """
    Render the calendar using Streamlit components. Handles both SEvent objects and dictionaries.
    """
    # Convert events to JSON format
    events_json = []
    for event in events:
        if hasattr(event, 'to_json_event'):
            events_json.append(event.to_json_event())
        elif isinstance(event, dict):
            events_json.append(event)
        else:
            continue
    
    # Calendar HTML
    calendar_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js'></script>
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/locales-all.global.min.js'></script>
        <style>
            .fc-event {{
                cursor: pointer;
            }}
            .fc-event:hover {{
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div id='calendar'></div>
        <script>
            var calendarEl = document.getElementById('calendar');
            var calendar = new FullCalendar.Calendar(calendarEl, {{
                initialView: 'dayGridMonth',
                height: {height},
                events: {json.dumps(events_json)},
                eventClick: function(info) {{
                    // Send event data to Streamlit
                    window.parent.postMessage({{
                        type: 'event_click',
                        event: info.event.toPlainObject()
                    }}, '*');
                }},
                dateClick: function(info) {{
                    // Send date click to Streamlit
                    window.parent.postMessage({{
                        type: 'date_click',
                        date: info.dateStr
                    }}, '*');
                }}
            }});
            calendar.render();
        </script>
    </body>
    </html>
    """
    
    components.html(calendar_html, height=height)

def render_month_navigation() -> Tuple[int, int]:
    """
    Render month navigation controls.
    """
    # Get current year and month from session state
    current_year = st.session_state.get("current_year", datetime.now().year)
    current_month = st.session_state.get("current_month", datetime.now().month)
    
    # Create a date object for the current month
    current_date = date(current_year, current_month, 1)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous Month"):
            if current_month == 1:
                new_year = current_year - 1
                new_month = 12
            else:
                new_year = current_year
                new_month = current_month - 1
            
            st.session_state.current_year = new_year
            st.session_state.current_month = new_month
            st.rerun()
    
    with col2:
        st.write(f"**{current_date.strftime('%B %Y')}**")
    
    with col3:
        if st.button("Next Month →"):
            if current_month == 12:
                new_year = current_year + 1
                new_month = 1
            else:
                new_year = current_year
                new_month = current_month + 1
            
            st.session_state.current_year = new_year
            st.session_state.current_month = new_month
            st.rerun()
    
    return current_year, current_month
