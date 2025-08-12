# =============================================================================
# Calendar UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, date
from typing import List, Dict, Any, Tuple
import json

from models.data_models import SEvent

def render_calendar(events: List[SEvent], height: int = 600) -> None:
    """
    Render the calendar using Streamlit components.
    """
    # Convert events to JSON format
    events_json = [event.to_json_event() for event in events]
    
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
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous Month"):
            current_date = st.session_state.get("current_month", date.today())
            if current_date.month == 1:
                new_date = current_date.replace(year=current_date.year - 1, month=12)
            else:
                new_date = current_date.replace(month=current_date.month - 1)
            st.session_state.current_month = new_date
            st.rerun()
    
    with col2:
        st.write(f"**{st.session_state.get('current_month', date.today()).strftime('%B %Y')}**")
    
    with col3:
        if st.button("Next Month →"):
            current_date = st.session_state.get("current_month", date.today())
            if current_date.month == 12:
                new_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                new_date = current_date.replace(month=current_date.month + 1)
            st.session_state.current_month = new_date
            st.rerun()
    
    current_date = st.session_state.get("current_month", date.today())
    return current_date.year, current_date.month#initial file
