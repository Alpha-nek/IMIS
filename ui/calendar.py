# =============================================================================
# Calendar UI Components for IMIS Scheduler
# =============================================================================

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, date
from typing import List, Dict, Any, Tuple
import json

from models.data_models import SEvent

def render_calendar(events: List[Any], height: int = 700) -> None:
    """
    Render the calendar using Streamlit components.
    Handles both SEvent objects and dictionaries.
    """
    # Convert events to JSON format
    events_json = []
    for event in events:
        if hasattr(event, 'to_json_event'):
            # It's an SEvent object
            events_json.append(event.to_json_event())
        elif isinstance(event, dict):
            # It's already a dictionary
            events_json.append(event)
        else:
            # Unknown type, skip
            continue
    
    # Calendar HTML with improved styling
    calendar_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js'></script>
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/locales-all.global.min.js'></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .fc {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .fc-event {{
                cursor: pointer;
                border-radius: 4px;
                border: none;
                font-weight: 500;
                font-size: 12px;
                padding: 2px 4px;
            }}
            .fc-event:hover {{
                opacity: 0.8;
                transform: scale(1.02);
                transition: all 0.2s ease;
            }}
            .fc-daygrid-day {{
                min-height: 80px;
            }}
            .fc-daygrid-day-frame {{
                min-height: 80px;
            }}
            .fc-daygrid-day-events {{
                min-height: 60px;
            }}
            .fc-header-toolbar {{
                margin-bottom: 1em;
            }}
            .fc-toolbar-title {{
                font-size: 1.5em;
                font-weight: 600;
            }}
            .fc-button {{
                background-color: #0068c9;
                border-color: #0068c9;
                border-radius: 4px;
                font-weight: 500;
            }}
            .fc-button:hover {{
                background-color: #0056b3;
                border-color: #0056b3;
            }}
            .fc-button:active {{
                background-color: #004494;
                border-color: #004494;
            }}
            .fc-daygrid-day-number {{
                font-weight: 500;
                color: #333;
            }}
            .fc-col-header-cell {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }}
        </style>
    </head>
    <body>
        <div id='calendar' style="width: 100%; height: 100%;"></div>
        <script>
            var calendarEl = document.getElementById('calendar');
            var calendar = new FullCalendar.Calendar(calendarEl, {{
                initialView: 'dayGridMonth',
                height: {height},
                width: '100%',
                expandRows: true,
                headerToolbar: {{
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth'
                }},
                events: {json.dumps(events_json)},
                eventDisplay: 'block',
                eventTimeFormat: {{
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false
                }},
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
                }},
                dayMaxEvents: 5,
                moreLinkClick: 'popover'
            }});
            calendar.render();
        </script>
    </body>
    </html>
    """
    
    # Use full width container
    components.html(calendar_html, height=height, scrolling=False)

def render_month_navigation() -> Tuple[int, int]:
    """
    Render month navigation controls with improved styling.
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous Month", use_container_width=True):
            current_date = st.session_state.get("current_month", date.today())
            if current_date.month == 1:
                new_date = current_date.replace(year=current_date.year - 1, month=12)
            else:
                new_date = current_date.replace(month=current_date.month - 1)
            st.session_state.current_month = new_date
            st.rerun()
    
    with col2:
        current_date = st.session_state.get("current_month", date.today())
        st.markdown(f"<h3 style='text-align: center; color: #1f77b4;'>{current_date.strftime('%B %Y')}</h3>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next Month →", use_container_width=True):
            current_date = st.session_state.get("current_month", date.today())
            if current_date.month == 12:
                new_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                new_date = current_date.replace(month=current_date.month + 1)
            st.session_state.current_month = new_date
            st.rerun()
    
    st.markdown("---")
    return current_date.year, current_date.month
