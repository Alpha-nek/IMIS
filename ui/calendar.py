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
    Render the calendar using Streamlit components with full width and improved styling.
    Handles both SEvent objects and dictionaries.
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
            
            #calendar {{
                width: 100%;
                height: 100%;
                background: white;
            }}
            
            .fc {{
                font-size: 14px;
                line-height: 1.4;
            }}
            
            .fc-header-toolbar {{
                padding: 10px;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            
            .fc-toolbar-title {{
                font-size: 1.5rem;
                font-weight: 600;
                color: #2c3e50;
            }}
            
            .fc-button {{
                background: #007bff;
                border: 1px solid #007bff;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            
            .fc-button:hover {{
                background: #0056b3;
                border-color: #0056b3;
            }}
            
            .fc-button:active {{
                background: #004085;
            }}
            
            .fc-daygrid-day {{
                border: 1px solid #dee2e6;
            }}
            
            .fc-daygrid-day-number {{
                font-weight: 500;
                color: #495057;
                padding: 8px;
            }}
            
            .fc-day-today {{
                background: #fff3cd !important;
            }}
            
            .fc-daygrid-day.fc-day-today .fc-daygrid-day-number {{
                background: #ffc107;
                color: #212529;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 4px;
            }}
            
            .fc-event {{
                cursor: pointer;
                border-radius: 4px;
                padding: 2px 4px;
                margin: 1px 0;
                font-size: 12px;
                font-weight: 500;
                border: none;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            .fc-event:hover {{
                opacity: 0.8;
                transform: translateY(-1px);
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }}
            
            .fc-event-title {{
                font-weight: 600;
            }}
            
            .fc-col-header-cell {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 8px;
                font-weight: 600;
                color: #495057;
                text-align: center;
            }}
            
            .fc-daygrid-day-frame {{
                min-height: 100px;
            }}
            
            .fc-daygrid-day-events {{
                margin: 2px;
            }}
            
            .fc-daygrid-event-dot {{
                border-width: 4px;
            }}
            
            .fc-daygrid-day.fc-day-past {{
                background: #f8f9fa;
            }}
            
            .fc-daygrid-day.fc-day-future {{
                background: white;
            }}
            
            .fc-daygrid-day.fc-day-other {{
                background: #e9ecef;
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
                width: '100%',
                aspectRatio: 1.35,
                headerToolbar: {{
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth,dayGridWeek'
                }},
                buttonText: {{
                    today: 'Today',
                    month: 'Month',
                    week: 'Week'
                }},
                events: {json.dumps(events_json)},
                eventDisplay: 'block',
                eventTimeFormat: {{
                    hour: 'numeric',
                    minute: '2-digit',
                    meridiem: 'short'
                }},
                dayMaxEvents: 5,
                moreLinkClick: 'popover',
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
                eventDidMount: function(info) {{
                    // Add custom styling based on event type
                    var event = info.event;
                    var element = info.el;
                    
                    // Add tooltip
                    element.title = event.title + ' (' + event.extendedProps.shift_type + ')';
                    
                    // Add custom classes based on shift type
                    if (event.extendedProps.shift_type) {{
                        element.classList.add('shift-' + event.extendedProps.shift_type.toLowerCase());
                    }}
                }}
            }});
            calendar.render();
        </script>
    </body>
    </html>
    """
    
    # Use full container width
    components.html(calendar_html, height=height, scrolling=False)

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
        if st.button("← Previous Month", use_container_width=True):
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
        st.markdown(f"**{current_date.strftime('%B %Y')}**", help="Current month and year")
    
    with col3:
        if st.button("Next Month →", use_container_width=True):
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
