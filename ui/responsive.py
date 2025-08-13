# =============================================================================
# Responsive Layout System for IMIS Scheduler
# =============================================================================

import streamlit as st
from typing import List, Tuple, Any
import streamlit.components.v1 as components
import json

def get_screen_size() -> str:
    """Detect screen size and return device type."""
    # This is a simplified detection - in production you'd use JavaScript
    # For now, we'll use Streamlit's container width as a proxy
    return "mobile" if st.get_option("server.enableCORS") else "desktop"

def is_mobile() -> bool:
    """Check if current view is mobile."""
    # Placeholder for mobile detection
    # In a real implementation, you'd use JavaScript to detect screen size
    return False  # For now, assume desktop

def responsive_columns(ratios: List[int], gap: str = "small") -> List[Any]:
    """
    Create responsive columns that adapt to screen size.
    
    Args:
        ratios: List of column ratios (e.g., [1, 2, 1])
        gap: Gap size between columns ("small", "medium", "large")
    """
    if is_mobile():
        # On mobile, stack vertically
        return [st.container() for _ in ratios]
    else:
        # On desktop, use columns
        return st.columns(ratios, gap=gap)

def mobile_card(title: str, content: str, color: str = "blue") -> None:
    """Create a mobile-friendly card component."""
    colors = {
        "blue": "#1f77b4",
        "green": "#2ca02c", 
        "orange": "#ff7f0e",
        "red": "#d62728",
        "purple": "#9467bd"
    }
    
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid {colors.get(color, colors['blue'])};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <h4 style="margin: 0 0 8px 0; color: {colors.get(color, colors['blue'])};">{title}</h4>
        <p style="margin: 0; color: #333; line-height: 1.5;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def mobile_button(text: str, key: str, type: str = "primary") -> bool:
    """Create a mobile-friendly button."""
    button_styles = {
        "primary": "background-color: #1f77b4; color: white;",
        "secondary": "background-color: #6c757d; color: white;",
        "success": "background-color: #28a745; color: white;",
        "danger": "background-color: #dc3545; color: white;",
        "warning": "background-color: #ffc107; color: black;"
    }
    
    style = button_styles.get(type, button_styles["primary"])
    
    st.markdown(f"""
    <style>
        .mobile-btn-{key} {{
            {style}
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
            margin: 4px 0;
            transition: all 0.3s ease;
        }}
        .mobile-btn-{key}:hover {{
            opacity: 0.8;
            transform: translateY(-1px);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    return st.button(text, key=key, use_container_width=True)

def mobile_tabs(tab_names: List[str]) -> Tuple[str, Any]:
    """Create mobile-friendly tabs."""
    if is_mobile():
        # On mobile, use selectbox for tabs
        selected_tab = st.selectbox("Select Section", tab_names)
        return selected_tab, st.container()
    else:
        # On desktop, use regular tabs
        tabs = st.tabs(tab_names)
        return None, tabs

def mobile_calendar(events: List[Any], height: int = 500) -> None:
    """Mobile-optimized calendar component."""
    if is_mobile():
        # Mobile calendar with touch-friendly interface
        render_mobile_calendar(events, height)
    else:
        # Desktop calendar - use the mobile calendar for now to avoid import issues
        render_mobile_calendar(events, height)

def render_mobile_calendar(events: List[Any], height: int = 500) -> None:
    """Render a mobile-optimized calendar."""
    # Convert events to JSON format
    events_json = []
    for event in events:
        if hasattr(event, 'to_json_event'):
            events_json.append(event.to_json_event())
        elif isinstance(event, dict):
            events_json.append(event)
    
    # Mobile-optimized calendar HTML
    calendar_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js'></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .fc {{
                font-size: 14px;
            }}
            .fc-event {{
                cursor: pointer;
                border-radius: 6px;
                border: none;
                font-weight: 500;
                font-size: 11px;
                padding: 3px 6px;
                margin: 1px 0;
            }}
            .fc-event:hover {{
                opacity: 0.8;
            }}
            .fc-daygrid-day {{
                min-height: 60px;
            }}
            .fc-daygrid-day-frame {{
                min-height: 60px;
            }}
            .fc-daygrid-day-events {{
                min-height: 40px;
            }}
            .fc-header-toolbar {{
                margin-bottom: 0.5em;
                flex-direction: column;
            }}
            .fc-toolbar-chunk {{
                margin: 2px 0;
            }}
            .fc-toolbar-title {{
                font-size: 1.2em;
                font-weight: 600;
            }}
            .fc-button {{
                background-color: #1f77b4;
                border-color: #1f77b4;
                border-radius: 6px;
                font-weight: 500;
                padding: 8px 12px;
                font-size: 14px;
            }}
            .fc-button:hover {{
                background-color: #0056b3;
                border-color: #0056b3;
            }}
            .fc-daygrid-day-number {{
                font-weight: 500;
                color: #333;
                font-size: 12px;
            }}
            .fc-col-header-cell {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
                font-size: 12px;
                padding: 4px 0;
            }}
            @media (max-width: 768px) {{
                .fc-event {{
                    font-size: 10px;
                    padding: 2px 4px;
                }}
                .fc-toolbar-title {{
                    font-size: 1em;
                }}
                .fc-button {{
                    padding: 6px 10px;
                    font-size: 12px;
                }}
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
                    left: 'prev,next',
                    center: 'title',
                    right: 'today'
                }},
                events: {json.dumps(events_json)},
                eventDisplay: 'block',
                eventTimeFormat: {{
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false
                }},
                eventClick: function(info) {{
                    window.parent.postMessage({{
                        type: 'event_click',
                        event: info.event.toPlainObject()
                    }}, '*');
                }},
                dateClick: function(info) {{
                    window.parent.postMessage({{
                        type: 'date_click',
                        date: info.dateStr
                    }}, '*');
                }},
                dayMaxEvents: 3,
                moreLinkClick: 'popover',
                height: 'auto'
            }});
            calendar.render();
        </script>
    </body>
    </html>
    """
    
    components.html(calendar_html, height=height, scrolling=False)
