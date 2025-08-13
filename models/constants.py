# =============================================================================
# Constants and Configuration for IMIS Scheduler
# =============================================================================

from datetime import date, timedelta
from typing import Dict, Any

# Google Calendar API Configuration
GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
GCAL_TOKEN_FILE = 'token.json'          # Created on first successful auth
GCAL_CREDENTIALS_FILE = 'credentials.json'  # Download from Google Cloud
APP_TIMEZONE = 'America/New_York'       # Application timezone

# Default Shift Types Configuration
DEFAULT_SHIFT_TYPES = [
    {"key": "R12", "label": "7am–7pm Rounder",   "start": "07:00", "end": "19:00", "color": "#16a34a"},
    {"key": "A12", "label": "7am–7pm Admitter",  "start": "07:00", "end": "19:00", "color": "#f59e0b"},
    {"key": "A10", "label": "10am–10pm Admitter", "start": "10:00", "end": "22:00", "color": "#ef4444"},
    {"key": "N12", "label": "7pm–7am (Night)", "start": "19:00", "end": "07:00", "color": "#7c3aed"},
    {"key": "NB",  "label": "Night Bridge",     "start": "23:00", "end": "07:00", "color": "#06b6d4"},
    {"key": "APP", "label": "APP Provider",      "start": "07:00", "end": "19:00", "color": "#8b5cf6"},
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Provider Rosters - Updated to match CSV file exactly
PROVIDER_INITIALS_DEFAULT = [
    "AA","AD","AM","FS","JL","JM","JT","KA","LN","SM","OI","NP","PR","UN",
    "DP","FY","YL","RR","SD","JK","NS","PD","AB","KF","AL","GB","KD","NG","GI","VT","DI","YD",
    "HS","YA","NM","EM","SS","YS","HW","AH","RJ","SI","FH","EB","RS","RG","CJ","MS","AT",
    "YH","XL","MA","LM","MQ","CM","AI"
]

# APP Provider roster - these providers can only take APP shifts
APP_PROVIDER_INITIALS = ["JA", "DN", "KP", "AR"]

# Default shift capacities
DEFAULT_SHIFT_CAPACITY = {"N12": 4, "NB": 1, "R12": 13, "A12": 1, "A10": 2, "APP": 2}

# Holiday rules - reduced capacity on major holidays
HOLIDAY_RULES = {
    "thanksgiving": {
        "date_func": lambda year: date(year, 11, 4) + timedelta(days=(3 - date(year, 11, 4).weekday()) % 7 + 21),  # 4th Thursday
        "capacity_multiplier": 0.6,  # 60% of normal capacity
        "description": "Thanksgiving Day"
    },
    "christmas": {
        "date_func": lambda year: date(year, 12, 25),
        "capacity_multiplier": 0.6,  # 60% of normal capacity
        "description": "Christmas Day"
    },
    "new_years": {
        "date_func": lambda year: date(year, 1, 1),
        "capacity_multiplier": 0.6,  # 60% of normal capacity
        "description": "New Year's Day"
    }
}
