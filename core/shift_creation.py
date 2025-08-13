# =============================================================================
# Shift Event Creation and Utilities
# =============================================================================

import uuid
import logging
from typing import Dict
from datetime import datetime, date, timedelta
from models.data_models import SEvent
from core.utils import parse_time

logger = logging.getLogger(__name__)

def create_shift_event(provider: str, shift_type: str, day: date) -> SEvent:
    """
    Create a shift event for a provider.
    """
    try:
        shift_config = get_shift_config(shift_type)
        start_time = datetime.combine(day, parse_time(shift_config["start"]))
        end_time = datetime.combine(day, parse_time(shift_config["end"]))
        
        # Handle overnight shifts
        if shift_config["end"] < shift_config["start"]:
            end_time += timedelta(days=1)
        
        return SEvent(
            id=str(uuid.uuid4()),
            title=f"{shift_config['label']} - {provider}",
            start=start_time,
            end=end_time,
            backgroundColor=shift_config["color"],
            extendedProps={
                "provider": provider,
                "shift_type": shift_type,
                "shift_label": shift_config["label"]
            }
        )
    except Exception as e:
        logger.error(f"Error creating shift event: {e}")
        raise

def get_shift_config(shift_type: str) -> Dict:
    """
    Get configuration for a specific shift type.
    """
    shift_configs = {
        "R12": {
            "label": "7am Rounding",
            "start": "07:00",
            "end": "19:00",
            "color": "#3b82f6"
        },
        "A12": {
            "label": "7am Admitting",
            "start": "07:00",
            "end": "19:00",
            "color": "#10b981"
        },
        "A10": {
            "label": "10am Admitting",
            "start": "10:00",
            "end": "22:00",
            "color": "#059669"
        },
        "N12": {
            "label": "7pm Night",
            "start": "19:00",
            "end": "07:00",
            "color": "#1e293b"
        },
        "NB": {
            "label": "Bridge",
            "start": "15:00",
            "end": "03:00",
            "color": "#7c3aed"
        },
        "APP": {
            "label": "APP Provider",
            "start": "07:00",
            "end": "19:00",
            "color": "#8b5cf6"
        }
    }
    
    return shift_configs.get(shift_type, {
        "label": f"Unknown Shift ({shift_type})",
        "start": "07:00",
        "end": "19:00",
        "color": "#6b7280"
    })
