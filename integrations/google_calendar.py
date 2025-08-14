# =============================================================================
# Google Calendar Integration for IMIS Scheduler
# =============================================================================

import streamlit as st
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json as _json

from models.constants import GCAL_SCOPES, APP_TIMEZONE
from models.data_models import SEvent


def _tokens_dir() -> str:
    base_dir = os.path.join("data", "tokens")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _token_path_for_provider(provider_initials: str) -> str:
    initials = (provider_initials or "").strip().upper() or "UNKNOWN"
    return os.path.join(_tokens_dir(), f"gcal_token_{initials}.json")


def _load_provider_creds(provider_initials: str) -> Optional[Credentials]:
    token_path = _token_path_for_provider(provider_initials)
    if not os.path.exists(token_path):
        return None
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            info = _json.load(f)
        creds = Credentials.from_authorized_user_info(info, scopes=GCAL_SCOPES)
        # Refresh if needed
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                _save_provider_creds(provider_initials, creds)
            except Exception:
                return None
        return creds
    except Exception:
        return None


def _save_provider_creds(provider_initials: str, creds: Credentials) -> None:
    token_path = _token_path_for_provider(provider_initials)
    try:
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    except Exception:
        pass


def _get_client_config() -> Optional[Dict[str, Any]]:
    # Prefer Streamlit secrets (Secrets acts like a dict but isn't an actual dict)
    client_id = None
    client_secret = None
    try:
        secrets_obj = getattr(st, "secrets", {}) or {}
        # Try nested sections first
        gsec = None
        try:
            gsec = secrets_obj.get("google_oauth", None)
        except Exception:
            gsec = None
        if not gsec:
            try:
                gsec = secrets_obj.get("gcal", None)
            except Exception:
                gsec = None
        if gsec:
            try:
                client_id = gsec.get("client_id", None)
                client_secret = gsec.get("client_secret", None)
            except Exception:
                pass
        # Try top-level fallbacks
        if not client_id:
            try:
                client_id = secrets_obj.get("gcal_client_id", None)
            except Exception:
                client_id = None
        if not client_secret:
            try:
                client_secret = secrets_obj.get("gcal_client_secret", None)
            except Exception:
                client_secret = None
    except Exception:
        # Ignore and fallback to env
        pass

    # Fallback to environment variables
    client_id = client_id or os.environ.get("GCAL_CLIENT_ID")
    client_secret = client_secret or os.environ.get("GCAL_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                "http://localhost",
                "http://localhost:8080/",
                "http://localhost:8501/",
            ]
        }
    }


def sign_in_google(provider_initials: str) -> bool:
    """Interactive OAuth sign-in for the given provider; stores token on success."""
    client_config = _get_client_config()
    if not client_config:
        st.error("Google OAuth client is not configured. Set Streamlit secrets or GCAL_CLIENT_ID/GCAL_CLIENT_SECRET env vars.")
        return False
    try:
        flow = InstalledAppFlow.from_client_config(client_config, scopes=GCAL_SCOPES)
        try:
            # Attempt normal flow (opens system browser)
            creds = flow.run_local_server(port=0)
        except Exception as e:
            # Fallback for environments without a runnable browser
            auth_url, _ = flow.authorization_url(
                prompt='consent', access_type='offline', include_granted_scopes='true'
            )
            st.info("Open the Google sign-in page in your browser, complete consent, and return to this app.")
            st.link_button("Open Google Sign-In", auth_url)
            # Run a local loopback server without trying to open a browser
            creds = flow.run_local_server(port=0, open_browser=False)
        _save_provider_creds(provider_initials, creds)
        return True
    except Exception as e:
        st.error(f"❌ Google sign-in failed: {e}")
        return False


def sign_out_google(provider_initials: str) -> None:
    """Remove stored token for the provider."""
    token_path = _token_path_for_provider(provider_initials)
    try:
        if os.path.exists(token_path):
            os.remove(token_path)
            st.success("Signed out from Google for this provider.")
    except Exception:
        pass


def get_gcal_service(provider_initials: str, interactive: bool = False):
    """Return Google Calendar service for provider. If interactive=True, will prompt sign-in if needed."""
    creds = _load_provider_creds(provider_initials)
    if not creds and interactive:
        if not sign_in_google(provider_initials):
            return None
        creds = _load_provider_creds(provider_initials)
    if not creds:
        return None
    try:
        return build('calendar', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"❌ Failed to build Google Calendar service: {e}")
        return None

def gcal_list_calendars(service) -> List[Tuple[str, str]]:
    """List available calendars."""
    try:
        calendars = service.calendarList().list().execute()
        return [(cal['id'], cal['summary']) for cal in calendars.get('items', [])]
    except Exception as e:
        st.error(f"❌ Failed to list calendars: {e}")
        return []

def local_event_to_gcal_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert local event to Google Calendar format."""
    start_time = datetime.fromisoformat(event["start"]) if isinstance(event.get("start"), str) else event["start"]
    end_time = datetime.fromisoformat(event["end"]) if isinstance(event.get("end"), str) else event["end"]
    
    # Handle timezone
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Create description with provider info
    provider = event.get("extendedProps", {}).get("provider", "")
    shift_type = event.get("extendedProps", {}).get("shift_type", "")
    description = f"Provider: {provider}\nShift Type: {shift_type}\nGenerated by IMIS Scheduler"
    
    return {
        "summary": event["title"],
        "description": description,
        "start": {
            "dateTime": start_str,
            "timeZone": APP_TIMEZONE
        },
        "end": {
            "dateTime": end_str,
            "timeZone": APP_TIMEZONE
        },
        "extendedProperties": {
            "private": {
                "app_event_id": event.get("id", ""),
                "provider": provider,
                "shift_type": shift_type
            }
        }
    }

def gcal_find_by_app_id(service, calendar_id: str, app_event_id: str) -> Optional[Dict]:
    """Find Google Calendar event by app event ID."""
    try:
        events_result = service.events().list(
            calendarId=calendar_id,
            privateExtendedProperty=f"app_event_id={app_event_id}"
        ).execute()
        
        events = events_result.get('items', [])
        return events[0] if events else None
    except Exception:
        return None

def _is_same_event_times(gcal_event: Dict, new_body: Dict) -> bool:
    """Check if Google Calendar event times match new body."""
    try:
        gcal_start = gcal_event.get("start", {}).get("dateTime", "")
        gcal_end = gcal_event.get("end", {}).get("dateTime", "")
        new_start = new_body.get("start", {}).get("dateTime", "")
        new_end = new_body.get("end", {}).get("dateTime", "")
        
        return gcal_start == new_start and gcal_end == new_end
    except Exception:
        return False

def sync_events_to_gcal(service, calendar_id: str, events: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Sync events to Google Calendar."""
    created, updated = 0, 0
    
    for event in events:
        body = local_event_to_gcal_body(event)
        
        # Try to find existing event
        gcal_event = gcal_find_by_app_id(service, calendar_id, event.get("id", ""))
        
        if gcal_event is None:
            # Create new event
            try:
                service.events().insert(calendarId=calendar_id, body=body).execute()
                created += 1
            except Exception as e:
                st.error(f"Failed to create event: {e}")
        else:
            # Update if changed
            if (gcal_event.get("summary") != body["summary"]) or (not _is_same_event_times(gcal_event, body)):
                try:
                    gcal_event["summary"] = body["summary"]
                    gcal_event["start"] = body["start"]
                    gcal_event["end"] = body["end"]
                    gcal_event["description"] = body["description"]
                    gcal_event.setdefault("extendedProperties", {}).setdefault("private", {}).update(
                        body["extendedProperties"]["private"]
                    )
                    service.events().update(calendarId=calendar_id, eventId=gcal_event["id"], body=gcal_event).execute()
                    updated += 1
                except Exception as e:
                    st.error(f"Failed to update event: {e}")
    
    return created, updated

def remove_events_from_gcal(service, calendar_id: str, year: int, month: int) -> int:
    """Remove events from Google Calendar for a specific month."""
    try:
        # Get events for the month
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
        
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_date.isoformat() + "T00:00:00Z",
            timeMax=end_date.isoformat() + "T00:00:00Z",
            privateExtendedProperty="app_event_id=*"
        ).execute()
        
        events = events_result.get('items', [])
        removed = 0
        
        for event in events:
            try:
                service.events().delete(calendarId=calendar_id, eventId=event['id']).execute()
                removed += 1
            except Exception:
                pass
        
        return removed
    except Exception as e:
        st.error(f"Failed to remove events: {e}")
        return 0
