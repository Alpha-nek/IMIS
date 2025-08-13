# =============================================================================
# Debug Logger for Streamlit
# =============================================================================

import time
from typing import List

class DebugLogger:
    """Simple debug logger that collects messages for display in Streamlit."""
    
    def __init__(self):
        self.messages: List[str] = []
        self.start_time = time.time()
    
    def log(self, message: str):
        """Add a debug message with timestamp."""
        elapsed = time.time() - self.start_time
        timestamped_message = f"[{elapsed:.2f}s] {message}"
        self.messages.append(timestamped_message)
        # Also print to console for server-side debugging
        print(timestamped_message)
    
    def get_messages(self) -> List[str]:
        """Get all collected messages."""
        return self.messages
    
    def get_log_text(self) -> str:
        """Get all messages as a single text string."""
        return "\n".join(self.messages)
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
        self.start_time = time.time()

# Global debug logger instance
debug_logger = DebugLogger()
