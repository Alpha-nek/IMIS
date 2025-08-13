# =============================================================================
# Custom Exceptions for IMIS Scheduler
# =============================================================================

class IMISError(Exception):
    """Base exception class for IMIS Scheduler."""
    pass

class DataValidationError(IMISError):
    """Raised when data validation fails."""
    pass

class ScheduleGenerationError(IMISError):
    """Raised when schedule generation fails."""
    pass

class ProviderError(IMISError):
    """Raised when there are issues with provider data."""
    pass

class RuleValidationError(IMISError):
    """Raised when rule validation fails."""
    pass

class FileOperationError(IMISError):
    """Raised when file operations fail."""
    pass

class ConfigurationError(IMISError):
    """Raised when configuration is invalid."""
    pass

class DateRangeError(IMISError):
    """Raised when date range operations fail."""
    pass

class ShiftAssignmentError(IMISError):
    """Raised when shift assignment fails."""
    pass

class ValidationError(IMISError):
    """Raised when general validation fails."""
    pass
