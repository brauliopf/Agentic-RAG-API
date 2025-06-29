import logging
import structlog
from typing import Any, Dict

from .config import settings


def event_name_first_processor(logger, method_name, event_dict):
    """
    Ensure 'event' key is present and move it to the start of the event dict.
    Raise ValueError if 'event' is missing.
    """
    event = event_dict.pop("event", None)
    if event is None:
        return event_dict
    # Rebuild the dict with 'event' as the first key
    new_event_dict = {"event": event}
    new_event_dict.update(event_dict)
    return new_event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            event_name_first_processor,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s"
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name) 