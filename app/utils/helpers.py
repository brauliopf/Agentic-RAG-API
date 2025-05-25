import re
from typing import Dict, Any
from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_string(text: str) -> str:
    """Sanitize input strings by removing potentially harmful characters."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_processing_time(seconds: float) -> str:
    """Format processing time in a human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def extract_metadata_safely(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract and sanitize metadata."""
    if not isinstance(metadata, dict):
        return {}
    
    safe_metadata = {}
    for key, value in metadata.items():
        if isinstance(key, str) and len(key) <= 100:  # Limit key length
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) <= 1000:  # Limit value length
                    safe_metadata[key] = sanitize_string(value) if isinstance(value, str) else value
                elif not isinstance(value, str):
                    safe_metadata[key] = value
    
    return safe_metadata 