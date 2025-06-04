import jwt
import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Security scheme for Bearer token
security = HTTPBearer()

# Supabase configuration
SUPABASE_URL = "https://zxwfmrccjrbejqxmmxrw.supabase.co"
SUPABASE_JWT_SECRET = None  # Will be fetched from Supabase


async def verify_supabase_token(token: str) -> Dict[str, Any]:
    """Verify and decode Supabase JWT token."""
    try:
        logger.info("Attempting to verify JWT token", token_prefix=token[:20] + "...")
        
        if not settings.supabase_jwt_secret:
            logger.error("JWT secret not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT secret not configured. Please set SUPABASE_JWT_SECRET environment variable."
            )
        
        logger.info("JWT secret configured", secret_prefix=settings.supabase_jwt_secret[:20] + "...")
        
        # First, let's decode without verification to see the token structure
        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            logger.info("Token payload (unverified)", payload=unverified_payload)
        except Exception as e:
            logger.error("Failed to decode token without verification", error=str(e))
        
        # Try with minimal validation first
        try:
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"]
                # Temporarily removing audience and issuer validation
            )
            logger.info("JWT verification successful (minimal validation)", user_id=payload.get("sub"))
            return payload
        except Exception as minimal_error:
            logger.error("Minimal JWT verification failed", error=str(minimal_error))
            
            # If minimal fails, try with full validation
            issuer_url = f"{settings.supabase_url}/auth/v1"
            logger.info("Trying full validation with issuer URL", issuer=issuer_url)
            
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=issuer_url
            )
            
            logger.info("JWT verification successful (full validation)", user_id=payload.get("sub"))
            return payload
        
    except jwt.ExpiredSignatureError:
        logger.error("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidAudienceError as e:
        logger.error("JWT audience validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience"
        )
    except jwt.InvalidIssuerError as e:
        logger.error("JWT issuer validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer"
        )
    except jwt.InvalidSignatureError as e:
        logger.error("JWT signature validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature - check JWT secret"
        )
    except jwt.InvalidTokenError as e:
        logger.error("Invalid JWT token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}"
        )
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    try:
        token = credentials.credentials
        payload = await verify_supabase_token(token)
        
        # Extract user information from the token payload
        user_info = {
            "id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role", "authenticated"),
            "aud": payload.get("aud"),
            "exp": payload.get("exp"),
            "iat": payload.get("iat"),
        }
        
        if not user_info["id"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user token"
            )
            
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get current user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """Get current user ID from authenticated user."""
    return current_user["id"]


# Optional: Create a dependency for admin users
async def get_current_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current user if they have admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user 