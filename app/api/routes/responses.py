"""
Centralized definition of API response models.
This module contains standard response models used across API endpoints.
"""

from fastapi import status
from typing import Dict, Any

class ResponseModels:
    """
    Collection of standard API response models for documentation.
    These are used to document the possible responses for each endpoint
    in the OpenAPI specification.
    """
    
    # Standard responses for chat endpoints
    CHAT_RESPONSES: Dict[int, Dict[str, Any]] = {
        status.HTTP_200_OK: {"description": "Successful response."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error."},
        status.HTTP_400_BAD_REQUEST: {"description": "Bad request."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service unavailable."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication error."},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Gateway timeout."}
    }
    
    # Standard responses for UUID generation endpoints
    UUID_RESPONSES: Dict[int, Dict[str, Any]] = {
        status.HTTP_200_OK: {"description": "Successfully generated UUID."},
        status.HTTP_400_BAD_REQUEST: {"description": "Bad request or invalid parameters."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error."},
    }
    
    @classmethod
    def get_chat_responses(cls) -> Dict[int, Dict[str, Any]]:
        """
        Get the standard response models for chat endpoints.
        
        Returns:
            Dictionary of status codes and descriptions
        """
        return cls.CHAT_RESPONSES
    
    @classmethod
    def get_uuid_responses(cls) -> Dict[int, Dict[str, Any]]:
        """
        Get the standard response models for UUID generation endpoints.
        
        Returns:
            Dictionary of status codes and descriptions
        """
        return cls.UUID_RESPONSES

# For simpler imports, expose the response models directly
chat_responses = ResponseModels.get_chat_responses()
uuid_responses = ResponseModels.get_uuid_responses()