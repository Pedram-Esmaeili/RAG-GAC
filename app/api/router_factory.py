"""
Router Factory module.
This module creates and configures all API routers.
"""

from fastapi import APIRouter
from .routes.chat_routes import chat_router
# Import additional routers as needed

class RouterFactory:
    """
    Factory class for creating and configuring API routers.
    This class centralizes router creation and organization.
    """
    
    @staticmethod
    def create_main_router() -> APIRouter:
        """
        Create and return the main API router with all sub-routers included.
        
        Returns:
            FastAPI router configured with all routes
        """
        main_router = APIRouter()
        
        # Include all routers
        main_router.include_router(chat_router)
        # Add additional routers as needed
        
        return main_router
