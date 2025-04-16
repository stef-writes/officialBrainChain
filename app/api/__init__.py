"""
API package for Gaffer
"""

from app.api.chain_router import router as chain_router
from app.api.config_router import router as config_router

# Combine all routers into a single router if needed
# from fastapi import APIRouter
# router = APIRouter()
# router.include_router(chain_router, prefix="/chains", tags=["Chains"])
# router.include_router(config_router, prefix="/configs", tags=["Configs"])

__all__ = ["chain_router", "config_router"]
