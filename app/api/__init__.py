"""
API package for Gaffer
"""

from app.api.chain_router import router as chain_router

# Combine all routers into a single router if needed
# from fastapi import APIRouter
# router = APIRouter()
# router.include_router(chain_router, prefix="/chains", tags=["Chains"])

__all__ = ["chain_router"]
