"""
FastAPI application entry point
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.logging import setup_logging

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    yield
    
    # Shutdown
    # Add any cleanup code here
    pass

# Create FastAPI app
app = FastAPI(
    title="Gaffer",
    description="AI-powered workflow automation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")