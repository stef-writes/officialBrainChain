"""
FastAPI application entry point
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.logging import setup_logger
from app.models.vector_store import VectorStoreConfig
from app.vector.pinecone_store import PineconeVectorStore

# Setup logging
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Initialize vector store if configured
    if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_ENVIRONMENT"):
        try:
            vector_store_config = VectorStoreConfig(
                index_name=os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index-gaffer"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                dimension=1024,  # Llama text embed v2 dimension
                metric="cosine",
                pod_type=os.getenv("PINECONE_POD_TYPE", "serverless"),
                replicas=int(os.getenv("PINECONE_REPLICAS", "1")),
                use_inference=True,
                inference_model="llama-text-embed-v2",
                api_key=os.getenv("PINECONE_API_KEY"),
                host=os.getenv("PINECONE_HOST")
            )
            
            app.state.vector_store = PineconeVectorStore(vector_store_config)
            await app.state.vector_store.initialize()
            logger.info("Initialized Pinecone vector store with inference")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    yield
    
    # Shutdown
    if hasattr(app.state, "vector_store"):
        await app.state.vector_store.cleanup()
        logger.info("Cleaned up vector store resources")

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