"""
Configuration management API for Gaffer.
Provides endpoints to create, retrieve, update, and delete named configurations.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import os
import uuid
import json
from pathlib import Path

from app.models.config import LLMConfig
from app.models.vector_store import VectorStoreConfig

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory storage for configurations
# In a production environment, this should be replaced with a database
llm_configs: Dict[str, LLMConfig] = {}
vector_store_configs: Dict[str, VectorStoreConfig] = {}

# Path for persistent storage
CONFIG_DIR = Path("config")
LLM_CONFIG_FILE = CONFIG_DIR / "llm_configs.json"
VECTOR_STORE_CONFIG_FILE = CONFIG_DIR / "vector_store_configs.json"

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)

# Load configurations from files if they exist
def load_configs():
    """Load configurations from disk on startup."""
    global llm_configs, vector_store_configs
    
    try:
        if LLM_CONFIG_FILE.exists():
            with open(LLM_CONFIG_FILE, "r") as f:
                config_dict = json.load(f)
                llm_configs = {k: LLMConfig.parse_obj(v) for k, v in config_dict.items()}
                logger.info(f"Loaded {len(llm_configs)} LLM configurations from disk")
    except Exception as e:
        logger.error(f"Error loading LLM configurations: {e}", exc_info=True)
    
    try:
        if VECTOR_STORE_CONFIG_FILE.exists():
            with open(VECTOR_STORE_CONFIG_FILE, "r") as f:
                config_dict = json.load(f)
                vector_store_configs = {k: VectorStoreConfig.parse_obj(v) for k, v in config_dict.items()}
                logger.info(f"Loaded {len(vector_store_configs)} Vector Store configurations from disk")
    except Exception as e:
        logger.error(f"Error loading Vector Store configurations: {e}", exc_info=True)

# Save configurations to files
def save_llm_configs():
    """Save LLM configurations to disk."""
    try:
        config_dict = {k: v.dict() for k, v in llm_configs.items()}
        with open(LLM_CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved {len(llm_configs)} LLM configurations to disk")
    except Exception as e:
        logger.error(f"Error saving LLM configurations: {e}", exc_info=True)

def save_vector_store_configs():
    """Save Vector Store configurations to disk."""
    try:
        config_dict = {k: v.dict() for k, v in vector_store_configs.items()}
        with open(VECTOR_STORE_CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved {len(vector_store_configs)} Vector Store configurations to disk")
    except Exception as e:
        logger.error(f"Error saving Vector Store configurations: {e}", exc_info=True)

# Load configs at module import time
load_configs()

# Create default configs from environment variables if needed
def create_default_configs():
    """Create default configurations from environment variables if they don't exist."""
    # Default LLM config
    if "default" not in llm_configs and os.getenv("OPENAI_API_KEY"):
        llm_configs["default"] = LLMConfig(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        )
        save_llm_configs()
        logger.info("Created default LLM configuration from environment variables")
    
    # Default Vector Store config
    if "default" not in vector_store_configs and os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_ENVIRONMENT"):
        vector_store_configs["default"] = VectorStoreConfig(
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
        save_vector_store_configs()
        logger.info("Created default Vector Store configuration from environment variables")

# Try to create default configs
create_default_configs()

class ConfigCreationRequest(BaseModel):
    """Base model for configuration creation requests."""
    name: str
    description: Optional[str] = None

class LLMConfigRequest(ConfigCreationRequest):
    """Request model for creating an LLM configuration."""
    config: LLMConfig

class VectorStoreConfigRequest(ConfigCreationRequest):
    """Request model for creating a Vector Store configuration."""
    config: VectorStoreConfig

class ConfigListItem(BaseModel):
    """Model for configuration list items."""
    id: str
    name: str
    description: Optional[str] = None
    type: str

class ConfigReference(BaseModel):
    """Model for referencing a configuration."""
    config_id: Optional[str] = None  # "default" or custom ID
    inline_config: Optional[Any] = None  # LLMConfig or VectorStoreConfig

# LLM Configuration Endpoints

@router.post(
    "/configs/llm",
    summary="Create a new LLM configuration",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def create_llm_config(request: LLMConfigRequest):
    """
    Creates a new named LLM configuration.
    """
    # Generate a unique ID if needed
    config_id = str(uuid.uuid4())
    
    try:
        # Store the configuration
        llm_configs[config_id] = request.config
        save_llm_configs()
        
        logger.info(f"Created LLM configuration '{request.name}' with ID: {config_id}")
        return {"config_id": config_id, "name": request.name}
    except Exception as e:
        logger.error(f"Failed to create LLM configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LLM configuration: {str(e)}"
        )

@router.get(
    "/configs/llm",
    summary="List all LLM configurations",
    response_model=List[ConfigListItem]
)
async def list_llm_configs():
    """
    Lists all available LLM configurations.
    """
    try:
        configs = [
            ConfigListItem(
                id=config_id,
                name=config_id,  # Using ID as name until we implement proper naming
                description=None,
                type="llm"
            )
            for config_id in llm_configs.keys()
        ]
        return configs
    except Exception as e:
        logger.error(f"Failed to list LLM configurations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list LLM configurations: {str(e)}"
        )

@router.get(
    "/configs/llm/{config_id}",
    summary="Get a specific LLM configuration",
    response_model=LLMConfig
)
async def get_llm_config(config_id: str):
    """
    Retrieves a specific LLM configuration by ID.
    """
    if config_id not in llm_configs:
        logger.warning(f"LLM configuration not found: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM configuration not found"
        )
    
    return llm_configs[config_id]

@router.delete(
    "/configs/llm/{config_id}",
    summary="Delete an LLM configuration",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_llm_config(config_id: str):
    """
    Deletes a specific LLM configuration by ID.
    """
    if config_id not in llm_configs:
        logger.warning(f"Attempt to delete non-existent LLM configuration: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM configuration not found"
        )
    
    # Prevent deleting the default configuration
    if config_id == "default":
        logger.warning("Attempt to delete default LLM configuration")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default configuration"
        )
        
    try:
        del llm_configs[config_id]
        save_llm_configs()
        logger.info(f"Deleted LLM configuration: {config_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to delete LLM configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete LLM configuration: {str(e)}"
        )

# Vector Store Configuration Endpoints

@router.post(
    "/configs/vector-store",
    summary="Create a new Vector Store configuration",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def create_vector_store_config(request: VectorStoreConfigRequest):
    """
    Creates a new named Vector Store configuration.
    """
    # Generate a unique ID
    config_id = str(uuid.uuid4())
    
    try:
        # Store the configuration
        vector_store_configs[config_id] = request.config
        save_vector_store_configs()
        
        logger.info(f"Created Vector Store configuration '{request.name}' with ID: {config_id}")
        return {"config_id": config_id, "name": request.name}
    except Exception as e:
        logger.error(f"Failed to create Vector Store configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create Vector Store configuration: {str(e)}"
        )

@router.get(
    "/configs/vector-store",
    summary="List all Vector Store configurations",
    response_model=List[ConfigListItem]
)
async def list_vector_store_configs():
    """
    Lists all available Vector Store configurations.
    """
    try:
        configs = [
            ConfigListItem(
                id=config_id,
                name=config_id,  # Using ID as name until we implement proper naming
                description=None,
                type="vector_store"
            )
            for config_id in vector_store_configs.keys()
        ]
        return configs
    except Exception as e:
        logger.error(f"Failed to list Vector Store configurations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list Vector Store configurations: {str(e)}"
        )

@router.get(
    "/configs/vector-store/{config_id}",
    summary="Get a specific Vector Store configuration",
    response_model=VectorStoreConfig
)
async def get_vector_store_config(config_id: str):
    """
    Retrieves a specific Vector Store configuration by ID.
    """
    if config_id not in vector_store_configs:
        logger.warning(f"Vector Store configuration not found: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector Store configuration not found"
        )
    
    return vector_store_configs[config_id]

@router.delete(
    "/configs/vector-store/{config_id}",
    summary="Delete a Vector Store configuration",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_vector_store_config(config_id: str):
    """
    Deletes a specific Vector Store configuration by ID.
    """
    if config_id not in vector_store_configs:
        logger.warning(f"Attempt to delete non-existent Vector Store configuration: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector Store configuration not found"
        )
    
    # Prevent deleting the default configuration
    if config_id == "default":
        logger.warning("Attempt to delete default Vector Store configuration")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default configuration"
        )
        
    try:
        del vector_store_configs[config_id]
        save_vector_store_configs()
        logger.info(f"Deleted Vector Store configuration: {config_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to delete Vector Store configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete Vector Store configuration: {str(e)}"
        )

# Helper functions for other routers

def get_llm_config_by_id_or_default(config_id: Optional[str] = None) -> Optional[LLMConfig]:
    """
    Helper function to get an LLM config by ID or return the default.
    Returns None if no configurations are available.
    """
    if not config_id and "default" in llm_configs:
        return llm_configs["default"]
    elif config_id and config_id in llm_configs:
        return llm_configs[config_id]
    return None

def get_vector_store_config_by_id_or_default(config_id: Optional[str] = None) -> Optional[VectorStoreConfig]:
    """
    Helper function to get a Vector Store config by ID or return the default.
    Returns None if no configurations are available.
    """
    if not config_id and "default" in vector_store_configs:
        return vector_store_configs["default"]
    elif config_id and config_id in vector_store_configs:
        return vector_store_configs[config_id]
    return None 