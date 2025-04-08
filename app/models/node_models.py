"""
Data models for node configurations and metadata
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime
from .config import LLMConfig, MessageTemplate  # Import MessageTemplate

class NodeMetadata(BaseModel):
    """Metadata model for node versioning and ownership"""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Type of node (ai, logic, data)")
    version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$",
                        description="Semantic version of node configuration")
    owner: Optional[str] = Field(None, description="Node owner/maintainer")
    created_at: datetime = Field(default_factory=datetime.utcnow,
                               description="Creation timestamp")
    modified_at: datetime = Field(default_factory=datetime.utcnow,
                                description="Last modification timestamp")
    description: Optional[str] = Field(None, description="Description of the node")
    error_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode='before')
    @classmethod
    def set_modified_at(cls, values):
        values['modified_at'] = datetime.utcnow()
        return values

class NodeConfig(BaseModel):
    """Complete node configuration model"""
    metadata: NodeMetadata
    llm_config: LLMConfig
    input_schema: Dict[str, str] = Field(default_factory=dict,
                                       description="Expected input parameters and types")
    output_schema: Dict[str, str] = Field(default_factory=dict,
                                        description="Produced output parameters and types")
    dependencies: List[str] = Field(default_factory=list,
                                  description="Node IDs this node depends on")
    timeout: int = Field(30, gt=0, description="Maximum execution time in seconds")
    templates: List[MessageTemplate] = Field(default_factory=list,
                                           description="Message templates for node execution")
    
    @field_validator('dependencies')
    @classmethod
    def check_self_reference(cls, v, info):
        if 'metadata' in info.data and info.data['metadata'].node_id in v:
            raise ValueError("Node cannot depend on itself")
        return v

    model_config = ConfigDict(extra="forbid")  # Prevent unexpected arguments

class NodeExecutionRecord(BaseModel):
    """Execution statistics and historical data"""
    node_id: str
    executions: int = Field(0, ge=0, description="Total execution attempts")
    successes: int = Field(0, ge=0, description="Successful executions")
    failures: int = Field(0, ge=0, description="Failed executions")
    avg_duration: float = Field(0.0, ge=0, description="Average execution time in seconds")
    last_executed: Optional[datetime] = None
    token_usage: Dict[str, int] = Field(default_factory=dict,
                                      description="Token usage by model version")

class NodeIO(BaseModel):
    """Input/Output validation model"""
    inputs: Dict[str, Union[str, int, float, bool, Dict, List]] 
    outputs: Dict[str, Union[str, int, float, bool, Dict, List]] = Field(default_factory=dict)
    context: Dict[str, Union[str, Dict]] = Field(default_factory=dict,
                                              description="Execution context metadata")

class UsageMetadata(BaseModel):
    """Metadata for tracking resource usage during node execution"""
    prompt_tokens: Optional[int] = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(default=0, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(default=0, description="Total number of tokens used")
    api_calls: Optional[int] = Field(default=0, description="Number of API calls made")
    model: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class NodeExecutionResult(BaseModel):
    """Result model for node execution"""
    success: bool = False
    output: Optional[Union[str, Dict[str, Any]]] = None  # Allow both string and dict outputs
    error: Optional[str] = None
    metadata: NodeMetadata
    duration: float = Field(default=0.0, ge=0.0, description="Execution duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    usage: Optional[UsageMetadata] = None  # Add usage metadata field

    def __init__(self, **data):
        # Handle case where metadata is passed as a dict
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata_dict = data["metadata"]
            # Ensure required fields are present
            if "node_id" not in metadata_dict or "node_type" not in metadata_dict:
                raise ValueError("metadata must contain node_id and node_type")
            data["metadata"] = NodeMetadata(**metadata_dict)
        
        # Handle usage metadata if present
        if "usage" in data and isinstance(data["usage"], dict):
            data["usage"] = UsageMetadata(**data["usage"])
        
        super().__init__(**data)

    # Updated Pydantic V2 config with serialization
    model_config = ConfigDict(
        extra="allow",
        json_schema_serialization_defaults={
            datetime: lambda v: v.isoformat()
        }
    )
