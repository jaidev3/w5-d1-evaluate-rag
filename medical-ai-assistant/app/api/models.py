"""
Pydantic models for FastAPI endpoints.
Defines request and response schemas for the Medical AI Assistant API.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Request model for medical queries."""
    
    query: str = Field(..., description="Medical query text", min_length=1, max_length=1000)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for processing")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    evaluate_with_ragas: bool = Field(default=True, description="Evaluate response with RAGAS metrics")
    retrieval_strategy: str = Field(default="similarity", description="Retrieval strategy to use")
    max_documents: Optional[int] = Field(default=None, description="Maximum number of documents to retrieve")
    
    @validator("retrieval_strategy")
    def validate_strategy(cls, v):
        allowed_strategies = ["similarity", "hybrid", "medical_focused"]
        if v not in allowed_strategies:
            raise ValueError(f"Strategy must be one of {allowed_strategies}")
        return v


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    
    filename: str
    file_path: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    chunk_tokens: int
    medical_sections: List[str]
    document_type: str
    source: str
    processed_at: str


class SourceDocument(BaseModel):
    """Source document model."""
    
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    relevance_score: float = Field(..., description="Relevance score for the query")


class RAGASMetrics(BaseModel):
    """RAGAS evaluation metrics."""
    
    faithfulness_score: float = Field(..., description="Faithfulness score (0-1)")
    answer_relevancy_score: float = Field(..., description="Answer relevancy score (0-1)")
    context_precision_score: float = Field(..., description="Context precision score (0-1)")
    context_recall_score: Optional[float] = Field(None, description="Context recall score (0-1)")
    overall_score: float = Field(..., description="Overall RAGAS score (0-1)")
    passes_thresholds: bool = Field(..., description="Whether response passes quality thresholds")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")
    recommendations: List[str] = Field(..., description="Improvement recommendations")


class QueryResponse(BaseModel):
    """Response model for medical queries."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated medical response")
    sources: Optional[List[SourceDocument]] = Field(None, description="Source documents used")
    ragas_metrics: Optional[RAGASMetrics] = Field(None, description="RAGAS evaluation metrics")
    
    # Performance metrics
    retrieval_time: float = Field(..., description="Document retrieval time in seconds")
    generation_time: float = Field(..., description="Response generation time in seconds")
    total_time: float = Field(..., description="Total processing time in seconds")
    
    # Model information
    model_used: str = Field(..., description="Language model used for generation")
    prompt_template: str = Field(..., description="Prompt template used")
    
    # Safety information
    safety_flags: List[str] = Field(..., description="Safety flags raised during processing")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    filename: str = Field(..., description="Document filename")
    content_type: str = Field(..., description="Document content type")
    document_type: str = Field(default="medical", description="Type of medical document")
    
    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = ["application/pdf", "text/plain"]
        if v not in allowed_types:
            raise ValueError(f"Content type must be one of {allowed_types}")
        return v


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Upload status message")
    document_id: Optional[str] = Field(None, description="Unique document identifier")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    total_tokens: Optional[int] = Field(None, description="Total tokens processed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    components: Dict[str, Any] = Field(..., description="Component health status")
    version: str = Field(..., description="Application version")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    
    system_metrics: Dict[str, Any] = Field(..., description="System performance metrics")
    ragas_metrics: Dict[str, Any] = Field(..., description="RAGAS evaluation metrics")
    vector_store_stats: Dict[str, Any] = Field(..., description="Vector store statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")


class EvaluationRequest(BaseModel):
    """Request model for RAGAS evaluation."""
    
    queries: List[str] = Field(..., description="List of queries to evaluate")
    answers: List[str] = Field(..., description="List of generated answers")
    contexts: List[List[str]] = Field(..., description="List of context lists")
    ground_truths: Optional[List[str]] = Field(None, description="Optional ground truth answers")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for RAGAS evaluation")
    
    @validator("queries", "answers", "contexts")
    def validate_equal_lengths(cls, v, values):
        if "queries" in values and len(v) != len(values["queries"]):
            raise ValueError("All lists must have the same length")
        return v


class EvaluationResponse(BaseModel):
    """Response model for RAGAS evaluation."""
    
    total_evaluations: int = Field(..., description="Total number of evaluations")
    average_scores: Dict[str, float] = Field(..., description="Average RAGAS scores")
    pass_rates: Dict[str, float] = Field(..., description="Threshold pass rates")
    individual_results: List[RAGASMetrics] = Field(..., description="Individual evaluation results")
    summary: Dict[str, Any] = Field(..., description="Evaluation summary")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ConfigResponse(BaseModel):
    """Configuration response model."""
    
    ragas_thresholds: Dict[str, float] = Field(..., description="RAGAS quality thresholds")
    model_settings: Dict[str, Any] = Field(..., description="Model configuration")
    retrieval_settings: Dict[str, Any] = Field(..., description="Retrieval configuration")
    safety_settings: Dict[str, Any] = Field(..., description="Safety configuration")


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    
    queries: List[str] = Field(..., description="List of medical queries", min_items=1, max_items=50)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for processing")
    include_sources: bool = Field(default=True, description="Include source documents")
    evaluate_with_ragas: bool = Field(default=True, description="Evaluate with RAGAS")
    retrieval_strategy: str = Field(default="similarity", description="Retrieval strategy")
    
    @validator("queries")
    def validate_queries(cls, v):
        if not v:
            raise ValueError("At least one query is required")
        for query in v:
            if not query.strip():
                raise ValueError("Empty queries are not allowed")
        return v


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    results: List[QueryResponse] = Field(..., description="Individual query results")
    batch_metrics: Dict[str, Any] = Field(..., description="Batch processing metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch processing timestamp") 