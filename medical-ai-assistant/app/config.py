"""
Configuration management for Medical AI Assistant.
Uses Pydantic settings for type validation and environment variable loading.
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")


class APIConfig(BaseSettings):
    """FastAPI configuration."""
    
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(4, env="API_WORKERS")
    reload: bool = Field(True, env="RELOAD_ON_CHANGE")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )


class StreamlitConfig(BaseSettings):
    """Streamlit configuration."""
    
    host: str = Field("0.0.0.0", env="STREAMLIT_HOST")
    port: int = Field(8501, env="STREAMLIT_PORT")


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    
    path: str = Field("./data/vector_db", env="VECTOR_DB_PATH")
    collection_name: str = Field("medical_documents", env="VECTOR_DB_COLLECTION_NAME")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""
    
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_retrieval_documents: int = Field(5, env="MAX_RETRIEVAL_DOCUMENTS")
    retrieval_strategy: str = Field("similarity", env="RETRIEVAL_STRATEGY")
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class RAGASConfig(BaseSettings):
    """RAGAS evaluation configuration."""
    
    faithfulness_threshold: float = Field(0.90, env="RAGAS_FAITHFULNESS_THRESHOLD")
    context_precision_threshold: float = Field(0.85, env="RAGAS_CONTEXT_PRECISION_THRESHOLD")
    context_recall_threshold: float = Field(0.80, env="RAGAS_CONTEXT_RECALL_THRESHOLD")
    answer_relevancy_threshold: float = Field(0.85, env="RAGAS_ANSWER_RELEVANCY_THRESHOLD")
    evaluation_enabled: bool = Field(True, env="RAGAS_EVALUATION_ENABLED")
    batch_size: int = Field(10, env="RAGAS_BATCH_SIZE")
    
    @validator("faithfulness_threshold", "context_precision_threshold", 
               "context_recall_threshold", "answer_relevancy_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


class SafetyConfig(BaseSettings):
    """Safety and filtering configuration."""
    
    filtering_enabled: bool = Field(True, env="SAFETY_FILTERING_ENABLED")
    harmful_content_detection: bool = Field(True, env="HARMFUL_CONTENT_DETECTION")
    response_validation_enabled: bool = Field(True, env="RESPONSE_VALIDATION_ENABLED")


class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration."""
    
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    metrics_collection_interval: int = Field(60, env="METRICS_COLLECTION_INTERVAL")


class PerformanceConfig(BaseSettings):
    """Performance configuration."""
    
    response_timeout: int = Field(30, env="RESPONSE_TIMEOUT")
    max_concurrent_requests: int = Field(100, env="MAX_CONCURRENT_REQUESTS")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    cache_ttl: int = Field(3600, env="CACHE_TTL")


class DataConfig(BaseSettings):
    """Data storage configuration."""
    
    documents_path: str = Field("./data/documents", env="DOCUMENTS_PATH")
    evaluation_data_path: str = Field("./data/evaluation", env="EVALUATION_DATA_PATH")
    upload_max_size: int = Field(10485760, env="UPLOAD_MAX_SIZE")  # 10MB


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    name: str = Field("Medical AI Assistant", env="APP_NAME")
    version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Sub-configurations
    openai: OpenAIConfig = OpenAIConfig()
    api: APIConfig = APIConfig()
    streamlit: StreamlitConfig = StreamlitConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    rag: RAGConfig = RAGConfig()
    ragas: RAGASConfig = RAGASConfig()
    safety: SafetyConfig = SafetyConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    performance: PerformanceConfig = PerformanceConfig()
    data: DataConfig = DataConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.data.documents_path,
            self.data.evaluation_data_path,
            self.vector_db.path,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = AppConfig()
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment variables."""
    global config
    config = AppConfig()
    return config 