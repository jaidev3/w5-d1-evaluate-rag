"""
Vector store implementation using ChromaDB for medical document embeddings.
Handles document storage, retrieval, and similarity search.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
from langchain.schema import Document

from app.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class MedicalVectorStore:
    """
    Vector store for medical documents using ChromaDB.
    
    Features:
    - Document embedding and storage
    - Similarity search with metadata filtering
    - Batch operations for efficient processing
    - Persistence and recovery
    - Duplicate detection and handling
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.current_api_key = None
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence."""
        try:
            # Try to create ChromaDB client with persistence
            try:
                self.client = chromadb.PersistentClient(
                    path=config.vector_db.path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("ChromaDB persistent client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize persistent client: {e}")
                logger.info("Falling back to in-memory client")
                self.client = chromadb.EphemeralClient()
                logger.info("ChromaDB in-memory client initialized successfully")
            
            # Initialize OpenAI embedding function only if API key is available
            if config.openai.api_key and config.openai.api_key != "test_key_for_development":
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.openai.api_key,
                    model_name=config.openai.embedding_model
                )
                self.current_api_key = config.openai.api_key
                logger.info("OpenAI embedding function initialized with config API key")
            else:
                logger.info("No OpenAI API key provided in config - embedding function will be created dynamically")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def _get_or_create_embedding_function(self, api_key: Optional[str] = None) -> embedding_functions.OpenAIEmbeddingFunction:
        """Get or create OpenAI embedding function with the provided API key."""
        if api_key and api_key != self.current_api_key:
            # Create new embedding function with provided API key
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=config.openai.embedding_model
            )
            self.current_api_key = api_key
            logger.info("Created new OpenAI embedding function with provided API key")
        elif self.embedding_function is None:
            if api_key:
                # Create embedding function with provided API key
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=config.openai.embedding_model
                )
                self.current_api_key = api_key
                logger.info("Created OpenAI embedding function with provided API key")
            else:
                raise ValueError("OpenAI API key is required for embedding operations")
        
        return self.embedding_function
    
    def _initialize_collection(self):
        """Initialize or get the medical documents collection."""
        try:
            # If we have an embedding function, use it; otherwise, initialize without it
            if self.embedding_function:
                # Try to get existing collection
                try:
                    self.collection = self.client.get_collection(
                        name=config.vector_db.collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Retrieved existing collection: {config.vector_db.collection_name}")
                    
                except Exception:
                    # Create new collection if it doesn't exist
                    self.collection = self.client.create_collection(
                        name=config.vector_db.collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": "Medical documents for RAG pipeline"}
                    )
                    logger.info(f"Created new collection: {config.vector_db.collection_name}")
            else:
                # Try to get existing collection without embedding function
                try:
                    self.collection = self.client.get_collection(
                        name=config.vector_db.collection_name
                    )
                    logger.info(f"Retrieved existing collection: {config.vector_db.collection_name}")
                except Exception:
                    # Collection will be created when embedding function is available
                    logger.info("Collection will be created when embedding function is available")
                    
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def _ensure_collection_with_embedding(self, api_key: Optional[str] = None):
        """Ensure collection exists with proper embedding function."""
        if self.collection is None:
            embedding_func = self._get_or_create_embedding_function(api_key)
            try:
                self.collection = self.client.get_collection(
                    name=config.vector_db.collection_name,
                    embedding_function=embedding_func
                )
                logger.info(f"Retrieved existing collection: {config.vector_db.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=config.vector_db.collection_name,
                    embedding_function=embedding_func,
                    metadata={"description": "Medical documents for RAG pipeline"}
                )
                logger.info(f"Created new collection: {config.vector_db.collection_name}")
    
    def add_documents(self, documents: List[Document], api_key: Optional[str] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            api_key: OpenAI API key for embeddings
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            # Ensure we have a collection with embedding function
            self._ensure_collection_with_embedding(api_key)
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Use chunk_id as document ID if available, otherwise generate one
                doc_id = doc.metadata.get("chunk_id")
                if not doc_id:
                    # Generate ID from hash of content and metadata
                    content_hash = hash(doc.page_content + str(doc.metadata))
                    doc_id = f"doc_{abs(content_hash)}"
                
                ids.append(doc_id)
                texts.append(doc.page_content)
                
                # Ensure metadata is JSON serializable
                metadata = self._serialize_metadata(doc.metadata)
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata to ensure ChromaDB compatibility."""
        serialized = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif isinstance(value, list):
                # Convert list to JSON string
                serialized[key] = json.dumps(value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                serialized[key] = json.dumps(value)
            else:
                # Convert other types to string
                serialized[key] = str(value)
        
        return serialized
    
    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize metadata from ChromaDB format."""
        deserialized = {}
        
        for key, value in metadata.items():
            if key in ["medical_sections"] and isinstance(value, str):
                try:
                    deserialized[key] = json.loads(value)
                except json.JSONDecodeError:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        
        return deserialized
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search for documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters to apply
            api_key: OpenAI API key for embeddings
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if k is None:
            k = config.rag.max_retrieval_documents
        
        logger.info(f"Performing similarity search for query: {query[:100]}...")
        
        try:
            # Ensure we have a collection with embedding function
            self._ensure_collection_with_embedding(api_key)
            
            # Prepare query parameters
            query_params = {
                "query_texts": [query],
                "n_results": k
            }
            
            # Add metadata filters if provided
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            # Perform search
            results = self.collection.query(**query_params)
            
            # Process results
            documents_with_scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc_content, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    # Deserialize metadata
                    deserialized_metadata = self._deserialize_metadata(metadata)
                    
                    # Create Document object
                    document = Document(
                        page_content=doc_content,
                        metadata=deserialized_metadata
                    )
                    
                    documents_with_scores.append((document, similarity_score))
            
            logger.info(f"Found {len(documents_with_scores)} similar documents")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results["documents"] and results["documents"][0]:
                metadata = self._deserialize_metadata(results["metadatas"][0])
                return Document(
                    page_content=results["documents"][0],
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> bool:
        """Delete documents by metadata filter."""
        try:
            self.collection.delete(where=metadata_filter)
            logger.info(f"Deleted documents matching filter: {metadata_filter}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents by metadata: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get sample of documents for analysis
            sample_size = min(100, count)
            sample_results = self.collection.get(limit=sample_size)
            
            # Analyze metadata
            document_types = {}
            sources = {}
            total_tokens = 0
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    # Document types
                    doc_type = metadata.get("document_type", "unknown")
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
                    
                    # Sources
                    source = metadata.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
                    
                    # Tokens
                    tokens = metadata.get("chunk_tokens", 0)
                    if isinstance(tokens, (int, float)):
                        total_tokens += tokens
            
            return {
                "total_documents": count,
                "document_types": document_types,
                "sources": sources,
                "estimated_total_tokens": total_tokens * (count / sample_size) if sample_size > 0 else 0,
                "collection_name": config.vector_db.collection_name,
                "embedding_model": config.openai.embedding_model,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)."""
        try:
            # Delete the collection
            self.client.delete_collection(name=config.vector_db.collection_name)
            
            # Recreate the collection
            self._initialize_collection()
            
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            # Check client connection
            collections = self.client.list_collections()
            
            # Check collection accessibility
            count = self.collection.count()
            
            # Test embedding function
            test_embedding = self.embedding_function(["test query"])
            
            return {
                "status": "healthy",
                "client_connected": True,
                "collection_accessible": True,
                "document_count": count,
                "embedding_function_working": len(test_embedding) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 