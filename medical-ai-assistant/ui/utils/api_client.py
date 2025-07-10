"""
API client for Streamlit UI to communicate with FastAPI backend.
Handles all API calls and response processing.
"""

import logging
import requests
import streamlit as st
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalAIClient:
    """
    Client for communicating with the Medical AI Assistant API.
    
    Handles all API calls from the Streamlit frontend to the FastAPI backend.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.api_base}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise Exception("Invalid response from API")
    
    def query_medical_knowledge(
        self,
        query: str,
        openai_api_key: Optional[str] = None,
        include_sources: bool = True,
        evaluate_with_ragas: bool = True,
        retrieval_strategy: str = "similarity",
        max_documents: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submit a medical query to the API.
        
        Args:
            query: Medical question
            openai_api_key: OpenAI API key for processing
            include_sources: Include source documents in response
            evaluate_with_ragas: Evaluate response with RAGAS metrics
            retrieval_strategy: Retrieval strategy to use
            max_documents: Maximum number of documents to retrieve
            
        Returns:
            Query response with answer, sources, and RAGAS metrics
        """
        payload = {
            "query": query,
            "include_sources": include_sources,
            "evaluate_with_ragas": evaluate_with_ragas,
            "retrieval_strategy": retrieval_strategy
        }
        
        if openai_api_key:
            payload["openai_api_key"] = openai_api_key
        
        if max_documents:
            payload["max_documents"] = max_documents
        
        return self._make_request("POST", "/query", json=payload)
    
    def batch_query(
        self,
        queries: List[str],
        openai_api_key: Optional[str] = None,
        include_sources: bool = True,
        evaluate_with_ragas: bool = True,
        retrieval_strategy: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Submit multiple queries in batch.
        
        Args:
            queries: List of medical questions
            openai_api_key: OpenAI API key for processing
            include_sources: Include source documents
            evaluate_with_ragas: Evaluate with RAGAS
            retrieval_strategy: Retrieval strategy
            
        Returns:
            Batch query response
        """
        payload = {
            "queries": queries,
            "include_sources": include_sources,
            "evaluate_with_ragas": evaluate_with_ragas,
            "retrieval_strategy": retrieval_strategy
        }
        
        if openai_api_key:
            payload["openai_api_key"] = openai_api_key
        
        return self._make_request("POST", "/query/batch", json=payload)
    
    def upload_document(self, file_content: bytes, filename: str, document_type: str = "medical") -> Dict[str, Any]:
        """
        Upload a document to the system.
        
        Args:
            file_content: Document content as bytes
            filename: Name of the file
            document_type: Type of medical document
            
        Returns:
            Upload response
        """
        files = {
            "file": (filename, file_content, "application/pdf")
        }
        
        data = {
            "document_type": document_type
        }
        
        # Remove Content-Type header for file upload
        headers = {k: v for k, v in self.session.headers.items() if k != "Content-Type"}
        
        try:
            response = requests.post(
                f"{self.api_base}/documents/upload",
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Document upload failed: {e}")
            raise Exception(f"Document upload failed: {str(e)}")
    
    def evaluate_with_ragas(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate query-answer pairs with RAGAS.
        
        Args:
            queries: List of queries
            answers: List of answers
            contexts: List of context lists
            ground_truths: Optional ground truth answers
            openai_api_key: OpenAI API key for RAGAS evaluation
            
        Returns:
            RAGAS evaluation results
        """
        payload = {
            "queries": queries,
            "answers": answers,
            "contexts": contexts
        }
        
        if ground_truths:
            payload["ground_truths"] = ground_truths
        
        if openai_api_key:
            payload["openai_api_key"] = openai_api_key
        
        return self._make_request("POST", "/evaluate", json=payload)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Health check response
        """
        return self._make_request("GET", "/health")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Returns:
            Metrics response
        """
        return self._make_request("GET", "/metrics")
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get system configuration.
        
        Returns:
            Configuration response
        """
        return self._make_request("GET", "/config")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get document collection statistics.
        
        Returns:
            Document statistics
        """
        return self._make_request("GET", "/documents/stats")
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the system.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Deletion response
        """
        return self._make_request("DELETE", f"/documents/{document_id}")
    
    def check_connection(self) -> bool:
        """
        Check if the API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# Streamlit session state management
def get_api_client() -> MedicalAIClient:
    """
    Get or create API client from Streamlit session state.
    
    Returns:
        MedicalAIClient instance
    """
    if "api_client" not in st.session_state:
        # Get API URL from environment or use default
        api_url = st.secrets.get("API_URL", "http://localhost:8000")
        st.session_state.api_client = MedicalAIClient(api_url)
    
    return st.session_state.api_client


def display_api_error(error: Exception):
    """
    Display API error in Streamlit UI.
    
    Args:
        error: Exception that occurred
    """
    st.error(f"🚨 API Error: {str(error)}")
    
    with st.expander("Error Details"):
        st.code(str(error))
        st.info("Please check if the FastAPI backend is running on the correct port.")


def format_response_time(seconds: float) -> str:
    """
    Format response time for display.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def format_ragas_score(score: float) -> str:
    """
    Format RAGAS score for display with color coding.
    
    Args:
        score: RAGAS score (0-1)
        
    Returns:
        Formatted score string
    """
    percentage = score * 100
    
    if score >= 0.9:
        return f"🟢 {percentage:.1f}%"
    elif score >= 0.8:
        return f"🟡 {percentage:.1f}%"
    else:
        return f"🔴 {percentage:.1f}%"


def cache_api_response(key: str, response: Dict[str, Any], ttl: int = 300):
    """
    Cache API response in session state.
    
    Args:
        key: Cache key
        response: Response to cache
        ttl: Time to live in seconds
    """
    if "api_cache" not in st.session_state:
        st.session_state.api_cache = {}
    
    st.session_state.api_cache[key] = {
        "response": response,
        "timestamp": datetime.now(),
        "ttl": ttl
    }


def get_cached_response(key: str) -> Optional[Dict[str, Any]]:
    """
    Get cached API response if still valid.
    
    Args:
        key: Cache key
        
    Returns:
        Cached response or None if expired/not found
    """
    if "api_cache" not in st.session_state:
        return None
    
    cached = st.session_state.api_cache.get(key)
    if not cached:
        return None
    
    # Check if cache is still valid
    elapsed = (datetime.now() - cached["timestamp"]).total_seconds()
    if elapsed > cached["ttl"]:
        del st.session_state.api_cache[key]
        return None
    
    return cached["response"] 