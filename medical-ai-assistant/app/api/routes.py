"""
FastAPI routes for Medical AI Assistant.
Implements all API endpoints for medical queries, document management, and RAGAS evaluation.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import get_config
from app.core.document_processor import MedicalDocumentProcessor
from app.core.vector_store import MedicalVectorStore
from app.core.retriever import MedicalRetriever
from app.core.generator import MedicalResponseGenerator
from app.evaluation.ragas_evaluator import MedicalRAGASEvaluator
from app.api.models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    HealthCheckResponse, MetricsResponse, EvaluationRequest, EvaluationResponse,
    ErrorResponse, ConfigResponse, BatchQueryRequest, BatchQueryResponse,
    SourceDocument, DocumentMetadata, RAGASMetrics
)

logger = logging.getLogger(__name__)
config = get_config()

# Initialize components
document_processor = MedicalDocumentProcessor()
vector_store = MedicalVectorStore()
retriever = MedicalRetriever(vector_store)
generator = MedicalResponseGenerator()
ragas_evaluator = MedicalRAGASEvaluator()

# Create router
router = APIRouter()


def get_components():
    """Dependency to get initialized components."""
    return {
        "document_processor": document_processor,
        "vector_store": vector_store,
        "retriever": retriever,
        "generator": generator,
        "ragas_evaluator": ragas_evaluator
    }


@router.post("/query", response_model=QueryResponse)
async def query_medical_knowledge(
    request: QueryRequest,
    components: Dict = Depends(get_components)
) -> QueryResponse:
    """
    Process a medical query and return AI-generated response with RAGAS evaluation.
    
    This endpoint performs the complete RAG pipeline:
    1. Document retrieval based on query
    2. Response generation using retrieved context
    3. RAGAS evaluation of the response quality
    """
    start_time = time.time()
    
    logger.info(f"Processing medical query: {request.query[:100]}...")
    
    try:
        # Step 1: Retrieve relevant documents
        retrieval_result = components["retriever"].retrieve(
            query=request.query,
            k=request.max_documents,
            strategy=request.retrieval_strategy,
            api_key=request.openai_api_key
        )
        
        # Step 2: Generate response
        generation_result = components["generator"].generate_response(
            query=request.query,
            context_documents=retrieval_result.documents,
            api_key=request.openai_api_key
        )
        
        # Step 3: Prepare source documents for response
        sources = None
        if request.include_sources:
            sources = []
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
                source_doc = SourceDocument(
                    content=doc.page_content,
                    metadata=DocumentMetadata(**doc.metadata),
                    relevance_score=score
                )
                sources.append(source_doc)
        
        # Step 4: RAGAS evaluation
        ragas_metrics = None
        if request.evaluate_with_ragas:
            try:
                ragas_result = components["ragas_evaluator"].evaluate_rag_pipeline(
                    retrieval_result=retrieval_result,
                    generation_result=generation_result,
                    api_key=request.openai_api_key
                )
                
                ragas_metrics = RAGASMetrics(
                    faithfulness_score=ragas_result.faithfulness_score,
                    answer_relevancy_score=ragas_result.answer_relevancy_score,
                    context_precision_score=ragas_result.context_precision_score,
                    context_recall_score=ragas_result.context_recall_score,
                    overall_score=ragas_result.overall_score,
                    passes_thresholds=ragas_result.passes_thresholds,
                    evaluation_time=ragas_result.evaluation_time,
                    recommendations=ragas_result.recommendations
                )
                
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")
                # Continue without RAGAS metrics
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create response
        response = QueryResponse(
            query=request.query,
            answer=generation_result.response,
            sources=sources,
            ragas_metrics=ragas_metrics,
            retrieval_time=retrieval_result.retrieval_time,
            generation_time=generation_result.generation_time,
            total_time=total_time,
            model_used=generation_result.model_used,
            prompt_template=generation_result.prompt_template,
            safety_flags=generation_result.safety_flags
        )
        
        logger.info(f"Query processed successfully in {total_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_medical_knowledge(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components)
) -> BatchQueryResponse:
    """
    Process multiple medical queries in batch.
    
    Efficiently processes multiple queries and returns aggregated results.
    """
    start_time = time.time()
    
    logger.info(f"Processing batch of {len(request.queries)} queries")
    
    try:
        results = []
        successful_queries = 0
        failed_queries = 0
        
        for query in request.queries:
            try:
                # Create individual query request
                query_request = QueryRequest(
                    query=query,
                    openai_api_key=request.openai_api_key,
                    include_sources=request.include_sources,
                    evaluate_with_ragas=request.evaluate_with_ragas,
                    retrieval_strategy=request.retrieval_strategy
                )
                
                # Process query
                result = await query_medical_knowledge(query_request, components)
                results.append(result)
                successful_queries += 1
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                failed_queries += 1
                # Add error result
                error_result = QueryResponse(
                    query=query,
                    answer=f"Error processing query: {str(e)}",
                    sources=None,
                    ragas_metrics=None,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0,
                    model_used="error",
                    prompt_template="error",
                    safety_flags=["processing_error"]
                )
                results.append(error_result)
        
        # Calculate batch metrics
        total_time = time.time() - start_time
        avg_time_per_query = total_time / len(request.queries) if request.queries else 0
        
        batch_metrics = {
            "total_processing_time": total_time,
            "average_time_per_query": avg_time_per_query,
            "success_rate": successful_queries / len(request.queries) if request.queries else 0,
            "queries_per_second": len(request.queries) / total_time if total_time > 0 else 0
        }
        
        # Add RAGAS summary if evaluation was enabled
        if request.evaluate_with_ragas:
            ragas_results = [r.ragas_metrics for r in results if r.ragas_metrics]
            if ragas_results:
                batch_metrics["ragas_summary"] = {
                    "avg_faithfulness": sum(r.faithfulness_score for r in ragas_results) / len(ragas_results),
                    "avg_relevancy": sum(r.answer_relevancy_score for r in ragas_results) / len(ragas_results),
                    "avg_precision": sum(r.context_precision_score for r in ragas_results) / len(ragas_results),
                    "threshold_pass_rate": sum(1 for r in ragas_results if r.passes_thresholds) / len(ragas_results)
                }
        
        response = BatchQueryResponse(
            total_queries=len(request.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            results=results,
            batch_metrics=batch_metrics
        )
        
        logger.info(f"Batch processing completed: {successful_queries}/{len(request.queries)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = "medical",
    components: Dict = Depends(get_components)
) -> DocumentUploadResponse:
    """
    Upload and process a medical document.
    
    Accepts PDF files and processes them for the RAG pipeline.
    """
    start_time = time.time()
    
    logger.info(f"Uploading document: {file.filename}")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = f"{config.data.documents_path}/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        documents, metadata = components["document_processor"].process_document(file_path)
        
        # Add to vector store
        doc_ids = components["vector_store"].add_documents(documents)
        
        processing_time = time.time() - start_time
        
        response = DocumentUploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully",
            document_id=metadata.file_hash,
            chunks_created=metadata.total_chunks,
            total_tokens=metadata.total_tokens,
            processing_time=processing_time
        )
        
        logger.info(f"Document uploaded successfully: {metadata.total_chunks} chunks, {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return DocumentUploadResponse(
            success=False,
            message=f"Error uploading document: {str(e)}",
            processing_time=time.time() - start_time
        )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_with_ragas(
    request: EvaluationRequest,
    components: Dict = Depends(get_components)
) -> EvaluationResponse:
    """
    Evaluate query-answer pairs using RAGAS metrics.
    
    Performs batch RAGAS evaluation on provided data.
    """
    logger.info(f"Starting RAGAS evaluation of {len(request.queries)} samples")
    
    try:
        # Run batch evaluation
        results = await components["ragas_evaluator"].evaluate_batch(
            queries=request.queries,
            answers=request.answers,
            contexts=request.contexts,
            ground_truths=request.ground_truths,
            api_key=request.openai_api_key
        )
        
        # Convert results to API format
        individual_results = []
        for result in results:
            ragas_metrics = RAGASMetrics(
                faithfulness_score=result.faithfulness_score,
                answer_relevancy_score=result.answer_relevancy_score,
                context_precision_score=result.context_precision_score,
                context_recall_score=result.context_recall_score,
                overall_score=result.overall_score,
                passes_thresholds=result.passes_thresholds,
                evaluation_time=result.evaluation_time,
                recommendations=result.recommendations
            )
            individual_results.append(ragas_metrics)
        
        # Generate summary
        summary = components["ragas_evaluator"].get_evaluation_summary(results)
        
        response = EvaluationResponse(
            total_evaluations=len(results),
            average_scores=summary.get("average_scores", {}),
            pass_rates=summary.get("pass_rates", {}),
            individual_results=individual_results,
            summary=summary
        )
        
        logger.info(f"RAGAS evaluation completed: {len(results)} samples evaluated")
        return response
        
    except Exception as e:
        logger.error(f"Error in RAGAS evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    components: Dict = Depends(get_components)
) -> HealthCheckResponse:
    """
    Perform system health check.
    
    Checks the health of all system components.
    """
    try:
        # Check vector store
        vector_store_health = components["vector_store"].health_check()
        
        # Check retrieval system
        retrieval_stats = components["retriever"].get_retrieval_stats()
        
        # Check generator
        generation_stats = components["generator"].get_generation_stats()
        
        # Overall system status
        all_healthy = (
            vector_store_health.get("status") == "healthy"
        )
        
        components_status = {
            "vector_store": vector_store_health,
            "retrieval_system": {
                "status": "healthy",
                "stats": retrieval_stats
            },
            "response_generator": {
                "status": "healthy",
                "stats": generation_stats
            },
            "ragas_evaluator": {
                "status": "healthy",
                "thresholds": components["ragas_evaluator"].thresholds
            }
        }
        
        return HealthCheckResponse(
            status="healthy" if all_healthy else "degraded",
            components=components_status,
            version=config.version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            components={"error": str(e)},
            version=config.version
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    components: Dict = Depends(get_components)
) -> MetricsResponse:
    """
    Get system metrics and statistics.
    
    Returns comprehensive metrics about system performance and usage.
    """
    try:
        # Get vector store statistics
        vector_stats = components["vector_store"].get_collection_stats()
        
        # Get retrieval statistics
        retrieval_stats = components["retriever"].get_retrieval_stats()
        
        # Get generation statistics
        generation_stats = components["generator"].get_generation_stats()
        
        # System metrics
        system_metrics = {
            "uptime": "N/A",  # Would need to track startup time
            "memory_usage": "N/A",  # Would need psutil
            "cpu_usage": "N/A",  # Would need psutil
            "active_connections": "N/A"  # Would need connection tracking
        }
        
        # RAGAS metrics summary
        ragas_metrics = {
            "thresholds": components["ragas_evaluator"].thresholds,
            "evaluation_criteria": "medical_focused",
            "supported_metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        }
        
        return MetricsResponse(
            system_metrics=system_metrics,
            ragas_metrics=ragas_metrics,
            vector_store_stats=vector_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=ConfigResponse)
async def get_configuration() -> ConfigResponse:
    """
    Get current system configuration.
    
    Returns configuration settings for RAGAS thresholds, model settings, etc.
    """
    try:
        return ConfigResponse(
            ragas_thresholds={
                "faithfulness": config.ragas.faithfulness_threshold,
                "answer_relevancy": config.ragas.answer_relevancy_threshold,
                "context_precision": config.ragas.context_precision_threshold,
                "context_recall": config.ragas.context_recall_threshold
            },
            model_settings={
                "model": config.openai.model,
                "max_tokens": config.openai.max_tokens,
                "temperature": config.openai.temperature
            },
            retrieval_settings={
                "max_documents": config.rag.max_retrieval_documents,
                "chunk_size": config.rag.chunk_size,
                "chunk_overlap": config.rag.chunk_overlap,
                "similarity_threshold": config.vector_db.similarity_threshold
            },
            safety_settings={
                "filtering_enabled": config.safety.filtering_enabled,
                "harmful_content_detection": config.safety.harmful_content_detection,
                "response_validation": config.safety.response_validation_enabled
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    components: Dict = Depends(get_components)
) -> Dict[str, Any]:
    """
    Delete a document from the vector store.
    
    Removes all chunks associated with the document.
    """
    try:
        # Delete documents with matching file hash
        success = components["vector_store"].delete_by_metadata({"file_hash": document_id})
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/stats")
async def get_document_stats(
    components: Dict = Depends(get_components)
) -> Dict[str, Any]:
    """
    Get document collection statistics.
    
    Returns statistics about the document collection.
    """
    try:
        stats = components["vector_store"].get_collection_stats()
        return {"document_stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 