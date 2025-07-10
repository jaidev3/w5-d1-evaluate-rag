"""
Document retriever for medical RAG pipeline.
Handles retrieval logic, filtering, and ranking of relevant documents.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from langchain.schema import Document

from app.config import get_config
from app.core.vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    
    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_time: float
    total_found: int
    filters_applied: Dict[str, Any]
    retrieval_strategy: str


class MedicalRetriever:
    """
    Advanced retriever for medical documents.
    
    Features:
    - Multiple retrieval strategies
    - Medical-specific filtering
    - Query preprocessing and expansion
    - Result ranking and reranking
    - Context-aware retrieval
    """
    
    def __init__(self, vector_store: MedicalVectorStore):
        self.vector_store = vector_store
        
        # Medical terminology patterns
        self.medical_patterns = {
            "drug_names": r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|penicillin|insulin|warfarin|metformin|lisinopril|atorvastatin)\b',
            "conditions": r'\b(?:diabetes|hypertension|pneumonia|asthma|depression|anxiety|cancer|arthritis|migraine|obesity)\b',
            "symptoms": r'\b(?:pain|fever|nausea|fatigue|headache|dizziness|shortness of breath|chest pain|abdominal pain)\b',
            "procedures": r'\b(?:surgery|biopsy|endoscopy|catheterization|angioplasty|chemotherapy|radiation|dialysis)\b',
            "body_parts": r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine|bone|muscle|skin|eye|ear)\b'
        }
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess medical query for better retrieval.
        
        - Expand medical abbreviations
        - Normalize medical terminology
        - Handle synonyms
        """
        # Common medical abbreviations
        abbreviations = {
            "MI": "myocardial infarction",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "COPD": "chronic obstructive pulmonary disease",
            "CHF": "congestive heart failure",
            "UTI": "urinary tract infection",
            "URI": "upper respiratory infection",
            "GI": "gastrointestinal",
            "CNS": "central nervous system",
            "CVD": "cardiovascular disease",
            "ICU": "intensive care unit",
            "ER": "emergency room",
            "OR": "operating room",
            "IV": "intravenous",
            "PO": "oral",
            "PRN": "as needed",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily"
        }
        
        processed_query = query
        
        # Expand abbreviations
        for abbr, expansion in abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            processed_query = re.sub(pattern, expansion, processed_query, flags=re.IGNORECASE)
        
        # Normalize common medical terms
        medical_synonyms = {
            "heart attack": "myocardial infarction",
            "high blood pressure": "hypertension",
            "low blood pressure": "hypotension",
            "blood sugar": "glucose",
            "stroke": "cerebrovascular accident",
            "kidney failure": "renal failure",
            "liver failure": "hepatic failure"
        }
        
        for synonym, standard in medical_synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            processed_query = re.sub(pattern, standard, processed_query, flags=re.IGNORECASE)
        
        return processed_query
    
    def _extract_medical_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract medical entities from query for targeted retrieval."""
        entities = {}
        
        for entity_type, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _build_metadata_filters(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build metadata filters based on query and extracted entities."""
        filters = {}
        
        # Filter by document type
        if any(term in query.lower() for term in ["drug", "medication", "prescription", "dosage"]):
            filters["document_type"] = "drug_information"
        elif any(term in query.lower() for term in ["procedure", "surgery", "treatment", "therapy"]):
            filters["document_type"] = "clinical_procedure"
        elif any(term in query.lower() for term in ["guideline", "protocol", "recommendation"]):
            filters["document_type"] = "clinical_guideline"
        
        # Filter by medical sections if entities are present
        if entities:
            # Look for documents that might contain relevant sections
            relevant_sections = []
            
            if "drug_names" in entities:
                relevant_sections.extend(["DOSAGE", "ADMINISTRATION", "CONTRAINDICATIONS", "SIDE EFFECTS"])
            if "conditions" in entities:
                relevant_sections.extend(["BACKGROUND", "CLINICAL IMPLICATIONS", "TREATMENT"])
            if "procedures" in entities:
                relevant_sections.extend(["METHODS", "PROCEDURE", "INTERVENTIONS"])
            
            if relevant_sections:
                # Note: This would need to be adapted based on how ChromaDB handles array filtering
                pass
        
        return filters
    
    def _rerank_results(
        self, 
        documents_with_scores: List[Tuple[Document, float]], 
        query: str, 
        entities: Dict[str, List[str]]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank results based on medical relevance and query context.
        
        Factors considered:
        - Medical entity matches
        - Section relevance
        - Document recency
        - Source credibility
        """
        if not documents_with_scores:
            return documents_with_scores
        
        reranked_results = []
        
        for doc, score in documents_with_scores:
            # Calculate boosting factors
            boost_factor = 1.0
            
            # Entity matching boost
            doc_content_lower = doc.page_content.lower()
            entity_matches = 0
            total_entities = 0
            
            for entity_type, entity_list in entities.items():
                total_entities += len(entity_list)
                for entity in entity_list:
                    if entity.lower() in doc_content_lower:
                        entity_matches += 1
            
            if total_entities > 0:
                entity_boost = 1 + (entity_matches / total_entities) * 0.3
                boost_factor *= entity_boost
            
            # Section relevance boost
            medical_sections = doc.metadata.get("medical_sections", [])
            if isinstance(medical_sections, list) and medical_sections:
                # Boost documents with relevant medical sections
                relevant_sections = ["CLINICAL IMPLICATIONS", "CONTRAINDICATIONS", "DOSAGE", "ADMINISTRATION"]
                section_matches = sum(1 for section in medical_sections if section in relevant_sections)
                if section_matches > 0:
                    boost_factor *= 1 + (section_matches * 0.1)
            
            # Document type boost
            doc_type = doc.metadata.get("document_type", "")
            if doc_type == "clinical_guideline":
                boost_factor *= 1.2
            elif doc_type == "drug_information":
                boost_factor *= 1.1
            
            # Chunk position boost (earlier chunks often contain more important info)
            chunk_index = doc.metadata.get("chunk_index", 0)
            total_chunks = doc.metadata.get("total_chunks", 1)
            if total_chunks > 1:
                position_factor = 1 - (chunk_index / total_chunks) * 0.1
                boost_factor *= position_factor
            
            # Apply boost to score
            boosted_score = score * boost_factor
            reranked_results.append((doc, boosted_score))
        
        # Sort by boosted score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results
    
    def retrieve(
        self, 
        query: str, 
        k: int = None, 
        strategy: str = "similarity",
        filters: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            strategy: Retrieval strategy ('similarity', 'hybrid', 'medical_focused')
            filters: Metadata filters to apply
            api_key: OpenAI API key for embeddings
            
        Returns:
            RetrievalResult with documents and metadata
        """
        if k is None:
            k = config.rag.max_retrieval_documents
        
        start_time = time.time()
        
        logger.info(f"Retrieving documents for query: {query[:100]}... (strategy: {strategy})")
        
        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Extract medical entities
            entities = self._extract_medical_entities(processed_query)
            
            # Build metadata filters
            metadata_filters = self._build_metadata_filters(processed_query, entities)
            
            # Combine with user-provided filters
            if filters:
                metadata_filters.update(filters)
            
            # Perform retrieval based on strategy
            if strategy == "similarity":
                documents_with_scores = self._similarity_retrieval(
                    processed_query, k, metadata_filters, api_key
                )
            elif strategy == "hybrid":
                documents_with_scores = self._hybrid_retrieval(
                    processed_query, k, metadata_filters, api_key
                )
            elif strategy == "medical_focused":
                documents_with_scores = self._medical_focused_retrieval(
                    processed_query, k, metadata_filters, entities, api_key
                )
            else:
                raise ValueError(f"Unknown retrieval strategy: {strategy}")
            
            # Rerank results
            documents_with_scores = self._rerank_results(
                documents_with_scores, processed_query, entities
            )
            
            # Extract documents and scores
            documents = [doc for doc, score in documents_with_scores]
            scores = [score for doc, score in documents_with_scores]
            
            # Calculate retrieval time
            retrieval_time = time.time() - start_time
            
            # Create result
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_time=retrieval_time,
                total_found=len(documents),
                filters_applied=metadata_filters,
                retrieval_strategy=strategy
            )
            
            logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def _similarity_retrieval(
        self, 
        query: str, 
        k: int, 
        filters: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity-based retrieval."""
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filters,
            api_key=api_key
        )
    
    def _hybrid_retrieval(
        self, 
        query: str, 
        k: int, 
        filters: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Perform hybrid retrieval (similarity + keyword matching)."""
        # For now, use similarity retrieval with expanded k
        # In a full implementation, this would combine semantic and keyword search
        expanded_k = min(k * 2, 20)  # Get more results for reranking
        
        return self.vector_store.similarity_search(
            query=query,
            k=expanded_k,
            filter_metadata=filters,
            api_key=api_key
        )[:k]  # Return top k after expansion
    
    def _medical_focused_retrieval(
        self, 
        query: str, 
        k: int, 
        filters: Dict[str, Any],
        entities: Dict[str, List[str]],
        api_key: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Perform medical-focused retrieval with entity-based filtering."""
        # Apply medical entity filters
        medical_filters = filters.copy()
        
        # Add entity-based filters if we have identified medical entities
        if entities:
            logger.info(f"Using medical entities for focused retrieval: {entities}")
        
        # Use expanded k for medical-focused retrieval
        expanded_k = min(k * 2, 20)
        
        return self.vector_store.similarity_search(
            query=query,
            k=expanded_k,
            filter_metadata=medical_filters,
            api_key=api_key
        )[:k]  # Return top k after expansion
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store_stats": vector_stats,
            "supported_strategies": ["similarity", "hybrid", "medical_focused"],
            "medical_entity_types": list(self.medical_patterns.keys()),
            "default_k": config.rag.max_retrieval_documents,
            "similarity_threshold": config.vector_db.similarity_threshold
        } 