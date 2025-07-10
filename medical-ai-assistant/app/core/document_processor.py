"""
Document processor for medical documents.
Handles PDF ingestion, text chunking, and preprocessing for RAG pipeline.
"""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import PyPDF2
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    
    filename: str
    file_path: str
    file_size: int
    file_hash: str
    processed_at: datetime
    total_chunks: int
    total_tokens: int
    document_type: str = "medical"
    source: str = "local"


class MedicalDocumentProcessor:
    """
    Processes medical documents for RAG pipeline.
    
    Features:
    - PDF text extraction
    - Intelligent chunking for medical content
    - Token counting and optimization
    - Metadata extraction and storage
    - Duplicate detection
    """
    
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation endings
                "? ",    # Question endings
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character breaks
            ]
        )
        
        # Medical-specific section separators
        self.medical_separators = [
            "ABSTRACT",
            "INTRODUCTION",
            "METHODS",
            "RESULTS",
            "DISCUSSION",
            "CONCLUSION",
            "REFERENCES",
            "BACKGROUND",
            "OBJECTIVE",
            "DESIGN",
            "SETTING",
            "PARTICIPANTS",
            "INTERVENTIONS",
            "MAIN OUTCOME MEASURES",
            "CLINICAL IMPLICATIONS",
            "CONTRAINDICATIONS",
            "SIDE EFFECTS",
            "DOSAGE",
            "ADMINISTRATION"
        ]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using OpenAI's tokenizer."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split())  # Fallback to word count
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for duplicate detection."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                
                return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def _preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text for better chunking and retrieval.
        
        - Normalize whitespace
        - Fix common OCR errors
        - Preserve medical terminology
        - Standardize section headers
        """
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Fix common OCR errors in medical texts
        ocr_fixes = {
            "µ": "micro",
            "α": "alpha",
            "β": "beta",
            "γ": "gamma",
            "δ": "delta",
            "°": " degrees ",
            "±": " plus/minus ",
            "≥": " greater than or equal to ",
            "≤": " less than or equal to ",
            "→": " leads to ",
            "↑": " increases ",
            "↓": " decreases "
        }
        
        for old, new in ocr_fixes.items():
            text = text.replace(old, new)
        
        # Standardize section headers
        for separator in self.medical_separators:
            # Make section headers more prominent
            text = text.replace(f"{separator}:", f"\n\n{separator}:\n")
            text = text.replace(f"{separator.lower()}:", f"\n\n{separator}:\n")
        
        # Clean up extra whitespace
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        return text
    
    def _create_chunks_with_metadata(
        self, 
        text: str, 
        metadata: DocumentMetadata
    ) -> List[Document]:
        """Create document chunks with enhanced metadata."""
        # Preprocess text
        processed_text = self._preprocess_medical_text(text)
        
        # Create chunks
        chunks = self.text_splitter.split_text(processed_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Calculate chunk-specific metadata
            chunk_tokens = self._count_tokens(chunk)
            
            # Determine if chunk contains medical section
            chunk_sections = []
            for section in self.medical_separators:
                if section.lower() in chunk.lower():
                    chunk_sections.append(section)
            
            # Create document with metadata
            doc_metadata = {
                "filename": metadata.filename,
                "file_path": metadata.file_path,
                "file_hash": metadata.file_hash,
                "chunk_id": f"{metadata.file_hash}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_tokens": chunk_tokens,
                "document_type": metadata.document_type,
                "source": metadata.source,
                "processed_at": metadata.processed_at.isoformat(),
                "medical_sections": chunk_sections,
                "chunk_size": len(chunk)
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_document(self, file_path: str) -> Tuple[List[Document], DocumentMetadata]:
        """
        Process a single document and return chunks with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (document_chunks, document_metadata)
        """
        logger.info(f"Processing document: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Get file info
        file_stats = os.stat(file_path)
        file_hash = self._calculate_file_hash(file_path)
        
        # Extract text based on file type
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            raise ValueError(f"No text extracted from document: {file_path}")
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size=file_stats.st_size,
            file_hash=file_hash,
            processed_at=datetime.now(),
            total_chunks=0,  # Will be updated after chunking
            total_tokens=self._count_tokens(text),
            document_type="medical",
            source="local"
        )
        
        # Create chunks
        documents = self._create_chunks_with_metadata(text, metadata)
        
        # Update metadata with chunk count
        metadata.total_chunks = len(documents)
        
        logger.info(
            f"Successfully processed {metadata.filename}: "
            f"{metadata.total_chunks} chunks, {metadata.total_tokens} tokens"
        )
        
        return documents, metadata
    
    def process_directory(self, directory_path: str) -> List[Tuple[List[Document], DocumentMetadata]]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of (document_chunks, document_metadata) tuples
        """
        logger.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        results = []
        supported_extensions = [".pdf"]
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip non-files and unsupported extensions
            if not os.path.isfile(file_path):
                continue
            
            file_extension = Path(filename).suffix.lower()
            if file_extension not in supported_extensions:
                logger.warning(f"Skipping unsupported file: {filename}")
                continue
            
            try:
                documents, metadata = self.process_document(file_path)
                results.append((documents, metadata))
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        logger.info(f"Processed {len(results)} documents from {directory_path}")
        return results
    
    def get_processing_stats(self, results: List[Tuple[List[Document], DocumentMetadata]]) -> Dict[str, Any]:
        """Get processing statistics."""
        if not results:
            return {}
        
        total_documents = len(results)
        total_chunks = sum(len(docs) for docs, _ in results)
        total_tokens = sum(metadata.total_tokens for _, metadata in results)
        total_size = sum(metadata.file_size for _, metadata in results)
        
        avg_chunks_per_doc = total_chunks / total_documents if total_documents > 0 else 0
        avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0
        avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "total_size_bytes": total_size,
            "avg_chunks_per_document": round(avg_chunks_per_doc, 2),
            "avg_tokens_per_document": round(avg_tokens_per_doc, 2),
            "avg_tokens_per_chunk": round(avg_tokens_per_chunk, 2),
            "processing_timestamp": datetime.now().isoformat()
        } 