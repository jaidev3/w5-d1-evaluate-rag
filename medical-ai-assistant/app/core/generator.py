"""
OpenAI response generator for medical RAG pipeline.
Handles response generation with medical-specific prompts and safety measures.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import openai
from langchain.schema import Document

from app.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class GenerationResult:
    """Result of response generation."""
    
    response: str
    query: str
    context_documents: List[Document]
    generation_time: float
    token_usage: Dict[str, int]
    model_used: str
    prompt_template: str
    safety_flags: List[str]


class MedicalResponseGenerator:
    """
    Medical response generator using OpenAI GPT models.
    
    Features:
    - Medical-specific prompt templates
    - Context injection and formatting
    - Safety measures and disclaimers
    - Token usage tracking
    - Response validation
    """
    
    def __init__(self):
        # Note: OpenAI client will be initialized per request with provided API key
        
        # Medical prompt templates
        self.prompt_templates = {
            "general_medical": """You are a medical AI assistant designed to help healthcare professionals. 
You provide accurate, evidence-based information from medical literature and clinical guidelines.

IMPORTANT SAFETY GUIDELINES:
- Always base your responses on the provided medical literature
- Include appropriate medical disclaimers
- Recommend consulting healthcare professionals for diagnosis and treatment
- Do not provide personalized medical advice
- Indicate when information is insufficient or unclear

Query: {query}

Relevant Medical Literature:
{context}

Please provide a comprehensive, evidence-based response that addresses the query using the provided medical literature. Include:
1. Direct answer to the query
2. Supporting evidence from the literature
3. Important considerations or contraindications
4. Appropriate medical disclaimers

Response:""",
            
            "drug_information": """You are a medical AI assistant specializing in drug information for healthcare professionals.
Provide accurate, evidence-based information about medications, including dosages, contraindications, and interactions.

CRITICAL SAFETY REQUIREMENTS:
- Base all information on provided medical literature
- Include dosage information only if explicitly stated in sources
- Highlight contraindications and warnings prominently
- Recommend consulting prescribing information and healthcare professionals
- Do not provide personalized dosing recommendations

Query: {query}

Medical Literature on Drug Information:
{context}

Provide a structured response covering:
1. Drug mechanism and indications
2. Dosage and administration (if available in sources)
3. Contraindications and warnings
4. Common side effects and interactions
5. Special populations considerations
6. References to source materials

Response:""",
            
            "clinical_procedure": """You are a medical AI assistant providing information about clinical procedures and treatments.
Focus on evidence-based procedural information for healthcare professionals.

SAFETY AND ACCURACY GUIDELINES:
- Reference only information from provided medical literature
- Include procedural steps only if clearly documented in sources
- Highlight risks and complications
- Emphasize proper training and certification requirements
- Recommend following institutional protocols

Query: {query}

Clinical Literature:
{context}

Provide information covering:
1. Procedure overview and indications
2. Technique and methodology (if detailed in sources)
3. Risks and complications
4. Contraindications and precautions
5. Post-procedure care considerations
6. Training and certification requirements

Response:""",
            
            "diagnosis_support": """You are a medical AI assistant providing diagnostic support information for healthcare professionals.
Help with differential diagnosis and diagnostic considerations based on medical literature.

CRITICAL DIAGNOSTIC GUIDELINES:
- Provide differential diagnosis support only
- Never make definitive diagnoses
- Base all suggestions on provided medical literature
- Emphasize clinical correlation and professional judgment
- Recommend appropriate diagnostic tests and consultations

Query: {query}

Diagnostic Literature:
{context}

Provide diagnostic support covering:
1. Differential diagnosis considerations
2. Key diagnostic features from literature
3. Recommended diagnostic approaches
4. Important clinical correlations
5. When to seek specialist consultation
6. Limitations of provided information

Response:"""
        }
        
        # Safety phrases to detect potentially harmful content
        self.safety_flags = [
            "self-medication",
            "stop taking medication",
            "ignore doctor's advice",
            "home surgery",
            "self-diagnosis",
            "emergency situation"
        ]
    
    def _get_api_key(self, provided_key: Optional[str] = None) -> str:
        """Get API key from provided key or fallback to config."""
        if provided_key:
            return provided_key
        elif config.openai.api_key and config.openai.api_key != "test_key_for_development":
            return config.openai.api_key
        else:
            raise ValueError("OpenAI API key is required. Please provide it in the request or set it in the environment.")
    
    def _select_prompt_template(self, query: str, context_documents: List[Document]) -> str:
        """Select appropriate prompt template based on query and context."""
        query_lower = query.lower()
        
        # Analyze query intent
        if any(term in query_lower for term in ["drug", "medication", "dosage", "prescription", "contraindication"]):
            return "drug_information"
        elif any(term in query_lower for term in ["procedure", "surgery", "treatment", "therapy", "operation"]):
            return "clinical_procedure"
        elif any(term in query_lower for term in ["diagnosis", "symptom", "differential", "diagnostic"]):
            return "diagnosis_support"
        else:
            return "general_medical"
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format context documents for prompt injection."""
        if not documents:
            return "No relevant medical literature found."
        
        formatted_context = []
        
        for i, doc in enumerate(documents, 1):
            # Get document metadata
            filename = doc.metadata.get("filename", "Unknown source")
            sections = doc.metadata.get("medical_sections", [])
            
            # Format document section
            context_section = f"--- Source {i}: {filename} ---\n"
            
            if sections:
                context_section += f"Medical Sections: {', '.join(sections)}\n"
            
            context_section += f"Content:\n{doc.page_content}\n"
            
            formatted_context.append(context_section)
        
        return "\n".join(formatted_context)
    
    def _check_safety_flags(self, query: str, response: str) -> List[str]:
        """Check for safety flags in query and response."""
        flags = []
        
        # Check query for safety concerns
        query_lower = query.lower()
        for flag in self.safety_flags:
            if flag in query_lower:
                flags.append(f"query_safety_{flag.replace(' ', '_')}")
        
        # Check response for safety concerns
        response_lower = response.lower()
        for flag in self.safety_flags:
            if flag in response_lower:
                flags.append(f"response_safety_{flag.replace(' ', '_')}")
        
        return flags
    
    def _add_safety_disclaimer(self, response: str) -> str:
        """Add medical disclaimer to response."""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is provided for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare professionals for diagnosis and treatment decisions."
        
        return response + disclaimer
    
    def generate_response(
        self,
        query: str,
        context_documents: List[Document],
        template_override: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate medical response using OpenAI API.
        
        Args:
            query: Medical query
            context_documents: Retrieved documents for context
            template_override: Optional template override
            api_key: OpenAI API key (required)
            
        Returns:
            GenerationResult with response and metadata
        """
        start_time = time.time()
        
        logger.info(f"Generating response for query: {query[:100]}...")
        
        try:
            # Get API key
            openai_api_key = self._get_api_key(api_key)
            
            # Initialize OpenAI client with the provided key
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Select prompt template
            template_name = template_override or self._select_prompt_template(query, context_documents)
            prompt_template = self.prompt_templates[template_name]
            
            # Format context
            context_text = self._format_context(context_documents)
            
            # Create final prompt
            final_prompt = prompt_template.format(
                query=query,
                context=context_text
            )
            
            # Generate response
            response = client.chat.completions.create(
                model=config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant for healthcare professionals."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=config.openai.max_tokens,
                temperature=config.openai.temperature,
                timeout=30
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Add safety disclaimer
            response_content = self._add_safety_disclaimer(response_content)
            
            # Check for safety flags
            safety_flags = self._check_safety_flags(query, response_content)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Create result
            result = GenerationResult(
                response=response_content,
                query=query,
                context_documents=context_documents,
                generation_time=generation_time,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model_used=config.openai.model,
                prompt_template=template_name,
                safety_flags=safety_flags
            )
            
            logger.info(f"Response generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def validate_response(self, generation_result: GenerationResult) -> Dict[str, Any]:
        """
        Validate generated response for medical safety and quality.
        
        Args:
            generation_result: Result from generate_response
            
        Returns:
            Validation results with scores and recommendations
        """
        response = generation_result.response
        safety_flags = generation_result.safety_flags
        
        validation_results = {
            "is_safe": len(safety_flags) == 0,
            "safety_flags": safety_flags,
            "has_disclaimer": "⚠️ **Medical Disclaimer**: This information is provided for educational purposes only and should not replace professional medical advice." in response,
            "response_length": len(response),
            "token_efficiency": generation_result.token_usage["completion_tokens"] / len(response) if response else 0,
            "context_utilization": len(generation_result.context_documents) > 0,
            "recommendations": []
        }
        
        # Add recommendations based on validation
        if not validation_results["is_safe"]:
            validation_results["recommendations"].append("Review response for safety concerns")
        
        if not validation_results["has_disclaimer"]:
            validation_results["recommendations"].append("Add medical disclaimer")
        
        if validation_results["response_length"] < 100:
            validation_results["recommendations"].append("Response may be too brief")
        
        if not validation_results["context_utilization"]:
            validation_results["recommendations"].append("No context documents used - consider improving retrieval")
        
        return validation_results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics and model information."""
        return {
            "model": config.openai.model,
            "max_tokens": config.openai.max_tokens,
            "temperature": config.openai.temperature,
            "available_templates": list(self.prompt_templates.keys()),
            "safety_checks_enabled": True,
            "disclaimer_added": True,
            "supported_query_types": [
                "general_medical",
                "drug_information", 
                "clinical_procedure",
                "diagnosis_support"
            ]
        } 