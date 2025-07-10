"""
RAGAS (Retrieval-Augmented Generation Assessment) evaluator for medical RAG pipeline.
Implements all core RAGAS metrics with medical-specific adaptations.
"""

import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain.schema import Document

from app.config import get_config
from app.core.retriever import RetrievalResult
from app.core.generator import GenerationResult

logger = logging.getLogger(__name__)
config = get_config()


@contextmanager
def temporary_openai_key(api_key: Optional[str] = None):
    """Context manager to temporarily set OpenAI API key."""
    if api_key:
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            yield
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
    else:
        yield


@dataclass
class RAGASEvaluationResult:
    """Result of RAGAS evaluation."""
    
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    
    # Core RAGAS metrics
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: Optional[float]
    
    # Overall assessment
    overall_score: float
    passes_thresholds: bool
    
    # Metadata
    evaluation_time: float
    model_used: str
    evaluation_timestamp: datetime
    
    # Detailed feedback
    metric_details: Dict[str, Any]
    recommendations: List[str]


class MedicalRAGASEvaluator:
    """
    RAGAS evaluator specifically designed for medical RAG systems.
    
    Features:
    - All core RAGAS metrics implementation
    - Medical-specific evaluation criteria
    - Batch and real-time evaluation
    - Quality threshold enforcement
    - Detailed feedback and recommendations
    """
    
    def __init__(self):
        self.thresholds = {
            "faithfulness": config.ragas.faithfulness_threshold,
            "answer_relevancy": config.ragas.answer_relevancy_threshold,
            "context_precision": config.ragas.context_precision_threshold,
            "context_recall": config.ragas.context_recall_threshold
        }
        
        # Medical-specific evaluation criteria
        self.medical_quality_indicators = {
            "safety_phrases": [
                "consult healthcare professional",
                "seek medical attention",
                "medical professional",
                "healthcare provider",
                "doctor",
                "physician"
            ],
            "evidence_phrases": [
                "according to",
                "based on",
                "studies show",
                "research indicates",
                "evidence suggests",
                "clinical trials"
            ],
            "uncertainty_phrases": [
                "may",
                "might",
                "could",
                "possibly",
                "potentially",
                "appears to"
            ]
        }
    
    def _prepare_evaluation_dataset(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """Prepare dataset for RAGAS evaluation."""
        
        data = {
            "question": queries,
            "answer": answers,
            "contexts": contexts
        }
        
        if ground_truths:
            data["ground_truths"] = ground_truths
        
        return Dataset.from_dict(data)
    
    def _calculate_medical_quality_score(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Calculate medical-specific quality indicators."""
        answer_lower = answer.lower()
        
        # Safety indicators
        safety_count = sum(1 for phrase in self.medical_quality_indicators["safety_phrases"] 
                          if phrase in answer_lower)
        safety_score = min(safety_count / 2, 1.0)  # Normalize to 0-1
        
        # Evidence grounding
        evidence_count = sum(1 for phrase in self.medical_quality_indicators["evidence_phrases"] 
                           if phrase in answer_lower)
        evidence_score = min(evidence_count / 2, 1.0)
        
        # Appropriate uncertainty
        uncertainty_count = sum(1 for phrase in self.medical_quality_indicators["uncertainty_phrases"] 
                              if phrase in answer_lower)
        uncertainty_score = min(uncertainty_count / 3, 1.0)
        
        # Context utilization
        context_text = " ".join(contexts).lower()
        context_overlap = 0
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        
        if context_words:
            context_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        
        return {
            "safety_score": safety_score,
            "evidence_score": evidence_score,
            "uncertainty_score": uncertainty_score,
            "context_utilization": context_overlap,
            "overall_medical_quality": (safety_score + evidence_score + uncertainty_score + context_overlap) / 4
        }
    
    def _generate_recommendations(self, scores: Dict[str, float], medical_quality: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation scores."""
        recommendations = []
        
        # Faithfulness recommendations
        if scores.get("faithfulness", 0) < self.thresholds["faithfulness"]:
            recommendations.append(
                f"Improve faithfulness (current: {scores.get('faithfulness', 0):.3f}, "
                f"threshold: {self.thresholds['faithfulness']:.3f}). "
                "Ensure responses are grounded in provided context."
            )
        
        # Answer relevancy recommendations
        if scores.get("answer_relevancy", 0) < self.thresholds["answer_relevancy"]:
            recommendations.append(
                f"Improve answer relevancy (current: {scores.get('answer_relevancy', 0):.3f}, "
                f"threshold: {self.thresholds['answer_relevancy']:.3f}). "
                "Make sure answers directly address the query."
            )
        
        # Context precision recommendations
        if scores.get("context_precision", 0) < self.thresholds["context_precision"]:
            recommendations.append(
                f"Improve context precision (current: {scores.get('context_precision', 0):.3f}, "
                f"threshold: {self.thresholds['context_precision']:.3f}). "
                "Improve document retrieval to get more relevant contexts."
            )
        
        # Context recall recommendations
        if scores.get("context_recall") is not None and scores.get("context_recall", 0) < self.thresholds["context_recall"]:
            recommendations.append(
                f"Improve context recall (current: {scores.get('context_recall', 0):.3f}, "
                f"threshold: {self.thresholds['context_recall']:.3f}). "
                "Ensure all relevant information is retrieved."
            )
        
        # Medical quality recommendations
        if medical_quality["safety_score"] < 0.5:
            recommendations.append(
                "Add more safety disclaimers and recommendations to consult healthcare professionals."
            )
        
        if medical_quality["evidence_score"] < 0.5:
            recommendations.append(
                "Better ground responses in medical evidence and research."
            )
        
        if medical_quality["context_utilization"] < 0.3:
            recommendations.append(
                "Improve utilization of retrieved context in response generation."
            )
        
        return recommendations
    
    async def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> RAGASEvaluationResult:
        """
        Evaluate a single query-answer pair using RAGAS metrics.
        
        Args:
            query: The user query
            answer: The generated answer
            contexts: List of context strings used for generation
            ground_truth: Optional ground truth answer for context recall
            api_key: OpenAI API key for RAGAS evaluation
            
        Returns:
            RAGASEvaluationResult with all metrics and recommendations
        """
        start_time = datetime.now()
        
        logger.info(f"Evaluating single query: {query[:100]}...")
        
        try:
            # Prepare dataset
            dataset = self._prepare_evaluation_dataset(
                queries=[query],
                answers=[answer],
                contexts=[contexts],
                ground_truths=[ground_truth] if ground_truth else None
            )
            
            # Select metrics based on available data
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)
            
            # Run RAGAS evaluation with temporary API key
            with temporary_openai_key(api_key):
                result = evaluate(dataset, metrics=metrics)
            
            # Extract scores
            scores = {
                "faithfulness": result["faithfulness"][0] if "faithfulness" in result else 0.0,
                "answer_relevancy": result["answer_relevancy"][0] if "answer_relevancy" in result else 0.0,
                "context_precision": result["context_precision"][0] if "context_precision" in result else 0.0,
                "context_recall": result["context_recall"][0] if "context_recall" in result and ground_truth else None
            }
            
            # Calculate medical quality indicators
            medical_quality = self._calculate_medical_quality_score(answer, contexts)
            
            # Calculate overall score
            score_values = [v for v in scores.values() if v is not None]
            overall_score = sum(score_values) / len(score_values) if score_values else 0.0
            
            # Check if passes thresholds
            passes_thresholds = all(
                scores[metric] >= threshold 
                for metric, threshold in self.thresholds.items()
                if scores[metric] is not None
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(scores, medical_quality)
            
            # Calculate evaluation time
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s. Overall score: {overall_score:.3f}")
            
            return RAGASEvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                faithfulness_score=scores["faithfulness"],
                answer_relevancy_score=scores["answer_relevancy"],
                context_precision_score=scores["context_precision"],
                context_recall_score=scores["context_recall"],
                overall_score=overall_score,
                passes_thresholds=passes_thresholds,
                evaluation_time=evaluation_time,
                model_used=config.openai.model,
                evaluation_timestamp=datetime.now(),
                metric_details={
                    "ragas_scores": scores,
                    "medical_quality": medical_quality
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            raise
    
    async def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ) -> List[RAGASEvaluationResult]:
        """
        Evaluate multiple query-answer pairs in batch.
        
        Args:
            queries: List of queries
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional list of ground truth answers
            api_key: OpenAI API key for RAGAS evaluation
            
        Returns:
            List of RAGASEvaluationResult objects
        """
        logger.info(f"Starting batch evaluation of {len(queries)} queries")
        
        # Validate input lengths
        if not (len(queries) == len(answers) == len(contexts)):
            raise ValueError("Queries, answers, and contexts must have the same length")
        
        if ground_truths and len(ground_truths) != len(queries):
            raise ValueError("Ground truths must have the same length as queries")
        
        # Run individual evaluations
        results = []
        for i, (query, answer, context_list) in enumerate(zip(queries, answers, contexts)):
            ground_truth = ground_truths[i] if ground_truths else None
            
            try:
                result = await self.evaluate_single(query, answer, context_list, ground_truth, api_key)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating query {i}: {e}")
                # Create a failed result
                results.append(RAGASEvaluationResult(
                    query=query,
                    answer=answer,
                    contexts=context_list,
                    ground_truth=ground_truth,
                    faithfulness_score=0.0,
                    answer_relevancy_score=0.0,
                    context_precision_score=0.0,
                    context_recall_score=None,
                    overall_score=0.0,
                    passes_thresholds=False,
                    evaluation_time=0.0,
                    model_used=config.openai.model,
                    evaluation_timestamp=datetime.now(),
                    metric_details={"error": str(e)},
                    recommendations=["Evaluation failed - check logs for details"]
                ))
        
        logger.info(f"Batch evaluation completed. {len(results)} results generated")
        return results
    
    def evaluate_rag_pipeline(
        self,
        retrieval_result: RetrievalResult,
        generation_result: GenerationResult,
        ground_truth: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> RAGASEvaluationResult:
        """
        Evaluate complete RAG pipeline result.
        
        Args:
            retrieval_result: Result from document retrieval
            generation_result: Result from response generation
            ground_truth: Optional ground truth answer
            api_key: OpenAI API key for RAGAS evaluation
            
        Returns:
            RAGASEvaluationResult
        """
        # Extract contexts from retrieval result
        contexts = [doc.page_content for doc in retrieval_result.documents]
        
        # Run evaluation
        return asyncio.run(self.evaluate_single(
            query=generation_result.query,
            answer=generation_result.response,
            contexts=contexts,
            ground_truth=ground_truth,
            api_key=api_key
        ))
    
    def get_evaluation_summary(self, results: List[RAGASEvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results."""
        if not results:
            return {}
        
        # Calculate average scores
        avg_scores = {
            "faithfulness": sum(r.faithfulness_score for r in results) / len(results),
            "answer_relevancy": sum(r.answer_relevancy_score for r in results) / len(results),
            "context_precision": sum(r.context_precision_score for r in results) / len(results),
            "overall": sum(r.overall_score for r in results) / len(results)
        }
        
        # Calculate context recall average (only for results with ground truth)
        recall_results = [r for r in results if r.context_recall_score is not None]
        if recall_results:
            avg_scores["context_recall"] = sum(r.context_recall_score for r in recall_results) / len(recall_results)
        
        # Calculate pass rates
        pass_rates = {
            "faithfulness": sum(1 for r in results if r.faithfulness_score >= self.thresholds["faithfulness"]) / len(results),
            "answer_relevancy": sum(1 for r in results if r.answer_relevancy_score >= self.thresholds["answer_relevancy"]) / len(results),
            "context_precision": sum(1 for r in results if r.context_precision_score >= self.thresholds["context_precision"]) / len(results),
            "overall": sum(1 for r in results if r.passes_thresholds) / len(results)
        }
        
        if recall_results:
            pass_rates["context_recall"] = sum(1 for r in recall_results if r.context_recall_score >= self.thresholds["context_recall"]) / len(recall_results)
        
        # Common recommendations
        all_recommendations = [rec for r in results for rec in r.recommendations]
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_evaluations": len(results),
            "average_scores": avg_scores,
            "pass_rates": pass_rates,
            "thresholds": self.thresholds,
            "top_recommendations": top_recommendations,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, results: List[RAGASEvaluationResult], filepath: str) -> bool:
        """Save evaluation results to file."""
        try:
            # Convert results to DataFrame
            data = []
            for result in results:
                data.append({
                    "query": result.query,
                    "answer": result.answer[:500] + "..." if len(result.answer) > 500 else result.answer,
                    "ground_truth": result.ground_truth,
                    "faithfulness_score": result.faithfulness_score,
                    "answer_relevancy_score": result.answer_relevancy_score,
                    "context_precision_score": result.context_precision_score,
                    "context_recall_score": result.context_recall_score,
                    "overall_score": result.overall_score,
                    "passes_thresholds": result.passes_thresholds,
                    "evaluation_time": result.evaluation_time,
                    "evaluation_timestamp": result.evaluation_timestamp.isoformat(),
                    "recommendations": "; ".join(result.recommendations)
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Saved {len(results)} evaluation results to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False 