from src.evaluation.llm_judge import LLMJudge
import logging

logger = logging.getLogger(__name__)

class LightweightEvaluator:
    """Lightweight RAG evaluator using LLM-as-a-Judge approach."""
    
    def __init__(self, llm_client=None):
        self.judge = LLMJudge() if llm_client is None else LLMJudge(llm_client)
        logger.info("Lightweight Evaluator initialized")
    
    def evaluate_query(self, question: str, context: str, answer: str) -> dict:
        """Evaluates a single query using LLM Judge."""
        try:
            # Use the existing LLMJudge for faithfulness and relevance
            scores = self.judge.evaluate(question, context, answer)
            
            # Add some simple heuristic metrics
            metrics = {
                "faithfulness": scores.get("faithfulness", 0),
                "relevance": scores.get("relevance", 0),
                "answer_length": len(answer.split()),
                "context_utilization": self._calculate_context_utilization(context, answer),
                "response_quality": self._calculate_response_quality(answer)
            }
            
            logger.info(f"Evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_context_utilization(self, context: str, answer: str) -> float:
        """Simple heuristic: how much of the context seems to be used in the answer."""
        if not context or not answer:
            return 0.0
        
        # Count how many words from context appear in answer
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(context_words.intersection(answer_words))
        utilization = overlap / len(context_words)
        
        return min(utilization * 2, 1.0)  # Scale up and cap at 1.0
    
    def _calculate_response_quality(self, answer: str) -> float:
        """Simple heuristic based on answer characteristics."""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # Length bonus (not too short, not too long)
        word_count = len(answer.split())
        if 20 <= word_count <= 200:
            score += 0.3
        elif word_count > 200:
            score += 0.2  # Penalty for being too verbose
        
        # Structure bonus (has sentences, proper punctuation)
        if '.' in answer and len(answer.split('.')) > 1:
            score += 0.2
        
        # Confidence indicators
        confidence_words = ['clearly', 'definitely', 'certainly', 'obviously']
        if any(word in answer.lower() for word in confidence_words):
            score += 0.1
        
        # Uncertainty penalties
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'uncertain', 'not sure']
        if any(word in answer.lower() for word in uncertainty_words):
            score -= 0.1
        
        return max(0.0, min(score, 1.0))