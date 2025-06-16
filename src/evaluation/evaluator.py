from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
import json
import re


# Work in progress for outputing evaluation metrics for each query


class CrewAIModel(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm
        
    def load_model(self):
        return self.llm
        
    def generate(self, prompt: str) -> str:
        response = self.llm.call(prompt)
        return str(response)
        
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
        
    def get_model_name(self) -> str:
        return "crewai_model"


def format_context_chunks(chunks):
    """Convert context chunks to a list of strings."""
    if isinstance(chunks, str):
        try:
            chunks = json.loads(chunks)
        except:
            return [chunks]
    
    if isinstance(chunks, list):
        return [str(chunk) for chunk in chunks]
    
    return [str(chunks)]


def get_retrieved_context(tasks):
    """Get the retrieved context from the retriever task."""
    retriever_output = tasks[0].output
    
    if isinstance(retriever_output, str):
        try:
            chunks = json.loads(retriever_output)
            if isinstance(chunks, list):
                return chunks
        except:
            pass
    
    if isinstance(retriever_output, str):
        if "content" in retriever_output.lower():
            content_matches = re.findall(r'content["\s:]+([^"]+)', retriever_output, re.IGNORECASE)
            if content_matches:
                return content_matches
    
    return [str(retriever_output)]


def evaluate_response(query: str, answer: str, context_chunks: list, llm):
    """Evaluate the response using DeepEval metrics."""
    try:
        eval_model = CrewAIModel(llm)
        
        formatted_context = format_context_chunks(context_chunks)
        if not formatted_context:
            formatted_context = ["No context available"]
        
        print("\nContext chunks for evaluation:")
        for i, chunk in enumerate(formatted_context):
            print(f"\nChunk {i+1}:")
            print(chunk)
        
        metrics = [
            HallucinationMetric(model=eval_model),
            AnswerRelevancyMetric(model=eval_model),
            ContextualRelevancyMetric(model=eval_model)
        ]
        
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            context=formatted_context
        )
        
        result = evaluate([test_case], metrics=metrics)
        
        print("\nEvaluation Results:")
        for metric in metrics:
            print(f"{metric.__class__.__name__} Score: {metric.score}")
            
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        print("Continuing without evaluation...") 