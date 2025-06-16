import requests
from config.llm_config import TOGETHER_MODEL, TOGETHER_URL, get_together_headers
from .query_classifier import QueryClassifier

class QueryRewriter:
    """Rewrites user queries into a format optimized for retrieval."""
    
    def __init__(self):
        """Initialize the rewriter with Together LLM."""
        self.classifier = QueryClassifier()
        # Template for query rewriting
        self.rewrite_prompt = """
        Rewrite the following query to be more specific and retrieval-friendly.
        Focus on key entities and relationships. Remove unnecessary words.
        
        IMPORTANT RULES:
        1. DO NOT change the subject matter or add new information
        2. Only add words like 'definition', 'explanation', or 'information' if:
           - The original query explicitly asks for one (e.g., "What is X?" → "X definition")
           - The query is about understanding a concept (e.g., "How does X work?" → "X explanation")
        3. For factual queries, keep it simple (e.g., "Who won the Super Bowl?" → "Super Bowl winner")
        4. Preserve technical terms and proper nouns exactly as they appear
        5. Keep the query as simple as possible while maintaining its meaning
       
        Original query: {query}
        
        Rewritten query:"""
    
    def rewrite(self, 
                query: str, 
                max_length: int = 50) -> str:
        """Rewrite the query to be more retrieval-friendly.
        
        Args:
            query: The original user query
            max_length: Maximum length of the rewritten query
            
        Returns:
            Rewritten query optimized for retrieval
            
        Raises:
            ValueError: If the query is empty, too long, or the API call fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        if len(query) > 1000:  # Reasonable limit for API
            raise ValueError("Query is too long (max 1000 characters)")
        
        if max_length < 10:
            raise ValueError("max_length must be at least 10 characters")
            
        # Get query type from classifier
        try:
            query_type = self.classifier.get_primary_type(query)
        except ValueError:
            query_type = "lookup"  # Default to lookup if classification fails
            
        # Prepare the prompt
        prompt = self.rewrite_prompt.format(query=query)
        
        # Add query type context
        prompt += f"\nQuery type: {query_type}"
        
        try:
            # Get rewritten query from Together
            response = requests.post(
                TOGETHER_URL,
                headers=get_together_headers(),
                json={
                    "model": TOGETHER_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a query rewriter for a RAG (Retrieval Augmented Generation) system. Your task is to rewrite user queries to be more effective for semantic search and retrieval. Focus on key terms, entities, and relationships that will help find relevant documents. Return only the rewritten query without any additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_length
                }
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if "error" in response_json:
                raise ValueError(f"API Error: {response_json['error']}")
                
            # Extract and clean the rewritten query
            rewritten = response_json["choices"][0]["message"]["content"].strip()
            
            if not rewritten:
                raise ValueError("Empty response from API")
                
            # Remove the prompt from the output if it's included
            if "Rewritten query:" in rewritten:
                rewritten = rewritten.split("Rewritten query:")[-1].strip()
                
            if len(rewritten) > max_length:
                rewritten = rewritten[:max_length].strip()
            
            return rewritten
            
        except requests.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
