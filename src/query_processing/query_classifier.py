from typing import Literal, Dict
import json
import requests
from config.llm_config import TOGETHER_API_KEY, TOGETHER_MODEL, TOGETHER_URL, get_together_headers

class QueryClassifier:
    """Lightweight classifier for categorizing user queries."""
    
    QUERY_TYPES = Literal["lookup", "relational", "ambiguous", "summarization", "comparison"]
    
    def __init__(self):
        """Initialize the classifier with Together LLM."""
        # Define candidate labels for classification
        self.candidate_labels = [
            "lookup - simple fact retrieval",
            "relational - connecting multiple pieces of information",
            "ambiguous - unclear or needs clarification",
            "summarization - requires condensing information",
            "comparison - comparing multiple items or concepts"
        ]
    
    def classify(self, query: str) -> Dict[str, float]:
        """Classify the query into one or more categories.
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary mapping query types to confidence scores
            
        Raises:
            ValueError: If the query is empty or the API call fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Prepare the prompt
        prompt = f"""Classify the following query into one or more of these categories:
        {', '.join(self.candidate_labels)}
        
        Query: {query}
        
        Return a JSON object with category names as keys and confidence scores (0-1) as values.
        Example: {{"lookup": 0.8, "relational": 0.2}}
        """
        
        try:
            # Get classification from Together
            response = requests.post(
                TOGETHER_URL,
                headers=get_together_headers(),
                json={
                    "model": TOGETHER_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a query classifier. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if "error" in response_json:
                raise ValueError(f"API Error: {response_json['error']}")
                
            # Extract and parse the classification
            raw_output = response_json["choices"][0]["message"]["content"]
            try:
                classifications = json.loads(raw_output)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                import re
                match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_output, re.DOTALL)
                if match:
                    classifications = json.loads(match.group(1))
                else:
                    raise ValueError(f"Could not parse classification output: {raw_output}")
                
            # Validate classifications
            if not isinstance(classifications, dict):
                raise ValueError("Classification output must be a dictionary")
                
            for category, score in classifications.items():
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid confidence score for {category}: {score}")
                    
            return classifications
            
        except requests.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
    
    def get_primary_type(self, query: str) -> QUERY_TYPES:
        """Get the primary (highest confidence) query type.
        
        Args:
            query: The user's query string
            
        Returns:
            The primary query type
            
        Raises:
            ValueError: If the query is empty or classification fails
        """
        classifications = self.classify(query)
        if not classifications:
            raise ValueError("No classifications returned")
            
        primary_type = max(classifications.items(), key=lambda x: x[1])[0]
        if primary_type not in self.QUERY_TYPES.__args__:
            raise ValueError(f"Invalid primary type: {primary_type}")
            
        return primary_type 