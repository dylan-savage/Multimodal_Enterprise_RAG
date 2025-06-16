import json
import requests
import re
from typing import Dict
from ingestion.document_chunk import DocumentChunk
from config.llm_config import TOGETHER_MODEL, TOGETHER_URL, get_together_headers

def extract_graph_data_from_chunk(chunk: DocumentChunk) -> Dict:
    system_prompt = """You are an expert at extracting structured information from text.
Your task is to identify entities and their relationships in the given text.

For entities, identify:
- People (PER)
- Organizations (ORG)
- Locations (LOC)
- Dates (DATE)
- Numbers (NUM)
- Other relevant entities (MISC)

For relationships, identify:
- Subject-verb-object triplets
- Hierarchical relationships
- Temporal relationships
- Causal relationships
- Part-whole relationships

Return your output in this JSON format:
{
    "entities": [
        {
            "name": "entity name",
            "type": "entity type",
            "start": start_index,
            "end": end_index
        }
    ],
    "relationships": [
        {
            "subject": "subject entity",
            "predicate": "relationship type",
            "object": "object entity",
        }
    ]
}"""

    user_prompt = f"""Analyze the following text and extract entities and relationships:

Text: {chunk.content}

Focus on:
1. Named entities (people, organizations, locations, dates, numbers)
2. Relationships between entities
3. Hierarchical and temporal relationships
4. Causal and part-whole relationships

Return the results in the specified JSON format."""

    response = requests.post(
        TOGETHER_URL,
        headers=get_together_headers(),
        json={
            "model": TOGETHER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
    )

    response_json = response.json()
    if "error" in response_json:
        raise ValueError(f"API Error: {response_json['error']}")

    raw_output = response_json["choices"][0]["message"]["content"]

    match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_output, re.DOTALL)
    if match:
        cleaned_output = match.group(1)
    else:
        cleaned_output = raw_output.strip()

    try:
        result = json.loads(cleaned_output)
        
        # Validate the structure
        if not isinstance(result, dict):
            raise ValueError("Result is not a dictionary")
        if "entities" not in result or "relationships" not in result:
            raise ValueError("Missing required keys: entities and relationships")
        
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\nRaw output: {raw_output}")

