import json
import requests
import re
from typing import Dict
from ingestion.document_chunk import DocumentChunk
from config.llm_config import TOGETHER_MODEL, TOGETHER_URL, get_together_headers

def extract_graph_data_from_chunk(chunk: DocumentChunk) -> Dict:
    prompt = f"""
    You are an information extraction engine. Your task is to read the following document text and extract:

    1. Entities — people, organizations, locations, or any other important nouns and other named things.
    2. Relationships — in the form of triplets (subject, predicate, object) that describe meaningful connections.

    Even if the input is phrased as a short sentence or question, infer any implied relationships that can be reasonably identified.

    Return your output in this JSON format:
    {{
    "entities": [...],
    "relationships": [...]
    }}

    Return your output in valid JSON with two top-level keys:
    - "entities": a list of objects with name and type
    - "relationships": a list of subject-predicate-object triplets


    Here is the input text:

    \"\"\"{chunk.content}\"\"\"
    """

    response = requests.post(
        TOGETHER_URL,
        headers=get_together_headers(),
        json={
            "model": TOGETHER_MODEL,
            "messages": [
                {"role": "system", "content": "You extract structured entities and relationships from documents."},
                {"role": "user", "content": prompt}
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
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        raise ValueError(f"LLM response could not be parsed as JSON:\n{cleaned_output}")

