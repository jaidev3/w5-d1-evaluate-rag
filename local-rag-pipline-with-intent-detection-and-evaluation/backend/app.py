from fastapi import FastAPI
from pydantic import BaseModel
from .lm_wrapper import LMWrapper
from .classifier import IntentClassifier
from typing import List
import os

app = FastAPI()
lm_wrapper = LMWrapper(use_local=True)

# Try to load classifier, create a new one if it doesn't exist
try:
    intent_classifier = IntentClassifier.load('intent_classifier.pkl')
except FileNotFoundError:
    print("Classifier not found, creating a new one...")
    intent_classifier = IntentClassifier()
    # Train with some sample data
    sample_queries = [
        "My internet is not working",
        "I can't log into my account", 
        "Can you add a new feature?",
        "How do I reset my password?",
        "I want to cancel my subscription",
        "Can you implement dark mode?"
    ]
    sample_labels = [0, 1, 2, 1, 1, 2]  # 0: Technical, 1: Billing, 2: Feature
    intent_classifier.train(sample_queries, sample_labels)
    intent_classifier.save('intent_classifier.pkl')

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    intent: str
    response: str

@app.post("/query/", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    intent = intent_classifier.predict(request.query)
    prompt = generate_prompt(intent, request.query)
    response = lm_wrapper.query(prompt)
    return QueryResponse(intent=intent, response=response)

def generate_prompt(intent, query):
    if intent == "Technical Support":
        return f"Provide technical support for this issue: {query}"
    elif intent == "Billing/Account":
        return f"Answer the billing or account question: {query}"
    elif intent == "Feature Requests":
        return f"Provide information about the feature request: {query}"
    else:
        return query

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
