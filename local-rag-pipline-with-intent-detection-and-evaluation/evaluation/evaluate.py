from sklearn.metrics import accuracy_score
from .metrics import calculate_cosine_similarity

def generate_prompt(intent, query):
    if intent == "Technical Support":
        return f"Provide technical support for this issue: {query}"
    elif intent == "Billing/Account":
        return f"Answer the billing or account question: {query}"
    elif intent == "Feature Requests":
        return f"Provide information about the feature request: {query}"
    else:
        return query

def evaluate_model(test_data, classifier, lm_wrapper):
    y_true = [data['intent'] for data in test_data]
    y_pred = []
    responses = []
    
    for data in test_data:
        query = data['query']
        intent = classifier.predict(query)
        prompt = generate_prompt(intent, query)
        response = lm_wrapper.query(prompt)
        y_pred.append(intent)
        responses.append(response)
    
    accuracy = accuracy_score(y_true, y_pred)
    relevance = calculate_cosine_similarity(responses, [data['expected_response'] for data in test_data])
    return accuracy, relevance 