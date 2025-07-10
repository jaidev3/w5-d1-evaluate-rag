from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.intents = ["Technical Support", "Billing/Account", "Feature Requests"]
    
    def train(self, queries, labels):
        X = self.vectorizer.fit_transform(queries)
        self.model.fit(X, labels)

    def predict(self, query):
        X_new = self.vectorizer.transform([query])
        intent_idx = self.model.predict(X_new)[0]
        return self.intents[intent_idx]
    
    def save(self, filename="intent_classifier.pkl"):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load(cls, filename="intent_classifier.pkl"):
        with open(filename, 'rb') as file:
            return pickle.load(file) 