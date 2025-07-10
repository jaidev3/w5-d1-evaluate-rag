import streamlit as st
import requests

API_URL = "http://localhost:8000/query/"

def get_response(query):
    response = requests.post(API_URL, json={"query": query})
    return response.json()

st.title("Customer Support System")
query = st.text_input("Enter your query:")
if query:
    result = get_response(query)
    st.write(f"Intent: {result['intent']}")
    st.write(f"Response: {result['response']}") 