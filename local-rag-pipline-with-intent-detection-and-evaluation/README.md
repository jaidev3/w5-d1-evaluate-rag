# Customer Support System

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd customer_support_system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start FastAPI backend:

   ```bash
   uvicorn backend.app:app --reload
   ```

4. Start Streamlit UI:

   ```bash
   streamlit run frontend/app.py
   ```

5. Test queries through the Streamlit UI.

## Evaluation

* Evaluate the model using a set of test queries in the `evaluation/test_data.json`.
* Use `evaluation/evaluate.py` to evaluate the classifier and LM model performance. 