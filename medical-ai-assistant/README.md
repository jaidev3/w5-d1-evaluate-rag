# Medical AI Assistant RAG Pipeline

A production-ready Medical Knowledge Assistant RAG (Retrieval-Augmented Generation) pipeline for healthcare professionals to query medical literature, drug interactions, and clinical guidelines using OpenAI API with comprehensive RAGAS evaluation framework.

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Medical AI Assistant                          │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI  │  FastAPI Backend  │  RAGAS Monitoring          │
├─────────────────────────────────────────────────────────────────┤
│                    RAG Pipeline Core                            │
│  Document Ingestion → Vector DB → Retrieval → OpenAI Generation │
├─────────────────────────────────────────────────────────────────┤
│                    RAGAS Evaluation                             │
│  Context Precision | Context Recall | Faithfulness | Relevancy  │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: FastAPI (RESTful API)
- **Frontend**: Streamlit (Interactive UI)
- **Vector Database**: ChromaDB (Document embeddings)
- **LLM**: OpenAI GPT-4 (Generation)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Evaluation**: RAGAS (Retrieval-Augmented Generation Assessment)
- **Deployment**: Docker + Docker Compose

## 📋 Project Requirements

### Core System
- ✅ RAG Pipeline: Medical document ingestion → Vector DB → Retrieval → OpenAI generation
- ✅ Data Sources: Medical PDFs, drug databases, clinical protocols
- ✅ API: RESTful endpoints for medical queries

### RAGAS Implementation
- ✅ Core Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
- ✅ Medical Evaluation: Custom datasets with medical Q&A pairs
- ✅ Automated Pipeline: Batch evaluation and real-time monitoring
- ✅ Quality Thresholds: Faithfulness >0.90, Context Precision >0.85

### Production Features
- ✅ Monitoring Dashboard: Real-time RAGAS metrics tracking
- ✅ Safety System: RAGAS-validated response filtering
- ✅ Performance: Response latency p95 < 3 seconds
- ✅ Deployment: Dockerized API with RAGAS monitoring

## 🚀 Step-by-Step Implementation Process

### Phase 1: Foundation Setup
1. **Project Structure Creation**
   - Set up directory structure with proper separation of concerns
   - Create configuration files and environment setup

2. **Dependencies Installation**
   - Install core libraries: FastAPI, Streamlit, RAGAS, OpenAI
   - Set up vector database (ChromaDB) and data processing tools

### Phase 2: RAG Pipeline Development
3. **Document Processing System**
   - Implement PDF ingestion for medical documents
   - Create text chunking strategy optimized for medical content
   - Build embedding pipeline using OpenAI embeddings

4. **Vector Database Setup**
   - Configure ChromaDB for medical document storage
   - Implement similarity search and retrieval functions
   - Add metadata filtering for document types

5. **OpenAI Integration**
   - Set up OpenAI API client with proper error handling
   - Implement prompt engineering for medical queries
   - Add response generation with context injection

### Phase 3: RAGAS Evaluation Framework
6. **Core Metrics Implementation**
   - **Context Precision**: Measures relevance of retrieved contexts
   - **Context Recall**: Measures completeness of retrieved information
   - **Faithfulness**: Ensures generated answers are grounded in context
   - **Answer Relevancy**: Measures how well answers address the query

7. **Medical Evaluation Dataset**
   - Create custom medical Q&A pairs for evaluation
   - Implement automated evaluation pipeline
   - Set up quality thresholds and monitoring

### Phase 4: API Development
8. **FastAPI Backend**
   - Create RESTful endpoints for medical queries
   - Implement real-time RAGAS evaluation
   - Add monitoring and logging capabilities

9. **Streamlit Frontend**
   - Build interactive UI for medical professionals
   - Display real-time RAGAS metrics
   - Create query interface with response visualization

### Phase 5: Production Features
10. **Monitoring Dashboard**
    - Real-time RAGAS metrics tracking
    - Performance monitoring and alerting
    - Usage analytics and reporting

11. **Safety System**
    - RAGAS-validated response filtering
    - Harmful content detection
    - Quality threshold enforcement

12. **Deployment**
    - Dockerized application with multi-service architecture
    - Production-ready configuration
    - Scalability and reliability features

## 🎯 Why This Architecture?

### RAG Pipeline Benefits
- **Accuracy**: Grounds responses in verified medical literature
- **Transparency**: Shows source documents for each answer
- **Updatability**: Easy to add new medical documents without retraining
- **Cost-Effective**: Uses smaller, focused knowledge base vs. fine-tuning

### RAGAS Evaluation Importance
- **Quality Assurance**: Ensures medical accuracy and safety
- **Continuous Monitoring**: Real-time quality assessment
- **Regulatory Compliance**: Provides audit trail for medical AI systems
- **Trust Building**: Transparent evaluation metrics for healthcare professionals

### Production-Ready Features
- **Scalability**: Docker deployment supports horizontal scaling
- **Monitoring**: Real-time metrics for system health
- **Safety**: Multi-layer validation prevents harmful responses
- **Performance**: Optimized for sub-3-second response times

## 📁 Project Structure

```
medical-ai-assistant/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Multi-service deployment
├── Dockerfile                  # Container configuration
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore patterns
│
├── app/                       # Main application code
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── config.py             # Configuration management
│   │
│   ├── core/                 # Core RAG pipeline
│   │   ├── __init__.py
│   │   ├── document_processor.py    # PDF ingestion and chunking
│   │   ├── vector_store.py         # ChromaDB operations
│   │   ├── retriever.py           # Document retrieval logic
│   │   └── generator.py           # OpenAI response generation
│   │
│   ├── evaluation/           # RAGAS evaluation framework
│   │   ├── __init__.py
│   │   ├── ragas_evaluator.py     # Core RAGAS implementation
│   │   ├── metrics.py             # Individual metric calculations
│   │   └── safety_filter.py       # Response safety validation
│   │
│   ├── api/                  # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── routes.py              # API route definitions
│   │   └── models.py              # Pydantic models
│   │
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── logging.py             # Logging configuration
│       └── monitoring.py          # Metrics collection
│
├── ui/                       # Streamlit frontend
│   ├── __init__.py
│   ├── app.py                # Main Streamlit application
│   ├── components/           # UI components
│   │   ├── __init__.py
│   │   ├── query_interface.py     # Query input interface
│   │   ├── results_display.py     # Response visualization
│   │   └── metrics_dashboard.py   # RAGAS metrics display
│   └── utils/
│       ├── __init__.py
│       └── api_client.py          # FastAPI client
│
├── data/                     # Data storage
│   ├── documents/            # Medical documents (PDFs)
│   ├── evaluation/           # RAGAS evaluation datasets
│   │   ├── medical_qa_pairs.json
│   │   └── evaluation_results.json
│   └── vector_db/            # ChromaDB storage
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_rag_pipeline.py
│   ├── test_ragas_evaluation.py
│   └── test_api_endpoints.py
│
├── scripts/                  # Utility scripts
│   ├── setup_data.py         # Data preparation
│   ├── run_evaluation.py     # Batch RAGAS evaluation
│   └── demo.py               # Complete system demo
│
└── monitoring/               # Monitoring and logging
    ├── prometheus.yml        # Metrics collection
    └── grafana/              # Dashboard configuration
        └── dashboards/
            └── ragas_metrics.json
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd medical-ai-assistant
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Sample Data
```bash
python scripts/setup_data.py
```

### 4. Run Development Server

#### Option 1: Using Startup Scripts (Recommended)
```bash
# Backend (FastAPI) - handles asyncio event loop properly
python start_backend.py

# Frontend (Streamlit) - in another terminal, handles imports properly
python start_ui.py
```

#### Option 2: Manual Commands
```bash
# Backend (FastAPI) - use asyncio loop to avoid uvloop conflict
uvicorn app.main:app --reload --port 8000 --loop asyncio

# Frontend (Streamlit) - run from ui directory
cd ui && streamlit run app.py --server.port 8501
```

### 5. Docker Deployment
```bash
docker-compose up --build
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Backend Startup Error (uvloop conflict)
**Error**: `ValueError: Can't patch loop of type <class 'uvloop.Loop'>`

**Solution**: Use the provided startup script or the asyncio loop option:
```bash
# Use startup script (recommended)
python start_backend.py

# Or use asyncio loop manually
uvicorn app.main:app --reload --port 8000 --loop asyncio
```

#### 2. Frontend Import Error
**Error**: `ModuleNotFoundError: No module named 'ui'`

**Solution**: Use the provided startup script or run from the ui directory:
```bash
# Use startup script (recommended)
python start_ui.py

# Or run from ui directory
cd ui && streamlit run app.py --server.port 8501
```

#### 3. Missing Dependencies
**Error**: Various import errors for packages

**Solution**: Install requirements and activate virtual environment:
```bash
pip install -r requirements.txt
# Make sure you're in the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

## 🔍 Usage

### Web Interface
1. Open http://localhost:8501 for Streamlit UI
2. Enter medical query in the search box
3. View generated response with RAGAS metrics
4. Check source documents and evaluation scores

### API Endpoints
- `POST /query` - Submit medical query
- `GET /metrics` - Get RAGAS evaluation metrics
- `GET /health` - System health check
- `GET /docs` - API documentation

### Example Query
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the contraindications for aspirin?",
    "include_sources": True,
    "evaluate_with_ragas": True
})

print(response.json())
```

## 📊 RAGAS Metrics Explained

### 1. Context Precision (Target: >0.85)
- Measures how relevant the retrieved contexts are to the query
- Higher scores indicate better retrieval quality
- Critical for medical accuracy

### 2. Context Recall (Target: >0.80)
- Measures completeness of retrieved information
- Ensures all relevant information is captured
- Important for comprehensive medical advice

### 3. Faithfulness (Target: >0.90)
- Ensures generated answers are grounded in retrieved context
- Prevents hallucination in medical responses
- Most critical metric for medical safety

### 4. Answer Relevancy (Target: >0.85)
- Measures how well the answer addresses the original query
- Ensures responses are on-topic and useful
- Important for user experience

## 🛡️ Safety Features

### Response Filtering
- RAGAS-validated responses with quality thresholds
- Automatic rejection of low-quality answers
- Harmful content detection and filtering

### Quality Thresholds
- Faithfulness >0.90 (medical accuracy)
- Context Precision >0.85 (retrieval quality)
- Answer Relevancy >0.85 (response quality)

### Monitoring & Alerts
- Real-time RAGAS metrics tracking
- Automated alerts for quality degradation
- Comprehensive logging and audit trails

## 🚀 Production Deployment

### Docker Compose Services
- **api**: FastAPI backend
- **ui**: Streamlit frontend
- **vector-db**: ChromaDB service
- **monitoring**: Prometheus + Grafana

### Scaling Considerations
- Horizontal scaling of API services
- Load balancing for high availability
- Vector database optimization
- Caching layer for frequent queries

## 📈 Performance Targets

- **Response Latency**: p95 < 3 seconds
- **Faithfulness Score**: >0.90
- **Context Precision**: >0.85
- **System Uptime**: >99.9%

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_rag_pipeline.py
pytest tests/test_ragas_evaluation.py
pytest tests/test_api_endpoints.py
```

## 📝 Demo Script

Run the complete demonstration:
```bash
python scripts/demo.py
```

This will show:
1. Document ingestion process
2. Query processing pipeline
3. RAGAS evaluation in action
4. Real-time monitoring dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure RAGAS metrics meet thresholds
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for healthcare professionals and should not replace professional medical judgment. Always consult with qualified healthcare providers for medical decisions. 