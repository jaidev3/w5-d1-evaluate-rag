# 🎯 Customer Support RAG Pipeline with Intent Detection and Queue System

A production-ready customer support system with RAG (Retrieval-Augmented Generation) pipeline, intent classification, and advanced request queue management.

## 🚀 Features

### ✅ Core RAG Pipeline
- **Intent Classification**: Automatic categorization of queries into Technical Support, Billing/Account, and Feature Requests
- **Smart Response Generation**: Context-aware responses using local LLM (LM Studio) or OpenAI API
- **Prompt Engineering**: Intent-specific prompts for better response quality

### ✅ Advanced Queue System
- **Asynchronous Processing**: Submit queries without blocking the UI
- **Real-time Updates**: WebSocket-based live status monitoring
- **Concurrent Handling**: Process multiple requests simultaneously
- **Flexible Processing**: Choose between queued or immediate processing
- **Bulk Operations**: Submit multiple queries at once

### ✅ Enhanced User Interface
- **Dual Submission Modes**: Queue processing vs. immediate processing
- **Real-time Monitoring**: Auto-refresh queue status with configurable intervals
- **Request Management**: View, track, and manage all requests
- **Visual Indicators**: Status emojis and color-coded request states
- **Comprehensive History**: Track all requests with expandable details

### ✅ Production-Ready Features
- **WebSocket Support**: Real-time bidirectional communication
- **Request Persistence**: In-memory storage with request tracking
- **Background Processing**: Continuous queue processing
- **Error Handling**: Comprehensive error management and recovery
- **Statistics Dashboard**: Queue metrics and performance monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Customer Support System                      │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI  │  FastAPI Backend  │  WebSocket Updates         │
├─────────────────────────────────────────────────────────────────┤
│                    Request Queue System                         │
│  Queue Management → Background Processing → Real-time Updates   │
├─────────────────────────────────────────────────────────────────┤
│                    RAG Pipeline Core                            │
│  Intent Classification → Prompt Generation → LLM Response       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
cd backend
python app.py
```

The backend will start on `http://localhost:8000`

### 3. Start the Frontend

```bash
cd frontend
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

### 4. Test the System

Access the Streamlit interface and try submitting queries using either:
- **Queue Processing**: For asynchronous handling with real-time updates
- **Immediate Processing**: For synchronous processing with instant results

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/queue/submit/` | POST | Submit a request to the queue |
| `/queue/submit/bulk/` | POST | Submit multiple requests to the queue |
| `/queue/status/{request_id}` | GET | Get status of a specific request |
| `/queue/all/` | GET | Get all requests and their status |
| `/queue/clear/` | DELETE | Clear completed/failed requests |
| `/query/` | POST | Direct processing (bypasses queue) |
| `/ws` | WebSocket | Real-time updates |

## 🎮 Usage Guide

### Submitting Requests

#### Option 1: Queue Processing (Recommended)
1. Enter your query in the text area
2. Click "🚀 Submit to Queue"
3. Monitor the request in the Queue Status section
4. View the response when processing completes

#### Option 2: Bulk Processing
1. Enter multiple queries (one per line)
2. Click "🚀 Submit All to Queue"
3. Monitor all requests in real-time

#### Option 3: Immediate Processing
1. Enter your query in the text area
2. Click "⚡ Process Immediately"
3. Get instant response (blocks UI during processing)

### Monitoring Queue Status

- **Auto-refresh**: Enable automatic status updates (1-10 second intervals)
- **Manual refresh**: Click "🔄 Refresh All Requests"
- **Request details**: Expand any request to see full details
- **Queue statistics**: View real-time metrics in the sidebar

### Request States

- **🕐 QUEUED**: Request submitted and waiting for processing
- **🔄 PROCESSING**: Request is currently being processed
- **✅ COMPLETED**: Request processed successfully
- **❌ FAILED**: Request processing failed

## 🔧 Configuration

### LLM Configuration
The system supports both local and cloud-based LLMs:

- **Local LLM**: Uses LM Studio (default) - requires LM Studio running on `localhost:1234`
- **OpenAI API**: Configure `openai_api_key` in the LMWrapper class

### Intent Classification
The system automatically classifies queries into:
- **Technical Support**: Network issues, login problems, technical troubleshooting
- **Billing/Account**: Payment issues, account management, subscription queries
- **Feature Requests**: New feature suggestions, product improvements

## 📊 Queue System Features

### Real-time Updates
- WebSocket-based live status monitoring
- Automatic UI refresh with configurable intervals
- Instant notification of request state changes

### Request Management
- View all requests with status indicators
- Expandable request details with full context
- Clear completed requests to manage queue size
- Bulk operations for multiple queries

### Statistics Dashboard
- Total requests submitted, completed, and failed
- Success rate calculation
- Queue health indicators
- Request distribution by status

## 🧪 Testing

### Manual Testing
1. Submit single queries and monitor processing
2. Submit multiple queries to test concurrent processing
3. Test both queue and immediate processing modes
4. Verify real-time updates and WebSocket functionality

### Sample Queries
- "My internet is not working"
- "How do I reset my password?"
- "I want to cancel my subscription"
- "Can you add a dark mode feature?"

## 🔍 Evaluation

The system includes evaluation capabilities:
- **Classifier Performance**: Evaluate intent classification accuracy
- **Response Quality**: Assess LLM response relevance and helpfulness
- **Queue Performance**: Monitor processing times and success rates

Use `evaluation/evaluate.py` to run comprehensive evaluations.

## 📁 Project Structure

```
local-rag-pipline-with-intent-detection-and-evaluation/
├── README.md                    # This file
├── QUEUE_SYSTEM_README.md       # Detailed queue system documentation
├── requirements.txt             # Python dependencies
├── intent_classifier.pkl        # Trained intent classifier
│
├── backend/                     # FastAPI backend
│   ├── app.py                  # Main application with queue system
│   ├── classifier.py           # Intent classification logic
│   ├── lm_wrapper.py           # LLM integration (local/OpenAI)
│   └── utils.py                # Utility functions
│
├── frontend/                    # Streamlit frontend
│   └── app.py                  # Enhanced UI with queue management
│
└── evaluation/                  # Evaluation framework
    ├── evaluate.py             # Model evaluation
    └── metrics.py              # Performance metrics
```

## 🚀 Recent Enhancements

### Queue System Implementation
- Added comprehensive request queue with async processing
- Implemented WebSocket support for real-time updates
- Enhanced UI with dual processing modes and live monitoring
- Added bulk query processing capabilities

### UI Improvements
- Real-time queue status monitoring
- Enhanced request management with expandable details
- Statistics dashboard with performance metrics
- Auto-refresh functionality with configurable intervals

### API Enhancements
- Added multiple queue management endpoints
- Implemented WebSocket endpoint for real-time updates
- Added bulk submission capabilities
- Enhanced error handling and response formatting

## 🎯 Next Steps

For detailed information about the queue system implementation, see [QUEUE_SYSTEM_README.md](QUEUE_SYSTEM_README.md).

---

**Ready to handle customer support queries with intelligent intent detection and efficient queue management!** 🎯✨ 