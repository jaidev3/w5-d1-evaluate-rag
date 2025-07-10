# 🚀 Request Queue System

This document explains the new request queuing system implemented for the Customer Support RAG pipeline.

## 🎯 Overview

The request queue system allows for:
- **Asynchronous request processing** - Submit queries without blocking the UI
- **Concurrent request handling** - Process multiple requests simultaneously
- **Real-time status updates** - Monitor queue status and get live updates
- **Request management** - View all requests, their status, and responses
- **Flexible processing** - Choose between queued or immediate processing

## 🏗️ Architecture

### Backend Components

#### 1. Queue Management (`app.py`)
- **Request Queue**: Async queue for processing requests
- **Request Storage**: In-memory storage for request status and results
- **Background Processor**: Continuously processes queued requests
- **WebSocket Support**: Real-time updates to connected clients

#### 2. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/queue/submit/` | POST | Submit a request to the queue |
| `/queue/status/{request_id}` | GET | Get status of a specific request |
| `/queue/all/` | GET | Get all requests and their status |
| `/queue/clear/` | DELETE | Clear completed/failed requests |
| `/query/` | POST | Direct processing (bypasses queue) |
| `/ws` | WebSocket | Real-time updates |

#### 3. Request States

- **🕐 QUEUED**: Request submitted and waiting for processing
- **🔄 PROCESSING**: Request is currently being processed
- **✅ COMPLETED**: Request processed successfully
- **❌ FAILED**: Request processing failed

### Frontend Components

#### 1. Enhanced UI (`frontend/app.py`)
- **Dual submission modes**: Queue vs. immediate processing
- **Real-time queue monitoring** with auto-refresh
- **Request status display** with expandable details
- **Queue statistics** and management controls
- **Comprehensive request history** with filtering

#### 2. Key Features
- **Auto-refresh**: Automatically updates queue status
- **Visual indicators**: Status emojis and color coding
- **Request details**: Full query, response, and metadata
- **Queue management**: Clear completed requests
- **Summary table**: Overview of all requests

## 🚀 Getting Started

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

```bash
python test_queue_system.py
```

## 🎮 Usage Guide

### Submitting Requests

#### Option 1: Queue Processing (Recommended)
1. Enter your query in the text area
2. Click "🚀 Submit to Queue"
3. Monitor the request in the Queue Status section
4. View the response when processing completes

#### Option 2: Immediate Processing
1. Enter your query in the text area
2. Click "⚡ Process Immediately"
3. Get instant response (blocks UI during processing)

### Monitoring Queue Status

#### Real-time Updates
- Enable "Auto-refresh queue status" in the sidebar
- Set refresh interval (1-10 seconds)
- Queue status updates automatically

#### Manual Refresh
- Click "🔄 Refresh All Requests" in the sidebar
- Updates all request statuses

### Managing Requests

#### View Request Details
- Click on any request in the Queue Status section
- View full query, response, intent, and timestamps
- See processing status and error messages

#### Clear Completed Requests
- Click "🗑️ Clear Completed Requests" in the sidebar
- Removes all completed and failed requests
- Keeps queued and processing requests

### Queue Statistics

The sidebar shows:
- **Request counts** by status
- **Real-time metrics** updated automatically
- **Queue health** indicators

## 🔧 API Reference

### Submit to Queue

```bash
curl -X POST "http://localhost:8000/queue/submit/" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

Response:
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "position_in_queue": 3
}
```

### Check Status

```bash
curl "http://localhost:8000/queue/status/123e4567-e89b-12d3-a456-426614174000"
```

Response:
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "intent": "Billing/Account",
  "response": "To reset your password, please follow these steps...",
  "position_in_queue": null,
  "error": null
}
```

### Get All Requests

```bash
curl "http://localhost:8000/queue/all/"
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Queue update:', data);
};
```

## 🧪 Testing

### Automated Testing

Run the comprehensive test suite:

```bash
python test_queue_system.py
```

Tests include:
- **Queue submission** - Submit multiple requests
- **Status monitoring** - Track request progress
- **Concurrent processing** - Test multiple simultaneous requests
- **Direct queries** - Test immediate processing
- **Request management** - Test clearing completed requests

### Manual Testing

1. **Submit multiple requests** to test queuing
2. **Monitor queue status** in real-time
3. **Test concurrent submissions** from multiple tabs
4. **Verify WebSocket updates** work correctly
5. **Test error handling** with invalid requests

## 📊 Performance Considerations

### Queue Processing
- **Single background worker** processes requests sequentially
- **Concurrent submissions** are queued efficiently
- **Memory-based storage** (consider database for production)

### Scalability
- **Horizontal scaling**: Add more worker processes
- **Persistent storage**: Use Redis or database for request storage
- **Load balancing**: Distribute requests across multiple instances

### Monitoring
- **Queue length** monitoring
- **Processing time** metrics
- **Error rate** tracking
- **WebSocket connection** health

## 🔍 Troubleshooting

### Common Issues

#### Backend Not Starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Check for import errors

#### Frontend Connection Errors
- Ensure backend is running on localhost:8000
- Check firewall settings
- Verify API endpoints are accessible

#### Queue Not Processing
- Check background task is running
- Monitor console for error messages
- Verify classifier and LM wrapper are working

#### WebSocket Issues
- Check browser WebSocket support
- Verify WebSocket endpoint is accessible
- Monitor connection status in browser dev tools

### Debug Mode

Enable debug logging in the backend:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Future Enhancements

### Planned Features
- **Persistent storage** with database integration
- **Request prioritization** based on user roles
- **Batch processing** for multiple queries
- **Advanced queue management** with pause/resume
- **Detailed analytics** and reporting
- **Email notifications** for completed requests
- **Request cancellation** functionality

### Configuration Options
- **Queue size limits** to prevent memory issues
- **Processing timeouts** for long-running requests
- **Retry mechanisms** for failed requests
- **Custom intent classifiers** per request type

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

For questions or support, please open an issue in the repository. 