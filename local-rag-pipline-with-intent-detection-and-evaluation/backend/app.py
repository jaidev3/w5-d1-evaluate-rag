from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from .lm_wrapper import LMWrapper
from .classifier import IntentClassifier
from typing import List, Dict, Optional
import os
import uuid
import asyncio
from datetime import datetime
from enum import Enum
import json

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

class RequestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class QueuedRequest(BaseModel):
    id: str
    query: str
    status: RequestStatus
    intent: Optional[str] = None
    response: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class BulkQueryRequest(BaseModel):
    queries: List[str]

class QueryResponse(BaseModel):
    intent: str
    response: str

class BulkQueryResponse(BaseModel):
    request_ids: List[str]
    total_submitted: int
    positions_in_queue: List[int]

class QueueSubmissionResponse(BaseModel):
    request_id: str
    status: RequestStatus
    position_in_queue: int

class QueueStatusResponse(BaseModel):
    request_id: str
    status: RequestStatus
    intent: Optional[str] = None
    response: Optional[str] = None
    position_in_queue: Optional[int] = None
    error: Optional[str] = None

# Global queue and request storage
request_queue: asyncio.Queue = asyncio.Queue()
requests_storage: Dict[str, QueuedRequest] = {}
active_connections: List[WebSocket] = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

async def process_queue():
    """Background task to process queued requests"""
    while True:
        try:
            request_id = await request_queue.get()
            if request_id in requests_storage:
                request = requests_storage[request_id]
                request.status = RequestStatus.PROCESSING
                
                # Broadcast status update
                await manager.broadcast(json.dumps({
                    "type": "status_update",
                    "request_id": request_id,
                    "status": request.status.value
                }))
                
                try:
                    # Process the request
                    intent = intent_classifier.predict(request.query)
                    prompt = generate_prompt(intent, request.query)
                    response = lm_wrapper.query(prompt)
                    
                    # Update request with results
                    request.intent = intent
                    request.response = response
                    request.status = RequestStatus.COMPLETED
                    request.completed_at = datetime.now()
                    
                except Exception as e:
                    request.status = RequestStatus.FAILED
                    request.error = str(e)
                    request.completed_at = datetime.now()
                
                # Broadcast completion
                await manager.broadcast(json.dumps({
                    "type": "request_completed",
                    "request_id": request_id,
                    "status": request.status.value,
                    "intent": request.intent,
                    "response": request.response,
                    "error": request.error
                }))
                
        except Exception as e:
            print(f"Error processing queue: {e}")
            await asyncio.sleep(1)

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            # Echo back or handle specific client messages if needed
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/queue/submit/", response_model=QueueSubmissionResponse)
async def submit_to_queue(request: QueryRequest):
    """Submit a request to the processing queue"""
    request_id = str(uuid.uuid4())
    
    queued_request = QueuedRequest(
        id=request_id,
        query=request.query,
        status=RequestStatus.QUEUED,
        created_at=datetime.now()
    )
    
    requests_storage[request_id] = queued_request
    await request_queue.put(request_id)
    
    # Get position in queue
    position = request_queue.qsize()
    
    # Broadcast new request
    await manager.broadcast(json.dumps({
        "type": "new_request",
        "request_id": request_id,
        "query": request.query,
        "status": RequestStatus.QUEUED.value,
        "position": position
    }))
    
    return QueueSubmissionResponse(
        request_id=request_id,
        status=RequestStatus.QUEUED,
        position_in_queue=position
    )

@app.post("/queue/submit/bulk/", response_model=BulkQueryResponse)
async def submit_bulk_to_queue(request: BulkQueryRequest):
    """Submit multiple requests to the processing queue"""
    request_ids = []
    positions = []
    
    for query in request.queries:
        if query.strip():  # Only process non-empty queries
            request_id = str(uuid.uuid4())
            
            queued_request = QueuedRequest(
                id=request_id,
                query=query.strip(),
                status=RequestStatus.QUEUED,
                created_at=datetime.now()
            )
            
            requests_storage[request_id] = queued_request
            await request_queue.put(request_id)
            request_ids.append(request_id)
            
            # Get position in queue
            position = request_queue.qsize()
            positions.append(position)
            
            # Broadcast new request
            await manager.broadcast(json.dumps({
                "type": "new_request",
                "request_id": request_id,
                "query": query.strip(),
                "status": RequestStatus.QUEUED.value,
                "position": position
            }))
    
    return BulkQueryResponse(
        request_ids=request_ids,
        total_submitted=len(request_ids),
        positions_in_queue=positions
    )

@app.get("/queue/status/{request_id}", response_model=QueueStatusResponse)
async def get_queue_status(request_id: str):
    """Get the status of a specific request"""
    if request_id not in requests_storage:
        return QueueStatusResponse(
            request_id=request_id,
            status=RequestStatus.FAILED,
            error="Request not found"
        )
    
    request = requests_storage[request_id]
    position = None
    
    if request.status == RequestStatus.QUEUED:
        # Calculate position in queue (approximate)
        position = request_queue.qsize()
    
    return QueueStatusResponse(
        request_id=request_id,
        status=request.status,
        intent=request.intent,
        response=request.response,
        position_in_queue=position,
        error=request.error
    )

@app.get("/queue/all/")
async def get_all_requests():
    """Get all requests with their current status"""
    return [
        {
            "id": req.id,
            "query": req.query,
            "status": req.status.value,
            "intent": req.intent,
            "response": req.response,
            "created_at": req.created_at.isoformat(),
            "completed_at": req.completed_at.isoformat() if req.completed_at else None,
            "error": req.error
        }
        for req in requests_storage.values()
    ]

@app.delete("/queue/clear/")
async def clear_completed_requests():
    """Clear all completed and failed requests"""
    global requests_storage
    requests_storage = {
        k: v for k, v in requests_storage.items() 
        if v.status in [RequestStatus.QUEUED, RequestStatus.PROCESSING]
    }
    return {"message": "Completed requests cleared"}

# Keep the original endpoint for backward compatibility
@app.post("/query/", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Direct query processing (bypasses queue)"""
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
