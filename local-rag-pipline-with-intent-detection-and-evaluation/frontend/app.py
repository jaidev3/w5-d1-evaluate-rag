import streamlit as st
import requests
import json
import time
import asyncio
import websockets
import threading
from datetime import datetime
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

# Initialize session state
if 'requests' not in st.session_state:
    st.session_state.requests = {}
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'request_list' not in st.session_state:
    st.session_state.request_list = []
if 'response_list' not in st.session_state:
    st.session_state.response_list = []
if 'statistics' not in st.session_state:
    st.session_state.statistics = {
        'total_submitted': 0,
        'total_completed': 0,
        'total_failed': 0,
        'completion_rate': 0.0
    }

def submit_to_queue(query):
    """Submit a query to the processing queue"""
    try:
        response = requests.post(f"{API_URL}/queue/submit/", json={"query": query})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error submitting request: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def submit_bulk_to_queue(queries):
    """Submit multiple queries to the processing queue"""
    try:
        response = requests.post(f"{API_URL}/queue/submit/bulk/", json={"queries": queries})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error submitting bulk request: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def get_queue_status(request_id):
    """Get the status of a specific request"""
    try:
        response = requests.get(f"{API_URL}/queue/status/{request_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting status: {e}")
        return None

def get_all_requests():
    """Get all requests and their status"""
    try:
        response = requests.get(f"{API_URL}/queue/all/")
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.error(f"Error getting all requests: {e}")
        return []

def clear_completed_requests():
    """Clear all completed requests"""
    try:
        response = requests.delete(f"{API_URL}/queue/clear/")
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error clearing requests: {e}")
        return False

def direct_query(query):
    """Direct query processing (bypasses queue)"""
    try:
        with st.spinner("Processing your query directly..."):
            response = requests.post(f"{API_URL}/query/", json={"query": query})
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error processing query: {response.status_code}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def format_datetime(dt_str):
    """Format datetime string for display"""
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return dt_str
    return "N/A"

def get_status_emoji(status):
    """Get emoji for status"""
    status_emojis = {
        "queued": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "failed": "❌"
    }
    return status_emojis.get(status, "❓")

def update_request_response_lists():
    """Update request and response lists based on current queue status"""
    all_requests = get_all_requests()
    
    # Separate requests by status
    pending_requests = []
    completed_responses = []
    
    for req in all_requests:
        if req['status'] in ['queued', 'processing']:
            pending_requests.append(req)
        elif req['status'] in ['completed', 'failed']:
            completed_responses.append(req)
    
    # Update session state
    st.session_state.request_list = pending_requests
    st.session_state.response_list = completed_responses
    
    # Update statistics
    total_submitted = len(all_requests)
    total_completed = len([r for r in all_requests if r['status'] == 'completed'])
    total_failed = len([r for r in all_requests if r['status'] == 'failed'])
    completion_rate = (total_completed / total_submitted * 100) if total_submitted > 0 else 0
    
    st.session_state.statistics = {
        'total_submitted': total_submitted,
        'total_completed': total_completed,
        'total_failed': total_failed,
        'completion_rate': completion_rate
    }

def parse_bulk_queries(text):
    """Parse bulk query text into individual queries"""
    if not text.strip():
        return []
    
    # Split by newlines and filter out empty lines
    queries = [q.strip() for q in text.split('\n') if q.strip()]
    return queries

# Main UI
st.set_page_config(page_title="Customer Support System with Queue", layout="wide")

st.title("🎯 Customer Support System with Request Queue")

# Sidebar for settings and controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh queue status", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh interval
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 3)
    
    st.divider()
    
    # Queue management
    st.header("🔧 Queue Management")
    
    if st.button("🔄 Refresh All Requests"):
        st.rerun()
    
    if st.button("🗑️ Clear Completed Requests"):
        if clear_completed_requests():
            st.success("Completed requests cleared!")
            st.rerun()
    
    st.divider()
    
    # Queue statistics
    st.header("📊 Queue Statistics")
    
    # Update the request/response lists and statistics
    update_request_response_lists()
    
    all_requests = get_all_requests()
    stats = st.session_state.statistics
    
    if all_requests:
        # Overall statistics
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("📝 Total Submitted", stats['total_submitted'])
            st.metric("✅ Completed", stats['total_completed'])
        with col_stat2:
            st.metric("❌ Failed", stats['total_failed'])
            st.metric("📊 Success Rate", f"{stats['completion_rate']:.1f}%")
        
        st.divider()
        
        # Status breakdown
        status_counts = {}
        for req in all_requests:
            status = req['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            st.metric(f"{get_status_emoji(status)} {status.title()}", count)
    else:
        st.info("No requests in queue")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Submit Queries")
    
    # Query input mode selection
    query_mode = st.radio("Query Mode:", ["Single Query", "Multiple Queries"], horizontal=True)
    
    if query_mode == "Single Query":
        # Single query input
        query = st.text_area("Enter your query:", placeholder="Type your question here...", height=100)
        
        # Submit options
        col_submit1, col_submit2 = st.columns(2)
        
        with col_submit1:
            if st.button("🚀 Submit to Queue", type="primary", use_container_width=True):
                if query.strip():
                    result = submit_to_queue(query.strip())
                    if result:
                        st.success(f"✅ Request submitted! ID: {result['request_id']}")
                        st.info(f"Position in queue: {result['position_in_queue']}")
                        st.rerun()
                else:
                    st.warning("Please enter a query")
        
        with col_submit2:
            if st.button("⚡ Process Immediately", use_container_width=True):
                if query.strip():
                    result = direct_query(query.strip())
                    if result:
                        st.success("✅ Query processed immediately!")
                        with st.expander("See response details", expanded=True):
                            st.subheader("🎯 Intent")
                            st.write(result["intent"])
                            st.subheader("💬 Response")
                            st.write(result["response"])
                else:
                    st.warning("Please enter a query")
    
    else:  # Multiple Queries
        # Multiple queries input
        bulk_queries = st.text_area(
            "Enter multiple queries (one per line):", 
            placeholder="Query 1\nQuery 2\nQuery 3\n...", 
            height=150
        )
        
        if bulk_queries.strip():
            queries = parse_bulk_queries(bulk_queries)
            st.info(f"📝 {len(queries)} queries detected")
            
            # Preview queries
            with st.expander("Preview Queries", expanded=False):
                for i, q in enumerate(queries, 1):
                    st.write(f"{i}. {q}")
        
        # Submit bulk queries
        if st.button("🚀 Submit All to Queue", type="primary", use_container_width=True):
            if bulk_queries.strip():
                queries = parse_bulk_queries(bulk_queries)
                if queries:
                    result = submit_bulk_to_queue(queries)
                    if result:
                        st.success(f"✅ {result['total_submitted']} queries submitted successfully!")
                        st.info(f"Request IDs: {', '.join([id[:8] + '...' for id in result['request_ids']])}")
                        st.rerun()
                else:
                    st.warning("No valid queries found")
            else:
                st.warning("Please enter some queries")

with col2:
    st.header("📋 Request & Response Lists")
    
    # Update lists
    update_request_response_lists()
    
    # Create tabs for requests and responses
    tab1, tab2 = st.tabs(["🔄 Pending Requests", "✅ Completed Responses"])
    
    with tab1:
        st.subheader("🔄 Pending Requests")
        request_list = st.session_state.request_list
        
        if request_list:
            # Sort by creation time (newest first)
            request_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            st.info(f"📊 {len(request_list)} requests pending")
            
            # Display pending requests
            for i, req in enumerate(request_list):
                status = req['status']
                status_emoji = get_status_emoji(status)
                
                # Create expandable container for each request
                with st.expander(f"{status_emoji} {req['query'][:50]}... ({status})", expanded=(status == 'processing')):
                    
                    # Request details
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.text(f"🆔 ID: {req['id'][:8]}...")
                        st.text(f"⏰ Created: {format_datetime(req['created_at'])}")
                    
                    with col_detail2:
                        st.text(f"📊 Status: {status.title()}")
                        if req['intent']:
                            st.text(f"🎯 Intent: {req['intent']}")
                    
                    # Full query
                    st.text_area("📝 Query:", value=req['query'], height=68, disabled=True, key=f"pending_query_{req['id']}")
                    
                    # Status message
                    if status == 'processing':
                        st.info("🔄 Processing your request...")
                    elif status == 'queued':
                        st.info("⏳ Request is queued for processing...")
        else:
            st.info("🔍 No pending requests. Submit a query to get started!")
    
    with tab2:
        st.subheader("✅ Completed Responses")
        response_list = st.session_state.response_list
        
        if response_list:
            # Sort by completion time (newest first)
            response_list.sort(key=lambda x: x['completed_at'] or x['created_at'], reverse=True)
            
            st.success(f"📊 {len(response_list)} responses available")
            
            # Display completed responses
            for i, req in enumerate(response_list):
                status = req['status']
                status_emoji = get_status_emoji(status)
                
                # Create expandable container for each response
                with st.expander(f"{status_emoji} {req['query'][:50]}... ({status})", expanded=False):
                    
                    # Response details
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.text(f"🆔 ID: {req['id'][:8]}...")
                        st.text(f"⏰ Created: {format_datetime(req['created_at'])}")
                        if req['completed_at']:
                            st.text(f"✅ Completed: {format_datetime(req['completed_at'])}")
                    
                    with col_detail2:
                        st.text(f"📊 Status: {status.title()}")
                        if req['intent']:
                            st.text(f"🎯 Intent: {req['intent']}")
                    
                    # Original query
                    st.text_area("📝 Original Query:", value=req['query'], height=68, disabled=True, key=f"completed_query_{req['id']}")
                    
                    # Response or error
                    if req['response']:
                        st.text_area("💬 Response:", value=req['response'], height=120, disabled=True, key=f"response_{req['id']}")
                    elif req['error']:
                        st.error(f"❌ Error: {req['error']}")
                    
                    # Action buttons
                    if st.button(f"🗑️ Remove from List", key=f"remove_{req['id']}"):
                        st.info("Use 'Clear Completed Requests' in sidebar to remove completed requests")
        else:
            st.info("🔍 No completed responses yet. Responses will appear here when queries are processed!")

# Auto-refresh functionality
if st.session_state.auto_refresh and auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Enhanced Statistics and Summary
st.header("📊 Enhanced Statistics & Summary")

# Update statistics
update_request_response_lists()
all_requests = get_all_requests()
stats = st.session_state.statistics

if all_requests:
    # Statistics overview
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("📝 Total Submitted", stats['total_submitted'])
    with col_stat2:
        st.metric("🔄 Pending", len(st.session_state.request_list))
    with col_stat3:
        st.metric("✅ Completed", stats['total_completed'])
    with col_stat4:
        st.metric("📊 Success Rate", f"{stats['completion_rate']:.1f}%")
    
    # Detailed summary table
    st.subheader("📋 All Requests Summary")
    
    # Create DataFrame for better visualization
    df_data = []
    for req in all_requests:
        df_data.append({
            'ID': req['id'][:8] + '...',
            'Query': req['query'][:40] + '...' if len(req['query']) > 40 else req['query'],
            'Status': f"{get_status_emoji(req['status'])} {req['status'].title()}",
            'Intent': req['intent'] or 'N/A',
            'Created': format_datetime(req['created_at']),
            'Completed': format_datetime(req['completed_at']) if req['completed_at'] else 'N/A',
            'List': 'Pending' if req['status'] in ['queued', 'processing'] else 'Completed'
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # List summaries
    col_list1, col_list2 = st.columns(2)
    
    with col_list1:
        st.subheader("🔄 Request List Summary")
        if st.session_state.request_list:
            for req in st.session_state.request_list[:3]:  # Show first 3
                st.write(f"• {get_status_emoji(req['status'])} {req['query'][:30]}...")
            if len(st.session_state.request_list) > 3:
                st.write(f"... and {len(st.session_state.request_list) - 3} more")
        else:
            st.info("No pending requests")
    
    with col_list2:
        st.subheader("✅ Response List Summary")
        if st.session_state.response_list:
            for req in st.session_state.response_list[:3]:  # Show first 3
                st.write(f"• {get_status_emoji(req['status'])} {req['query'][:30]}...")
            if len(st.session_state.response_list) > 3:
                st.write(f"... and {len(st.session_state.response_list) - 3} more")
        else:
            st.info("No completed responses")
else:
    st.info("No requests to display in summary")

# Footer
st.divider()
st.markdown("### 💡 How to Use")
st.markdown("""
- **Single Query**: Submit one query at a time to the queue or process immediately
- **Multiple Queries**: Submit multiple queries at once (one per line) for batch processing
- **Request List**: View pending queries that are queued or being processed
- **Response List**: View completed queries with their responses
- **Statistics**: Monitor success rates and completion statistics
- **Auto-refresh**: Enable to automatically update all lists and statistics
- **Queue Management**: Use sidebar controls to manage and monitor the queue
""")

# Real-time updates notice
if st.session_state.auto_refresh:
    st.success("🔄 Auto-refresh is enabled - queue status updates automatically")
else:
    st.info("🔄 Auto-refresh is disabled - click 'Refresh All Requests' to update manually") 