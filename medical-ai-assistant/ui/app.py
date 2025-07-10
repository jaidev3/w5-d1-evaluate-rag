"""
Medical AI Assistant - Streamlit UI
Interactive web interface for medical knowledge queries with RAGAS evaluation.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time
import json

from utils.api_client import (
    get_api_client, display_api_error, format_response_time, 
    format_ragas_score, cache_api_response, get_cached_response
)

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ragas-score {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .safety-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .source-document {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Production-ready Medical Knowledge Assistant with RAGAS Evaluation**")
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Query Medical Knowledge", 
        "📊 RAGAS Dashboard", 
        "📄 Document Management", 
        "⚙️ System Monitoring",
        "🧪 Batch Evaluation"
    ])
    
    with tab1:
        query_interface()
    
    with tab2:
        ragas_dashboard()
    
    with tab3:
        document_management()
    
    with tab4:
        system_monitoring()
    
    with tab5:
        batch_evaluation()


def setup_sidebar():
    """Setup sidebar with configuration and status."""
    
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # OpenAI API Key input
    st.sidebar.markdown("### 🔑 OpenAI API Key")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key is required for processing queries. It will be sent securely to the backend."
    )
    
    # Store API key in session state
    st.session_state.openai_api_key = api_key
    
    # API connection status
    api_client = get_api_client()
    
    if api_client.check_connection():
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.info("Make sure FastAPI backend is running on http://localhost:8000")
    
    # API key validation
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API Key to use the system")
    elif not api_key.startswith("sk-"):
        st.sidebar.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
    else:
        st.sidebar.success("✅ API Key provided")
    
    # Query settings
    st.sidebar.markdown("### 🔍 Query Settings")
    
    retrieval_strategy = st.sidebar.selectbox(
        "Retrieval Strategy",
        ["similarity", "hybrid", "medical_focused"],
        index=0,
        help="Strategy for retrieving relevant documents"
    )
    
    max_documents = st.sidebar.slider(
        "Max Documents",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of documents to retrieve"
    )
    
    include_sources = st.sidebar.checkbox(
        "Include Sources",
        value=True,
        help="Include source documents in response"
    )
    
    evaluate_with_ragas = st.sidebar.checkbox(
        "RAGAS Evaluation",
        value=True,
        help="Evaluate response quality with RAGAS metrics"
    )
    
    # Store settings in session state
    st.session_state.query_settings = {
        "retrieval_strategy": retrieval_strategy,
        "max_documents": max_documents,
        "include_sources": include_sources,
        "evaluate_with_ragas": evaluate_with_ragas
    }
    
    # RAGAS thresholds info
    st.sidebar.markdown("### 📊 RAGAS Thresholds")
    st.sidebar.info("""
    **Quality Targets:**
    - Faithfulness: >90%
    - Context Precision: >85%
    - Context Recall: >80%
    - Answer Relevancy: >85%
    """)
    
    # Medical disclaimer
    st.sidebar.markdown("### ⚠️ Medical Disclaimer")
    st.sidebar.warning("""
    This system is designed for healthcare professionals. 
    Always consult qualified healthcare providers for medical decisions.
    """)


def query_interface():
    """Main query interface."""
    
    st.markdown("## 🔍 Ask a Medical Question")
    st.markdown("Enter your medical query below to get AI-powered answers with quality evaluation.")
    
    # Example queries
    with st.expander("📝 Example Queries"):
        examples = [
            "What are the contraindications for aspirin?",
            "What is the recommended dosage of metformin for type 2 diabetes?",
            "What are the symptoms of myocardial infarction?",
            "How is hypertension diagnosed?",
            "What are the side effects of statins?"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"💡 {example}", key=f"example_{i}"):
                st.session_state.query_input = example
    
    # Query input
    query = st.text_area(
        "Enter your medical query:",
        height=100,
        placeholder="e.g., What are the contraindications for aspirin?",
        key="query_input"
    )
    
    # Query button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Get Medical Answer", type="primary", use_container_width=True):
            if query.strip():
                process_query(query)
            else:
                st.warning("Please enter a medical query.")
    
    # Display recent queries
    display_recent_queries()


def process_query(query: str):
    """Process a medical query and display results."""
    
    api_client = get_api_client()
    settings = st.session_state.get("query_settings", {})
    
    # Check if API key is provided
    api_key = st.session_state.get("openai_api_key")
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
        return
    
    # Show processing indicator
    with st.spinner("🔍 Processing your medical query..."):
        try:
            # Make API call
            start_time = time.time()
            response = api_client.query_medical_knowledge(
                query=query,
                openai_api_key=api_key,
                include_sources=settings.get("include_sources", True),
                evaluate_with_ragas=settings.get("evaluate_with_ragas", True),
                retrieval_strategy=settings.get("retrieval_strategy", "similarity"),
                max_documents=settings.get("max_documents", 5)
            )
            
            # Store in session state for history
            if "query_history" not in st.session_state:
                st.session_state.query_history = []
            
            st.session_state.query_history.insert(0, {
                "query": query,
                "response": response,
                "timestamp": datetime.now()
            })
            
            # Keep only last 10 queries
            st.session_state.query_history = st.session_state.query_history[:10]
            
            # Display results
            display_query_results(response)
            
        except Exception as e:
            display_api_error(e)


def display_query_results(response: dict):
    """Display query results with answer, sources, and RAGAS metrics."""
    
    st.markdown("## 📋 Medical Response")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Response Time",
            format_response_time(response.get("total_time", 0)),
            help="Total time to process query"
        )
    
    with col2:
        st.metric(
            "Retrieval Time",
            format_response_time(response.get("retrieval_time", 0)),
            help="Time to retrieve relevant documents"
        )
    
    with col3:
        st.metric(
            "Generation Time",
            format_response_time(response.get("generation_time", 0)),
            help="Time to generate response"
        )
    
    with col4:
        model_used = response.get("model_used", "Unknown")
        st.metric(
            "Model",
            model_used,
            help="Language model used for generation"
        )
    
    # Main answer
    st.markdown("### 🤖 AI Response")
    
    # Safety flags
    safety_flags = response.get("safety_flags", [])
    if safety_flags:
        st.markdown('<div class="safety-warning">', unsafe_allow_html=True)
        st.warning(f"⚠️ Safety flags: {', '.join(safety_flags)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display answer
    answer = response.get("answer", "No answer generated")
    st.markdown(answer)
    
    # RAGAS metrics
    ragas_metrics = response.get("ragas_metrics")
    if ragas_metrics:
        display_ragas_metrics(ragas_metrics)
    
    # Source documents
    sources = response.get("sources")
    if sources:
        display_source_documents(sources)


def display_ragas_metrics(ragas_metrics: dict):
    """Display RAGAS evaluation metrics."""
    
    st.markdown("### 📊 RAGAS Quality Evaluation")
    
    # Overall score and status
    overall_score = ragas_metrics.get("overall_score", 0)
    passes_thresholds = ragas_metrics.get("passes_thresholds", False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Overall Score",
            format_ragas_score(overall_score),
            help="Average of all RAGAS metrics"
        )
    
    with col2:
        status = "✅ Passes Quality Thresholds" if passes_thresholds else "❌ Below Quality Thresholds"
        st.metric(
            "Quality Status",
            status,
            help="Whether response meets quality standards"
        )
    
    # Individual metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        faithfulness = ragas_metrics.get("faithfulness_score", 0)
        st.metric(
            "Faithfulness",
            format_ragas_score(faithfulness),
            help="How well the answer is grounded in context (Target: >90%)"
        )
    
    with col2:
        relevancy = ragas_metrics.get("answer_relevancy_score", 0)
        st.metric(
            "Answer Relevancy",
            format_ragas_score(relevancy),
            help="How well the answer addresses the query (Target: >85%)"
        )
    
    with col3:
        precision = ragas_metrics.get("context_precision_score", 0)
        st.metric(
            "Context Precision",
            format_ragas_score(precision),
            help="Relevance of retrieved contexts (Target: >85%)"
        )
    
    with col4:
        recall = ragas_metrics.get("context_recall_score")
        if recall is not None:
            st.metric(
                "Context Recall",
                format_ragas_score(recall),
                help="Completeness of retrieved information (Target: >80%)"
            )
        else:
            st.metric(
                "Context Recall",
                "N/A",
                help="Requires ground truth for evaluation"
            )
    
    # RAGAS visualization
    create_ragas_chart(ragas_metrics)
    
    # Recommendations
    recommendations = ragas_metrics.get("recommendations", [])
    if recommendations:
        st.markdown("#### 💡 Improvement Recommendations")
        for rec in recommendations:
            st.info(f"• {rec}")


def create_ragas_chart(ragas_metrics: dict):
    """Create a radar chart for RAGAS metrics."""
    
    metrics = []
    scores = []
    
    if ragas_metrics.get("faithfulness_score") is not None:
        metrics.append("Faithfulness")
        scores.append(ragas_metrics["faithfulness_score"] * 100)
    
    if ragas_metrics.get("answer_relevancy_score") is not None:
        metrics.append("Answer Relevancy")
        scores.append(ragas_metrics["answer_relevancy_score"] * 100)
    
    if ragas_metrics.get("context_precision_score") is not None:
        metrics.append("Context Precision")
        scores.append(ragas_metrics["context_precision_score"] * 100)
    
    if ragas_metrics.get("context_recall_score") is not None:
        metrics.append("Context Recall")
        scores.append(ragas_metrics["context_recall_score"] * 100)
    
    if metrics and scores:
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name='RAGAS Scores',
            line_color='rgb(31, 119, 180)'
        ))
        
        # Add threshold lines
        thresholds = [90, 85, 85, 80]  # Faithfulness, Relevancy, Precision, Recall
        fig.add_trace(go.Scatterpolar(
            r=thresholds[:len(metrics)],
            theta=metrics,
            fill='toself',
            name='Target Thresholds',
            line_color='rgb(255, 127, 14)',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="RAGAS Metrics vs. Targets"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_source_documents(sources: list):
    """Display source documents used for response generation."""
    
    st.markdown("### 📚 Source Documents")
    st.markdown(f"Found **{len(sources)}** relevant documents:")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"📄 Source {i}: {source['metadata']['filename']} (Score: {source['relevance_score']:.3f})"):
            
            # Metadata
            metadata = source['metadata']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunk Index", f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
            with col2:
                st.metric("Tokens", metadata.get('chunk_tokens', 0))
            with col3:
                st.metric("Document Type", metadata.get('document_type', 'Unknown'))
            
            # Medical sections
            sections = metadata.get('medical_sections', [])
            if sections:
                st.markdown("**Medical Sections:** " + ", ".join(sections))
            
            # Content
            st.markdown("**Content:**")
            st.markdown(f'<div class="source-document">{source["content"]}</div>', unsafe_allow_html=True)


def display_recent_queries():
    """Display recent query history."""
    
    if "query_history" not in st.session_state or not st.session_state.query_history:
        return
    
    st.markdown("## 📜 Recent Queries")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        with st.expander(f"🕒 {item['timestamp'].strftime('%H:%M:%S')} - {item['query'][:50]}..."):
            
            # Quick metrics
            response = item['response']
            ragas = response.get('ragas_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", format_response_time(response.get('total_time', 0)))
            with col2:
                if ragas:
                    st.metric("Overall Score", format_ragas_score(ragas.get('overall_score', 0)))
            with col3:
                passes = ragas.get('passes_thresholds', False) if ragas else False
                st.metric("Quality", "✅ Pass" if passes else "❌ Fail")
            
            # Re-run button
            if st.button(f"🔄 Re-run Query", key=f"rerun_{i}"):
                st.session_state.query_input = item['query']
                st.rerun()


def ragas_dashboard():
    """RAGAS metrics dashboard."""
    
    st.markdown("## 📊 RAGAS Evaluation Dashboard")
    st.markdown("Monitor system quality and performance with real-time RAGAS metrics.")
    
    # Get recent query history
    history = st.session_state.get("query_history", [])
    
    if not history:
        st.info("No queries yet. Submit some medical queries to see RAGAS metrics here.")
        return
    
    # Filter queries with RAGAS metrics
    ragas_data = []
    for item in history:
        ragas = item['response'].get('ragas_metrics')
        if ragas:
            ragas_data.append({
                'timestamp': item['timestamp'],
                'query': item['query'][:50] + '...',
                'faithfulness': ragas.get('faithfulness_score', 0),
                'relevancy': ragas.get('answer_relevancy_score', 0),
                'precision': ragas.get('context_precision_score', 0),
                'recall': ragas.get('context_recall_score'),
                'overall': ragas.get('overall_score', 0),
                'passes': ragas.get('passes_thresholds', False)
            })
    
    if not ragas_data:
        st.info("No RAGAS evaluations yet. Enable RAGAS evaluation in the sidebar.")
        return
    
    df = pd.DataFrame(ragas_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_faithfulness = df['faithfulness'].mean()
        st.metric(
            "Avg Faithfulness",
            format_ragas_score(avg_faithfulness),
            help="Average faithfulness score"
        )
    
    with col2:
        avg_relevancy = df['relevancy'].mean()
        st.metric(
            "Avg Relevancy",
            format_ragas_score(avg_relevancy),
            help="Average answer relevancy score"
        )
    
    with col3:
        avg_precision = df['precision'].mean()
        st.metric(
            "Avg Precision",
            format_ragas_score(avg_precision),
            help="Average context precision score"
        )
    
    with col4:
        pass_rate = df['passes'].mean()
        st.metric(
            "Pass Rate",
            f"{pass_rate * 100:.1f}%",
            help="Percentage of queries passing quality thresholds"
        )
    
    # Time series chart
    st.markdown("### 📈 RAGAS Metrics Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['faithfulness'] * 100,
        mode='lines+markers',
        name='Faithfulness',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['relevancy'] * 100,
        mode='lines+markers',
        name='Answer Relevancy',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['precision'] * 100,
        mode='lines+markers',
        name='Context Precision',
        line=dict(color='orange')
    ))
    
    # Add threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color="blue", annotation_text="Faithfulness Target (90%)")
    fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Relevancy Target (85%)")
    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Precision Target (85%)")
    
    fig.update_layout(
        title="RAGAS Metrics Timeline",
        xaxis_title="Time",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("### 📋 Detailed RAGAS Results")
    
    # Format dataframe for display
    display_df = df.copy()
    display_df['faithfulness'] = display_df['faithfulness'].apply(lambda x: f"{x*100:.1f}%")
    display_df['relevancy'] = display_df['relevancy'].apply(lambda x: f"{x*100:.1f}%")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x*100:.1f}%")
    display_df['overall'] = display_df['overall'].apply(lambda x: f"{x*100:.1f}%")
    display_df['passes'] = display_df['passes'].apply(lambda x: "✅" if x else "❌")
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
    
    st.dataframe(
        display_df[['timestamp', 'query', 'faithfulness', 'relevancy', 'precision', 'overall', 'passes']],
        use_container_width=True
    )


def document_management():
    """Document management interface."""
    
    st.markdown("## 📄 Document Management")
    st.markdown("Upload and manage medical documents for the RAG pipeline.")
    
    # Document upload
    st.markdown("### 📤 Upload Medical Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload medical literature, clinical guidelines, or drug information"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            document_type = st.selectbox(
                "Document Type",
                ["medical", "clinical_guideline", "drug_information", "clinical_procedure"],
                help="Type of medical document"
            )
        
        with col2:
            if st.button("📤 Upload Document", type="primary"):
                upload_document(uploaded_file, document_type)
    
    # Document statistics
    st.markdown("### 📊 Document Collection Statistics")
    
    try:
        api_client = get_api_client()
        stats = api_client.get_document_stats()
        
        if stats and 'document_stats' in stats:
            doc_stats = stats['document_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Documents",
                    doc_stats.get('total_documents', 0),
                    help="Total number of document chunks"
                )
            
            with col2:
                st.metric(
                    "Total Tokens",
                    f"{doc_stats.get('estimated_total_tokens', 0):,.0f}",
                    help="Estimated total tokens in collection"
                )
            
            with col3:
                embedding_model = doc_stats.get('embedding_model', 'Unknown')
                st.metric(
                    "Embedding Model",
                    embedding_model,
                    help="Model used for document embeddings"
                )
            
            with col4:
                last_updated = doc_stats.get('last_updated', 'Unknown')
                if last_updated != 'Unknown':
                    last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                st.metric(
                    "Last Updated",
                    last_updated,
                    help="When collection was last updated"
                )
            
            # Document types breakdown
            doc_types = doc_stats.get('document_types', {})
            if doc_types:
                st.markdown("#### 📚 Document Types")
                
                fig = px.pie(
                    values=list(doc_types.values()),
                    names=list(doc_types.keys()),
                    title="Document Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading document statistics: {e}")


def upload_document(uploaded_file, document_type: str):
    """Handle document upload."""
    
    with st.spinner("📤 Uploading and processing document..."):
        try:
            api_client = get_api_client()
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Upload document
            response = api_client.upload_document(
                file_content=file_content,
                filename=uploaded_file.name,
                document_type=document_type
            )
            
            if response.get('success'):
                st.success(f"✅ {response.get('message')}")
                
                # Display processing results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chunks Created", response.get('chunks_created', 0))
                
                with col2:
                    st.metric("Total Tokens", response.get('total_tokens', 0))
                
                with col3:
                    processing_time = response.get('processing_time', 0)
                    st.metric("Processing Time", format_response_time(processing_time))
            
            else:
                st.error(f"❌ Upload failed: {response.get('message')}")
        
        except Exception as e:
            display_api_error(e)


def system_monitoring():
    """System monitoring and health dashboard."""
    
    st.markdown("## ⚙️ System Monitoring")
    st.markdown("Monitor system health, performance, and configuration.")
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        # Clear cached responses
        if "api_cache" in st.session_state:
            st.session_state.api_cache.clear()
    
    try:
        api_client = get_api_client()
        
        # Health status
        st.markdown("### 🏥 System Health")
        
        health = api_client.get_health_status()
        
        overall_status = health.get('status', 'unknown')
        status_color = {
            'healthy': '🟢',
            'degraded': '🟡',
            'unhealthy': '🔴'
        }.get(overall_status, '⚪')
        
        st.markdown(f"**Overall Status:** {status_color} {overall_status.title()}")
        
        # Component health
        components = health.get('components', {})
        
        col1, col2 = st.columns(2)
        
        for i, (component, status) in enumerate(components.items()):
            target_col = col1 if i % 2 == 0 else col2
            
            with target_col:
                component_status = status.get('status', 'unknown') if isinstance(status, dict) else 'unknown'
                component_color = {
                    'healthy': '🟢',
                    'degraded': '🟡',
                    'unhealthy': '🔴'
                }.get(component_status, '⚪')
                
                st.markdown(f"**{component.replace('_', ' ').title()}:** {component_color} {component_status.title()}")
        
        # System metrics
        st.markdown("### 📊 System Metrics")
        
        metrics = api_client.get_metrics()
        
        # Performance metrics
        system_metrics = metrics.get('system_metrics', {})
        vector_stats = metrics.get('vector_store_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = vector_stats.get('total_documents', 0)
            st.metric("Documents", f"{total_docs:,}")
        
        with col2:
            total_tokens = vector_stats.get('estimated_total_tokens', 0)
            st.metric("Total Tokens", f"{total_tokens:,.0f}")
        
        with col3:
            collection_name = vector_stats.get('collection_name', 'N/A')
            st.metric("Collection", collection_name)
        
        with col4:
            embedding_model = vector_stats.get('embedding_model', 'N/A')
            st.metric("Embedding Model", embedding_model)
        
        # Configuration
        st.markdown("### ⚙️ Configuration")
        
        config = api_client.get_configuration()
        
        # RAGAS thresholds
        st.markdown("#### 📊 RAGAS Thresholds")
        thresholds = config.get('ragas_thresholds', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Faithfulness", f"{thresholds.get('faithfulness', 0)*100:.0f}%")
        
        with col2:
            st.metric("Answer Relevancy", f"{thresholds.get('answer_relevancy', 0)*100:.0f}%")
        
        with col3:
            st.metric("Context Precision", f"{thresholds.get('context_precision', 0)*100:.0f}%")
        
        with col4:
            st.metric("Context Recall", f"{thresholds.get('context_recall', 0)*100:.0f}%")
        
        # Model settings
        st.markdown("#### 🤖 Model Settings")
        model_settings = config.get('model_settings', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", model_settings.get('model', 'N/A'))
        
        with col2:
            st.metric("Max Tokens", model_settings.get('max_tokens', 0))
        
        with col3:
            st.metric("Temperature", model_settings.get('temperature', 0))
        
        # Safety settings
        st.markdown("#### 🛡️ Safety Settings")
        safety_settings = config.get('safety_settings', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filtering = safety_settings.get('filtering_enabled', False)
            st.metric("Safety Filtering", "✅ Enabled" if filtering else "❌ Disabled")
        
        with col2:
            content_detection = safety_settings.get('harmful_content_detection', False)
            st.metric("Content Detection", "✅ Enabled" if content_detection else "❌ Disabled")
        
        with col3:
            validation = safety_settings.get('response_validation', False)
            st.metric("Response Validation", "✅ Enabled" if validation else "❌ Disabled")
    
    except Exception as e:
        display_api_error(e)


def batch_evaluation():
    """Batch evaluation interface for RAGAS testing."""
    
    st.markdown("## 🧪 Batch Evaluation")
    st.markdown("Evaluate multiple query-answer pairs with RAGAS metrics for system testing.")
    
    # Load sample data
    sample_file = "data/evaluation/medical_qa_pairs.json"
    
    try:
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        
        st.markdown("### 📋 Sample Medical Q&A Pairs")
        st.info(f"Loaded {len(sample_data)} sample medical Q&A pairs for evaluation.")
        
        # Select evaluation subset
        num_samples = st.slider(
            "Number of samples to evaluate",
            min_value=1,
            max_value=len(sample_data),
            value=min(5, len(sample_data)),
            help="Select how many samples to include in batch evaluation"
        )
        
        selected_samples = sample_data[:num_samples]
        
        # Display selected samples
        with st.expander("📝 Selected Samples"):
            for i, sample in enumerate(selected_samples, 1):
                st.markdown(f"**{i}. {sample['query']}**")
                st.markdown(f"*Ground Truth:* {sample['ground_truth'][:100]}...")
                st.markdown("---")
        
        # Run batch evaluation
        if st.button("🧪 Run Batch Evaluation", type="primary"):
            run_batch_evaluation(selected_samples)
    
    except FileNotFoundError:
        st.warning("Sample evaluation data not found. Please ensure medical_qa_pairs.json exists.")
        
        # Manual input option
        st.markdown("### ✍️ Manual Input")
        st.markdown("Enter your own query-answer pairs for evaluation:")
        
        manual_queries = st.text_area(
            "Queries (one per line):",
            height=100,
            placeholder="What are the contraindications for aspirin?\nWhat is the recommended dosage of metformin?"
        )
        
        manual_answers = st.text_area(
            "Expected Answers (one per line, corresponding to queries above):",
            height=100,
            placeholder="Aspirin is contraindicated in...\nThe initial dose of metformin is..."
        )
        
        if st.button("🧪 Evaluate Manual Input"):
            if manual_queries and manual_answers:
                queries = [q.strip() for q in manual_queries.split('\n') if q.strip()]
                answers = [a.strip() for a in manual_answers.split('\n') if a.strip()]
                
                if len(queries) == len(answers):
                    manual_samples = [
                        {"query": q, "ground_truth": a}
                        for q, a in zip(queries, answers)
                    ]
                    run_batch_evaluation(manual_samples)
                else:
                    st.error("Number of queries and answers must match.")


def run_batch_evaluation(samples: list):
    """Run batch evaluation on selected samples."""
    
    # Check if API key is provided
    api_key = st.session_state.get("openai_api_key")
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
        return
    
    with st.spinner("🧪 Running batch evaluation..."):
        try:
            api_client = get_api_client()
            
            # First, get AI answers for all queries
            st.write("Step 1: Generating AI responses...")
            progress_bar = st.progress(0)
            
            queries = []
            ai_answers = []
            contexts = []
            ground_truths = []
            
            for i, sample in enumerate(samples):
                # Get AI response
                response = api_client.query_medical_knowledge(
                    query=sample['query'],
                    openai_api_key=api_key,
                    include_sources=True,
                    evaluate_with_ragas=False  # We'll do batch evaluation separately
                )
                
                queries.append(sample['query'])
                ai_answers.append(response['answer'])
                ground_truths.append(sample['ground_truth'])
                
                # Extract contexts from sources
                sources = response.get('sources', [])
                context_list = [source['content'] for source in sources]
                contexts.append(context_list)
                
                progress_bar.progress((i + 1) / len(samples))
            
            # Run RAGAS evaluation
            st.write("Step 2: Running RAGAS evaluation...")
            
            eval_response = api_client.evaluate_with_ragas(
                queries=queries,
                answers=ai_answers,
                contexts=contexts,
                ground_truths=ground_truths,
                openai_api_key=api_key
            )
            
            # Display results
            st.success("✅ Batch evaluation completed!")
            
            # Summary metrics
            st.markdown("### 📊 Evaluation Summary")
            
            summary = eval_response.get('summary', {})
            avg_scores = eval_response.get('average_scores', {})
            pass_rates = eval_response.get('pass_rates', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Faithfulness",
                    format_ragas_score(avg_scores.get('faithfulness', 0))
                )
            
            with col2:
                st.metric(
                    "Avg Relevancy",
                    format_ragas_score(avg_scores.get('answer_relevancy', 0))
                )
            
            with col3:
                st.metric(
                    "Avg Precision",
                    format_ragas_score(avg_scores.get('context_precision', 0))
                )
            
            with col4:
                st.metric(
                    "Overall Pass Rate",
                    f"{pass_rates.get('overall', 0)*100:.1f}%"
                )
            
            # Detailed results
            st.markdown("### 📋 Detailed Results")
            
            individual_results = eval_response.get('individual_results', [])
            
            results_data = []
            for i, result in enumerate(individual_results):
                results_data.append({
                    'Query': queries[i][:50] + '...',
                    'Faithfulness': f"{result['faithfulness_score']*100:.1f}%",
                    'Relevancy': f"{result['answer_relevancy_score']*100:.1f}%",
                    'Precision': f"{result['context_precision_score']*100:.1f}%",
                    'Overall': f"{result['overall_score']*100:.1f}%",
                    'Pass': "✅" if result['passes_thresholds'] else "❌"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            st.markdown("### 📈 Results Visualization")
            
            # Create comparison chart
            metrics_df = pd.DataFrame({
                'Query': [f"Q{i+1}" for i in range(len(individual_results))],
                'Faithfulness': [r['faithfulness_score']*100 for r in individual_results],
                'Relevancy': [r['answer_relevancy_score']*100 for r in individual_results],
                'Precision': [r['context_precision_score']*100 for r in individual_results]
            })
            
            fig = px.bar(
                metrics_df.melt(id_vars=['Query'], var_name='Metric', value_name='Score'),
                x='Query',
                y='Score',
                color='Metric',
                title="RAGAS Scores by Query",
                barmode='group'
            )
            
            fig.add_hline(y=90, line_dash="dash", line_color="blue", annotation_text="Faithfulness Target")
            fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Relevancy/Precision Target")
            
            fig.update_layout(yaxis=dict(range=[0, 100]))
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            display_api_error(e)


if __name__ == "__main__":
    main() 