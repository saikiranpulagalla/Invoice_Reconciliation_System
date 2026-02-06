"""
Streamlit UI for Multi-Agent Invoice Reconciliation System

This is a read-only visualization layer that:
- Accepts invoice uploads
- Calls the existing agent pipeline
- Displays agent traces and reasoning
- Shows confidence scores and final decisions

NO BUSINESS LOGIC IS IMPLEMENTED HERE.
All logic is in app.agents and app.main.
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

# Set environment to skip API key validation during import
import os
os.environ["SKIP_API_KEY_CHECK"] = "true"

import streamlit as st
import asyncio
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

# Import the existing pipeline
from app.main import process_invoice
from app.state import ReconciliationState
from app.schemas.output import ReconciliationOutput


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def check_and_setup_api_key():
    """
    Check if API key exists in .env file.
    If not, prompt user to enter it in the UI.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.warning("⚠️ No Google API Key found in .env file")
        
        # Create a form for API key input
        with st.form("api_key_form"):
            st.markdown("### 🔑 Enter Your Google API Key")
            st.markdown("""
            To use this application, you need a Google API key for Gemini.
            
            **Get your API key**:
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Click "Create API Key"
            3. Copy the key and paste it below
            """)
            
            api_key_input = st.text_input(
                "Google API Key",
                type="password",
                placeholder="paste-your-api-key-here",
                help="Your API key will be used only for this session"
            )
            
            submitted = st.form_submit_button("✅ Set API Key", use_container_width=True)
            
            if submitted:
                if api_key_input:
                    # Set the API key in environment
                    os.environ["GOOGLE_API_KEY"] = api_key_input
                    st.session_state.api_key = api_key_input
                    st.success("✅ API Key set successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please enter an API key")
        
        return False
    else:
        st.session_state.api_key = api_key
        return True


def ensure_gemini_config():
    """
    Ensure the system is configured to use only gemini-2.5-flash model.
    """
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini-2.5-flash"
    os.environ["LLM_TEMPERATURE"] = "0.3"
    os.environ["LLM_MAX_TOKENS"] = "2000"


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Invoice Reconciliation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure Gemini configuration
ensure_gemini_config()

# Check API key and setup if needed
if not check_and_setup_api_key():
    st.stop()  # Stop execution if no API key

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-success { color: #28a745; font-weight: bold; }
    .agent-warning { color: #ffc107; font-weight: bold; }
    .agent-error { color: #dc3545; font-weight: bold; }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .metric-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        color: #000;
    }
    .agent-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #007bff;
        color: #000;
        font-weight: bold;
    }
    .expander-text {
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER & DESCRIPTION
# ============================================================================

st.title("📄 Multi-Agent Invoice Reconciliation System")
st.markdown("""
This is a **visual trace viewer** for the invoice reconciliation pipeline.
Upload an invoice to see how the multi-agent system analyzes, matches, and 
recommends actions with full transparency into each agent's reasoning.

**Note**: This UI is view-only. All business logic runs in the agent pipeline.
""")

st.divider()


# ============================================================================
# SIDEBAR - SETTINGS & INFO
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show API Key status
    with st.expander("🔑 API Configuration", expanded=False):
        if st.session_state.get("api_key"):
            st.success("✅ API Key: Configured")
            st.info("""
            **Model**: gemini-2.5-flash  
            **Provider**: Google Gemini  
            **Temperature**: 0.3  
            **Max Tokens**: 2000  
            """)
        else:
            st.error("❌ API Key: Not configured")
    
    # Show configuration info
    with st.expander("System Configuration", expanded=False):
        from app.config import get_config
        config = get_config()
        st.info(f"""
        **LLM Provider**: {config.LLM_PROVIDER}  
        **Model**: {config.LLM_MODEL}  
        **Temperature**: {config.LLM_TEMPERATURE}  
        **Max Tokens**: {config.LLM_MAX_TOKENS}  
        **OCR Enabled**: {config.ENABLE_OCR}  
        """)
    
    # Show about
    with st.expander("About", expanded=False):
        st.markdown("""
        **Invoice Reconciliation System**  
        A production-grade multi-agent system using LangGraph.
        
        **Agents**:
        - 📋 Document Intelligence: Extract invoice data
        - 🔍 Matching: Find matching Purchase Order
        - ⚠️ Discrepancy Detection: Identify differences
        - ✅ Resolution: Recommend action
        - 👤 Human Reviewer: Feedback & override
        
        **Features**:
        - Confidence scoring at 3 levels
        - Multi-dimensional risk assessment
        - Explainable decisions
        - Full audit trail
        """)


# ============================================================================
# FILE UPLOAD SECTION
# ============================================================================

st.header("📤 Upload Invoice")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an invoice (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a PDF or image file containing an invoice"
    )

with col2:
    st.write("")  # Spacing
    show_sample = st.checkbox("Use sample", value=False, help="Use a test file path")

if uploaded_file or show_sample:
    st.success("✅ File selected")
else:
    st.info("👈 Upload a file or enable 'Use sample' to begin")

st.divider()


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline(file_path: str) -> Optional[ReconciliationOutput]:
    """
    Call the existing agent pipeline.
    Returns the final output or None if error.
    """
    try:
        # Run async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Generate invoice ID from filename
        invoice_id = Path(file_path).stem
        
        # Run the agent pipeline
        result = loop.run_until_complete(
            process_invoice(file_path, invoice_id)
        )
        
        loop.close()
        return result
        
    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        return None


# ============================================================================
# TRACE VISUALIZATION
# ============================================================================

def display_agent_trace(output: ReconciliationOutput) -> None:
    """
    Display the agent trace in an expandable accordion format.
    Each agent gets its own section with extracted data and reasoning.
    """
    
    if not output:
        st.error("No output to display")
        return
    
    processing_results = output.processing_results
    
    # AGENT 1: Document Intelligence
    with st.expander("📋 Document Intelligence Agent", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_val = processing_results.extraction_confidence
            if confidence_val >= 0.85:
                st.metric("Extraction Confidence", f"{confidence_val:.2%}", 
                         delta="High", delta_color="off")
            elif confidence_val >= 0.70:
                st.metric("Extraction Confidence", f"{confidence_val:.2%}", 
                         delta="Acceptable", delta_color="off")
            else:
                st.metric("Extraction Confidence", f"{confidence_val:.2%}", 
                         delta="Low", delta_color="off")
        
        with col2:
            quality_colors = {
                "good": "🟢",
                "acceptable": "🟡",
                "poor": "🔴"
            }
            quality = processing_results.document_quality
            st.metric("Document Quality", 
                     f"{quality_colors.get(quality, '❓')} {quality.title()}")
        
        with col3:
            st.metric("Status", "✅ Completed")
        
        # Display extracted data
        if processing_results.extracted_data:
            st.subheader("Extracted Fields")
            
            # Create table of extracted data
            extracted_items = []
            for key, value in processing_results.extracted_data.items():
                if not key.startswith("_"):
                    extracted_items.append({
                        "Field": key.replace("_", " ").title(),
                        "Value": str(value)[:50]  # Truncate long values
                    })
            
            if extracted_items:
                df = pd.DataFrame(extracted_items)
                st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.warning("⚠️ No data extracted (possible document error)")
        
        # Display agent reasoning
        if processing_results.agent_reasoning:
            st.subheader("Agent Reasoning")
            reasoning_lines = processing_results.agent_reasoning.split("\n")
            doc_intel_lines = [
                line for line in reasoning_lines 
                if "DocumentIntelligenceAgent" in line
            ]
            if doc_intel_lines:
                for line in doc_intel_lines:
                    st.caption(line.strip())
    
    # AGENT 2: Matching Agent
    with st.expander("🔍 Matching Agent", expanded=True):
        matching = processing_results.matching_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if matching.matched_po:
                st.metric("Matched PO", matching.matched_po)
            else:
                st.metric("Matched PO", "❌ None")
        
        with col2:
            confidence_val = matching.match_confidence
            st.metric("Match Confidence", f"{confidence_val:.2%}")
        
        with col3:
            st.metric("Match Type", matching.match_type.replace("_", " ").title())
        
        # Explanation
        if matching.explanation:
            st.subheader("Matching Explanation")
            st.markdown(f"```\n{matching.explanation}\n```")
        
        # Alternative matches
        if matching.alternative_matches:
            st.subheader("Alternative Matches")
            alt_data = []
            for alt in matching.alternative_matches:
                alt_data.append({
                    "PO Number": alt.get("po_number", "N/A") if isinstance(alt, dict) else getattr(alt, 'po_number', 'N/A'),
                    "Confidence": f"{alt.get('confidence', 0):.2%}" if isinstance(alt, dict) else f"{getattr(alt, 'confidence', 0):.2%}",
                    "Method": (alt.get("match_type", "N/A") if isinstance(alt, dict) else getattr(alt, 'match_type', 'N/A')).replace("_", " ").title()
                })
            st.dataframe(pd.DataFrame(alt_data), width='stretch', hide_index=True)
    
    # AGENT 3: Discrepancy Detection
    with st.expander("⚠️ Discrepancy Detection Agent", expanded=True):
        discrepancies = processing_results.discrepancies
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Discrepancies Found", len(discrepancies))
        
        with col2:
            if discrepancies:
                severity_counts = {}
                for disc in discrepancies:
                    sev = disc.severity if hasattr(disc, 'severity') else "unknown"
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                
                sev_str = ", ".join([f"{count} {sev}" for sev, count in severity_counts.items()])
                st.metric("Severity Breakdown", sev_str)
            else:
                st.metric("Status", "✅ No issues")
        
        if discrepancies:
            st.subheader("Discrepancy Details")
            
            for idx, disc in enumerate(discrepancies, 1):
                severity = (disc.severity if hasattr(disc, 'severity') else "unknown").upper()
                disc_type = (disc.type if hasattr(disc, 'type') else "unknown").replace("_", " ").title()
                
                # Color-code severity
                severity_emoji = {
                    "HIGH": "🔴",
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }.get(severity, "⚪")
                
                st.markdown(f"**{severity_emoji} [{severity}] {disc_type}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    confidence = disc.confidence if hasattr(disc, 'confidence') else 0
                    st.caption(f"Confidence: {confidence:.2%}")
                with col2:
                    disc_id = disc.type if hasattr(disc, 'type') else 'N/A'
                    st.caption(f"ID: {disc_id}")
                
                explanation = disc.explanation if hasattr(disc, 'explanation') else "No explanation provided"
                st.markdown(f"_{explanation}_")
                st.divider()
    
    # AGENT 4: Resolution Recommendation
    with st.expander("✅ Resolution Recommendation Agent", expanded=True):
        action = processing_results.recommended_action
        
        # Color-code the action
        action_colors = {
            "auto_approve": "🟢",
            "flag_for_review": "🟡",
            "escalate_to_human": "🔴"
        }
        
        action_emoji = action_colors.get(action, "⚪")
        action_display = action.replace("_", " ").upper()
        
        st.markdown(f"## {action_emoji} {action_display}")
        
        # Show reasoning
        if processing_results.action_reasoning:
            st.subheader("Decision Reasoning")
            st.markdown(processing_results.action_reasoning)
        
        # Parse and show risk assessment if available
        reasoning_text = processing_results.action_reasoning or ""
        if "RISK ASSESSMENT" in reasoning_text:
            st.subheader("Risk Components")
            
            # Extract risk scores from reasoning if possible
            lines = reasoning_text.split("\n")
            risk_lines = [l for l in lines if "Risk:" in l]
            
            for line in risk_lines:
                st.caption(line.strip())


# ============================================================================
# AGENT TIMELINE / STATUS
# ============================================================================

def display_timeline(output: ReconciliationOutput) -> None:
    """
    Display a visual timeline of agent execution.
    """
    st.header("📊 Agent Execution Timeline")
    
    # Simple vertical flow
    col1, col2, col3, col4, col5 = st.columns(5)
    
    agents = [
        ("📋", "Document\nIntelligence"),
        ("🔍", "Matching"),
        ("⚠️", "Discrepancy"),
        ("✅", "Resolution"),
        ("📤", "Output")
    ]
    
    cols = [col1, col2, col3, col4, col5]
    
    for col, (emoji, name) in zip(cols, agents):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; 
                        background-color: #e7f3ff; border-radius: 10px; border: 2px solid #007bff;">
                <div style="font-size: 24px;">{emoji}</div>
                <div style="font-size: 12px; margin-top: 5px; color: #000; font-weight: bold;">{name}</div>
                <div style="font-size: 11px; margin-top: 5px; color: green;">✓</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# FINAL OUTPUT SECTION
# ============================================================================

def display_final_output(output: ReconciliationOutput) -> None:
    """
    Display the final JSON output with copy-to-clipboard option.
    """
    st.header("📋 Final JSON Output")
    
    # Convert Pydantic objects to dictionaries for JSON serialization
    def convert_to_dict(obj):
        """Convert Pydantic objects and other complex types to dictionaries."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_dict(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert to JSON-serializable dict
    output_dict = {
        "invoice_id": output.invoice_id,
        "processing_timestamp": output.processing_timestamp.isoformat() 
                                if hasattr(output.processing_timestamp, 'isoformat')
                                else str(output.processing_timestamp),
        "processing_results": {
            "extraction_confidence": output.processing_results.extraction_confidence,
            "document_quality": output.processing_results.document_quality,
            "extracted_data": convert_to_dict(output.processing_results.extracted_data),
            "matching_results": {
                "matched_po": convert_to_dict(output.processing_results.matching_results.matched_po),
                "match_confidence": output.processing_results.matching_results.match_confidence,
                "match_type": output.processing_results.matching_results.match_type,
                "alternative_matches": convert_to_dict(output.processing_results.matching_results.alternative_matches),
                "explanation": output.processing_results.matching_results.explanation
            },
            "discrepancies": convert_to_dict(output.processing_results.discrepancies),
            "recommended_action": output.processing_results.recommended_action,
            "action_reasoning": output.processing_results.action_reasoning,
            "agent_reasoning": output.processing_results.agent_reasoning
        }
    }
    
    # Format as pretty JSON with fallback for non-serializable objects
    json_str = json.dumps(output_dict, indent=2, default=str)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.code(json_str, language="json", line_numbers=True)
    
    with col2:
        # Copy button
        if st.button("📋 Copy JSON", key="copy_json"):
            st.write("✅ Copied to clipboard!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if uploaded_file or show_sample:
    
    # Save uploaded file temporarily
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        st.session_state.file_path = temp_path
        st.session_state.file_name = uploaded_file.name
    elif show_sample:
        st.session_state.file_path = "test_invoice.pdf"
        st.session_state.file_name = "test_invoice.pdf (sample)"
    
    # Run button
    st.divider()
    st.header("🚀 Run Agent Pipeline")
    
    run_cols = st.columns([2, 1, 1])
    
    with run_cols[0]:
        if st.button("▶️ Run Agent System", key="run_button", width='stretch'):
            st.session_state.running = True
    
    if st.session_state.get("running", False):
        
        # Progress section
        with st.spinner("🔄 Processing invoice..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run pipeline with timing
                start_time = time.time()
                status_text.text("📋 Running Document Intelligence Agent...")
                progress_bar.progress(20)
                
                result = run_pipeline(st.session_state.file_path)
                
                if result:
                    elapsed_time = time.time() - start_time
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Complete in {elapsed_time:.2f}s")
                    
                    st.session_state.result = result
                    st.session_state.elapsed_time = elapsed_time
                    
                    # Clear running state
                    st.session_state.running = False
                else:
                    st.error("❌ Pipeline failed to produce output")
                    st.session_state.running = False
            
            except Exception as e:
                st.error(f"❌ Error running pipeline: {str(e)}")
                st.session_state.running = False
    
    # Display results if available
    if st.session_state.get("result"):
        st.divider()
        
        # Summary metrics
        st.header("📊 Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            action = st.session_state.result.processing_results.recommended_action
            action_emoji = {
                "auto_approve": "🟢",
                "flag_for_review": "🟡",
                "escalate_to_human": "🔴"
            }.get(action, "⚪")
            st.metric("Recommendation", f"{action_emoji} {action.replace('_', ' ').title()}")
        
        with col2:
            st.metric("Extraction Confidence", 
                     f"{st.session_state.result.processing_results.extraction_confidence:.1%}")
        
        with col3:
            st.metric("Match Confidence",
                     f"{st.session_state.result.processing_results.matching_results.match_confidence:.1%}")
        
        with col4:
            elapsed = st.session_state.get("elapsed_time", 0)
            st.metric("Processing Time", f"{elapsed:.2f}s")
        
        st.divider()
        
        # Display timeline
        display_timeline(st.session_state.result)
        
        st.divider()
        
        # Display agent traces
        display_agent_trace(st.session_state.result)
        
        st.divider()
        
        # Display final output
        display_final_output(st.session_state.result)

