import streamlit as st
import asyncio
import json
import os
import time
from datetime import datetime

# Load environment settings
from dotenv import load_dotenv

from src.pipelines.agentic_hf_pipeline import run_agent

load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="Restaurant Search Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with chat-like interface
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 10rem;  /* Add padding to make room for fixed input */
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #D72638;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #D72638;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #333;
    }
    .result-box {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sql-query {
        background-color: #f7f7f7;
        border-left: 4px solid #D72638;
        padding: 0.5rem;
        font-family: monospace;
        color: #222;
        overflow-x: auto;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        max-width: 80%;
    }
    .user-message {
        background-color: #e9f5ff;
        border-left: 4px solid #1890ff;
        color: #000;
        margin-left: auto;
        margin-right: 0;
    }
    .bot-message {
        background-color: #e8f8e4;
        border-left: 4px solid #52c41a;
        color: #000;
        margin-left: 0;
        margin-right: auto;
    }
    .bot-message-content {
        display: flex;
        flex-direction: column;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .search-decision {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .decision-box {
        flex: 1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        text-align: center;
        font-weight: 500;
        color: #000;
    }
    .sql-decision {
        background-color: #e9f5ff;
        border: 1px solid #b3d9ff;
    }
    .vector-decision {
        background-color: #e8f8e4;
        border: 1px solid #b6e8b0;
    }
    .chat-history-decision {
        background-color: #fff0e8;
        border: 1px solid #ffcbb3;
    }
    .conversational-decision {
        background-color: #f8e4f8;
        border: 1px solid #e0b0e0;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 500px);
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 5rem;
        margin-top: 0.5rem;
        background-color: #0E1117;
        border-radius: 0px;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem 4rem 1rem 4rem;
        background-color: white;
        border-top: 1px solid #ddd;
        z-index: 1000;
        display: flex;
        align-items: center;
    }
    .chat-input {
        flex-grow: 1;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #ddd;
    }
    .send-button {
        margin-left: 10px;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #25D366;
        color: white;
        border: none;
        cursor: pointer;
    }
    .stButton>button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        background-color: #25D366 !important;
        color: white !important;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #999;
        text-align: right;
        margin-top: 0.2rem;
    }
    /* Hide default header and footer */
    header {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    /* Context indicator styling */
    .context-indicator {
        color: #52c41a;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 5px;
        background-color: rgba(82, 196, 26, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(82, 196, 26, 0.3);
        display: inline-block;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper functions for displaying search details
def display_search_decision(search_decision):
    """Display the search decision in a visual format"""
    st.markdown("<div class='sub-header'>Search Strategy</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sql_used = "✅" if search_decision.get("sql") == 1 else "❌"
        st.markdown(
            f"""
            <div class='decision-box sql-decision'>
                <h3>SQL Database Search</h3>
                <div style='font-size: 2rem;'>{sql_used}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        pinecone_used = "✅" if search_decision.get("pinecone") == 1 else "❌"
        st.markdown(
            f"""
            <div class='decision-box vector-decision'>
                <h3>Vector Search (Pinecone)</h3>
                <div style='font-size: 2rem;'>{pinecone_used}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        chat_history_used = "✅" if search_decision.get("chat_history_lookup") == 1 else "❌"
        st.markdown(
            f"""
            <div class='decision-box chat-history-decision'>
                <h3>Chat History Lookup</h3>
                <div style='font-size: 2rem;'>{chat_history_used}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    if "explanation" in search_decision:
        st.markdown(f"**Explanation:** {search_decision['explanation']}")
        
    # Display extracted context if available
    if "history_context" in search_decision and search_decision["history_context"]:
        st.markdown(f"**Extracted from chat history:** {search_decision['history_context']}")

def display_sql_queries(sql_queries):
    """Display SQL queries in a formatted way"""
    st.markdown("<div class='sub-header'>SQL Queries</div>", unsafe_allow_html=True)
    
    if sql_queries:
        for i, query in enumerate(sql_queries):
            st.markdown(f"**Query {i+1}:**")
            st.code(query, language="sql")
    else:
        st.info("No SQL queries were executed.")


def display_pinecone_queries(pinecone_queries):
    """Display Pinecone queries in a formatted way"""
    st.markdown("<div class='sub-header'>Vector Search Queries</div>", unsafe_allow_html=True)

    if pinecone_queries:
        for i, query_params in enumerate(pinecone_queries):
            with st.container():
                st.markdown(f"**Vector Search {i+1}:**")
                st.write("**Query Text:**", query_params.get("query_text", "N/A"))
                
                st.write("**Metadata Filters:**")
                filters = query_params.get("metadata_filters", {})
                if filters:
                    st.json(filters)
                else:
                    st.text("No metadata filters applied.")
    else:
        st.info("No vector searches were executed.")


def process_query(query):
    """Process the user query using the agentic pipeline with chat history"""
    with st.spinner('Searching for answers...'):
        # Include up to 5 recent messages from chat history (excluding the current query)
        recent_messages = []
        if len(st.session_state.messages) > 0:
            # Get the last 10 messages (5 exchanges) or fewer if there aren't that many yet
            history_count = min(10, len(st.session_state.messages))  # 5 exchanges = 10 messages (user/bot pairs)
            chat_messages = st.session_state.messages[-history_count:]
            
            # Create simplified versions of each message for the agent
            for msg in chat_messages:
                # Create a simplified copy with only essential fields
                simple_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", "")
                }
                recent_messages.append(simple_msg)
            
            # Log the history being used
            history_exchanges = history_count // 2
            if history_exchanges > 0:
                print(f"Using {history_exchanges} previous exchanges ({len(recent_messages)} messages) for context")
            
        # Only include chat history context if we have previous messages
        if recent_messages:
            # Format chat history as JSON
            chat_history_json = json.dumps(recent_messages)
            enhanced_query = f"{query}\nContext from previous conversation: {chat_history_json}"
        else:
            enhanced_query = query
        
        # Run the agent on the query with chat history context
        result = asyncio.run(run_agent(enhanced_query))
        return result

# Sidebar content
st.sidebar.markdown("## Restaurant Search Assistant")
st.sidebar.markdown("Ask questions about restaurants and their menus.")

# Display chat history status in sidebar
if len(st.session_state.messages) > 0:
    message_pairs = len(st.session_state.messages) // 2
    st.sidebar.markdown(f"**Conversation History:** {message_pairs} exchanges")
    st.sidebar.markdown("Your follow-up questions will use context from previous messages.")
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Create a clear button with custom styling
    st.sidebar.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        background-color: #f0f0f0;
        color: #D72638;
        border: 1px solid #ddd;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: 500;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #e6e6e6;
        border-color: #ccc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clear button (centered)
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        clear_button = st.button("Clear Conversation")
    
    if clear_button:
        st.session_state.messages = []
        st.rerun()
else:
    st.sidebar.markdown("**Conversation History:** None")
    st.sidebar.markdown("Start a conversation to enable contextual follow-up questions.")

# Main App UI
def main_interface():
    # Title is slightly smaller and more compact
    st.markdown("<div class='main-header'>Restaurant Search Assistant 🍽️</div>", unsafe_allow_html=True)
    
    # Chat container for messages - starts immediately after the header
    with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        # Display chat history
        display_chat_history()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Fixed input container at the bottom
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        
        # Create a form for input
        with st.form(key="chat_form", clear_on_submit=True):
            # Horizontal layout for input and button
            cols = st.columns([6, 1])
            with cols[0]:
                user_input = st.text_input("", placeholder="Type a message...", label_visibility="collapsed")
            with cols[1]:
                submit_button = st.form_submit_button("➤")
        
        # Process the input when submitted
        if submit_button and user_input:
            process_user_input(user_input)
            # Rerun to update the UI
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

def display_chat_history():
    """Display the chat history in a WhatsApp-like interface"""
    for i, message in enumerate(st.session_state.messages):
        # Add message timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().strftime("%H:%M")
            
        role = message["role"]
        timestamp = message["timestamp"]
        
        if role == "user":
            st.markdown(f"""
            <div class='chat-message user-message'>
                <div>{message["content"]}</div>
                <div class='timestamp'>{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check if this message used context from previous exchanges
            context_indicator = ""
            if message.get("has_context", False):
                context_count = message.get("context_count", 1)
                context_indicator = f"<span class='context-indicator'>✓ Using context from {context_count} previous exchange{'s' if context_count > 1 else ''}</span>"
            
            # Bot message with expandable details
            st.markdown(f"""
            <div class='chat-message bot-message'>
                <div>{context_indicator}{message["content"]}</div>
                <div class='timestamp'>{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create an expander for the details
            with st.expander("View search details"):
                # Display search decision
                if "search_decision" in message:
                    display_search_decision(message["search_decision"])
                
                # SQL Queries
                if "sql_queries" in message:
                    display_sql_queries(message["sql_queries"])
                
                # Pinecone Queries
                if "pinecone_queries" in message:
                    display_pinecone_queries(message["pinecone_queries"])
                
                # Show chat history used for context if available
                if message.get("has_context", False):
                    st.markdown("<div class='sub-header'>Conversation Context Used</div>", unsafe_allow_html=True)
                    context_count = message.get("context_count", 1)
                    
                    # Calculate which messages were used for context
                    message_index = st.session_state.messages.index(message)
                    context_start = max(0, message_index - 10)  # Get up to 10 messages back (5 exchanges)
                    
                    # Display the context messages
                    for j in range(context_start, message_index):
                        ctx_msg = st.session_state.messages[j]
                        role = ctx_msg.get("role", "")
                        content = ctx_msg.get("content", "")
                        
                        # Format the context message
                        st.markdown(f"**{role.capitalize()}:** {content}")
                    
                    st.markdown("---")

def process_user_input(query):
    """Process the user input and update the chat history"""
    if not query:
        return
        
    # Add user message to chat history with timestamp
    st.session_state.messages.append({
        "role": "user", 
        "content": query,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Process the query with chat history context
    result = process_query(query)
    
    # Add bot response to chat history with timestamp
    bot_response = {
        "role": "bot", 
        "content": result.get("final_response", "I couldn't process your query."),
        "search_decision": result.get("search_decision", {}),
        "sql_queries": result.get("sql_queries", []),
        "pinecone_queries": result.get("pinecone_queries", []),
        "timestamp": datetime.now().strftime("%H:%M")
    }
    
    # Add context badge if this is a follow-up (more than 1 exchange)
    if len(st.session_state.messages) > 2:  # More than one user-bot exchange
        bot_response["has_context"] = True
        bot_response["context_count"] = min(5, len(st.session_state.messages) // 2)  # Number of exchanges used (max 5)
    
    st.session_state.messages.append(bot_response)

# Run the main interface
main_interface()

# Footer
st.markdown("---")
st.markdown("<div class='info-text'>Powered by Hugging Face, Pinecone, and Streamlit</div>", unsafe_allow_html=True) 