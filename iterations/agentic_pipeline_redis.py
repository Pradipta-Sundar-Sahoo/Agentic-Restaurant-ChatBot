import streamlit as st
import asyncio
import json
import os
import time
import uuid
import pickle
from datetime import datetime
from agentic_openai_pipeline import run_agent
import pandas as pd
import redis
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="Restaurant Search Agent",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history and in-memory fallback
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
# In-memory fallback storage when Redis is not available
if "memory_storage" not in st.session_state:
    st.session_state.memory_storage = []
    
if "debug_redis" not in st.session_state:
    st.session_state.debug_redis = False

# Initialize Redis client
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
# Create two Redis clients: one for string data with decoding, and one for binary data without decoding
try:
    # Regular Redis client for text data
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Binary Redis client for embedding data
    redis_binary_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()  # Test connection
    redis_connected = True
    st.sidebar.success("✓ Connected to Redis")
except Exception as e:
    redis_connected = False
    redis_binary_client = None
    st.sidebar.error(f"✗ Failed to connect to Redis: {str(e)}")
    st.sidebar.info("Make sure Redis server is running: 'redis-server'")

# Load sentence transformer model for embeddings
try:
    # Define a local cache directory for the model
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Load the model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
    embedding_model_loaded = True
    st.sidebar.success("✓ Embedding model loaded")
except Exception as e:
    embedding_model_loaded = False
    embedding_model = None
    st.sidebar.error(f"✗ Failed to load embedding model: {str(e)}")

# Custom CSS for better UI with chat-like interface
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;  /* Add padding to make room for fixed input */
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
    .similar-query {
        background-color: #f8f4ff;
        border-left: 4px solid #722ed1;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
    }
    .similar-query-score {
        font-size: 0.8rem;
        color: #722ed1;
        margin-bottom: 0.5rem;
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
</style>
""", unsafe_allow_html=True)

# Redis helper functions
def save_chat_to_redis(query, response_data):
    """Save chat message and response to Redis or in-memory fallback"""
    # Create a unique ID for this chat
    chat_id = f"chat:{int(time.time())}:{uuid.uuid4()}"
    
    # Prepare chat data
    chat_data = {
        "id": chat_id,
        "query": query,
        "response": response_data.get("final_response", ""),
        "timestamp": datetime.now().isoformat(),
        "search_decision": json.dumps(response_data.get("search_decision", {})),
        "sql_queries": json.dumps(response_data.get("sql_queries", [])),
        "pinecone_queries": json.dumps(response_data.get("pinecone_queries", [])),
        "session_id": st.session_state.session_id
    }
    
    # Store embedding data if available
    if embedding_model_loaded:
        try:
            query_embedding = embedding_model.encode(query)
            # Add embedding to data - will be used by Redis or in-memory fallback
            chat_data["embedding_vector"] = query_embedding
        except Exception as e:
            if st.session_state.debug_redis:
                st.sidebar.error(f"Error generating embedding: {str(e)}")
    
    # Try Redis first if connected
    if redis_connected:
        try:
            # Debug info
            if st.session_state.debug_redis:
                st.sidebar.write("Saving to Redis key:", chat_id)
                st.sidebar.write("Data keys:", list(chat_data.keys()))
            
            # Store text data in Redis
            for key, value in chat_data.items():
                # Skip embedding, store separately
                if key == "embedding_vector":
                    continue
                    
                try:
                    redis_client.hset(chat_id, key, value)
                    if st.session_state.debug_redis:
                        st.sidebar.write(f"Saved {key}: {type(value)}")
                except Exception as e:
                    st.sidebar.error(f"Error saving {key}: {str(e)}")
            
            # Add to index for efficient retrieval
            try:
                redis_client.sadd("all_chats", chat_id)
                if st.session_state.debug_redis:
                    st.sidebar.write(f"Added {chat_id} to all_chats set")
            except Exception as e:
                st.sidebar.error(f"Error adding to index: {str(e)}")
            
            # Store embedding separately if available
            if "embedding_vector" in chat_data and embedding_model_loaded:
                try:
                    # Store binary embedding with the binary client
                    redis_binary_client.hset(chat_id, "embedding", pickle.dumps(chat_data["embedding_vector"]))
                    if st.session_state.debug_redis:
                        st.sidebar.write(f"Saved embedding: {chat_data['embedding_vector'].shape}")
                except Exception as e:
                    st.sidebar.error(f"Error saving embedding: {str(e)}")
                
            return chat_id
                
        except Exception as e:
            st.sidebar.error(f"Error saving to Redis: {str(e)}")
            # Fall through to in-memory storage
    
    # Fallback to in-memory storage
    st.session_state.memory_storage.append(chat_data)
    if st.session_state.debug_redis:
        st.sidebar.info(f"Saved to in-memory storage (Redis unavailable). Total entries: {len(st.session_state.memory_storage)}")
    return chat_id

def find_similar_queries(query, threshold=0.7, max_results=3):
    """Find similar queries in Redis or in-memory fallback using vector similarity"""
    if not embedding_model_loaded:
        return []
    
    # Get embedding for current query
    try:
        query_embedding = embedding_model.encode(query)
    except Exception as e:
        st.sidebar.error(f"Error encoding query: {str(e)}")
        return []
    
    similar_chats = []
    
    # Try Redis first if connected
    if redis_connected:
        try:
            # Get all chat IDs
            all_chat_ids = redis_client.smembers("all_chats")
            if st.session_state.debug_redis:
                st.sidebar.write(f"Found {len(all_chat_ids)} total chats in Redis for similarity search")
            
            # Counter for successful embeddings
            processed_embeddings = 0
            
            for chat_id in all_chat_ids:
                try:
                    # Get stored embedding using binary client
                    stored_embedding_bytes = redis_binary_client.hget(chat_id, "embedding")
                    if stored_embedding_bytes:
                        try:
                            # Convert bytes back to tensor
                            stored_embedding = pickle.loads(stored_embedding_bytes)
                            processed_embeddings += 1
                            
                            # Calculate cosine similarity
                            similarity = util.pytorch_cos_sim(
                                torch.tensor([query_embedding]), 
                                torch.tensor([stored_embedding])
                            ).item()
                            
                            if similarity >= threshold:
                                # Get chat data (text fields) with regular client
                                chat_data = redis_client.hgetall(chat_id)
                                chat_data["similarity"] = similarity
                                similar_chats.append(chat_data)
                                if st.session_state.debug_redis:
                                    st.sidebar.write(f"Found similar chat: {chat_id} (similarity: {similarity:.3f})")
                        except Exception as e:
                            if st.session_state.debug_redis:
                                st.sidebar.error(f"Error processing embedding for {chat_id}: {str(e)}")
                            continue
                except Exception as e:
                    if st.session_state.debug_redis:
                        st.sidebar.error(f"Error retrieving data for {chat_id}: {str(e)}")
                    continue
            
            if st.session_state.debug_redis:
                st.sidebar.write(f"Successfully processed {processed_embeddings} embeddings out of {len(all_chat_ids)} total chats")
            
        except Exception as e:
            st.sidebar.error(f"Error finding similar queries in Redis: {str(e)}")
            # Fall through to in-memory search
    
    # If no results from Redis or Redis is not connected, try in-memory
    if not similar_chats:
        try:
            # Search in-memory storage
            memory_storage = st.session_state.memory_storage
            if st.session_state.debug_redis:
                st.sidebar.write(f"Searching in-memory storage with {len(memory_storage)} items")
            
            for stored_chat in memory_storage:
                if "embedding_vector" in stored_chat:
                    stored_embedding = stored_chat["embedding_vector"]
                    
                    # Calculate cosine similarity
                    similarity = util.pytorch_cos_sim(
                        torch.tensor([query_embedding]), 
                        torch.tensor([stored_embedding])
                    ).item()
                    
                    if similarity >= threshold:
                        chat_copy = stored_chat.copy()
                        # Remove embedding vector to avoid serialization issues
                        if "embedding_vector" in chat_copy:
                            del chat_copy["embedding_vector"]
                        chat_copy["similarity"] = similarity
                        similar_chats.append(chat_copy)
            
            if st.session_state.debug_redis:
                st.sidebar.info(f"Found {len(similar_chats)} similar queries in memory.")
            
        except Exception as e:
            st.sidebar.error(f"Error finding similar queries in-memory: {str(e)}")
    
    # Sort by similarity score (descending)
    similar_chats.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    # Return top matches
    return similar_chats[:max_results]

def display_search_decision(search_decision):
    """Display the search decision in a visual format"""
    st.markdown("<div class='sub-header'>Search Strategy</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
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
    
    if "explanation" in search_decision:
        st.markdown(f"**Explanation:** {search_decision['explanation']}")

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


def process_query(query, use_cached=False, cached_response=None):
    """Process the user query using the agentic pipeline or use cached response"""
    if use_cached and cached_response:
        st.success("Using cached response from similar query")
        return json.loads(cached_response)
        
    with st.spinner('Searching for answers...'):
        # Run the agent on the query
        result = asyncio.run(run_agent(query))
        return result

# Add a button to test Redis connection
def test_redis_connection():
    """Test if Redis is available and responsive"""
    try:
        # Create Redis clients with short timeout
        test_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2.0)
        test_binary_client = redis.Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=2.0)
        
        # Test ping
        test_client.ping()
        
        # Test basic operations
        test_key = f"test:connection:{uuid.uuid4()}"
        test_client.set(test_key, "test_value")
        test_value = test_client.get(test_key)
        test_client.delete(test_key)
        
        # Test binary operations
        test_binary_key = f"test:binary:{uuid.uuid4()}"
        test_binary_client.set(test_binary_key, b"binary_test")
        binary_value = test_binary_client.get(test_binary_key)
        test_binary_client.delete(test_binary_key)
        
        return True, "Redis connection successful! ✅"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)} ❌"

# Sidebar organization 
st.sidebar.markdown("## Restaurant Search Assistant")
st.sidebar.markdown("Ask questions about restaurants and their menus.")

# Redis connection status
with st.sidebar.expander("Redis Connection", expanded=False):
    redis_status = "Connected ✅" if redis_connected else "Disconnected ❌"
    st.markdown(f"**Status**: {redis_status}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test", help="Test Redis connection"):
            success, message = test_redis_connection()
            if success:
                st.success(message)
            else:
                st.error(message)
                st.info("Make sure Redis server is running")
    
    with col2:
        if st.button("Refresh", help="Refresh connection status"):
            st.rerun()
            
    if redis_connected and st.session_state.debug_redis:
        try:
            info = redis_client.info()
            st.write(f"Redis version: {info.get('redis_version', 'unknown')}")
            st.write(f"Connected clients: {info.get('connected_clients', 'unknown')}")
            
            # Show number of keys
            total_keys = 0
            try:
                # Get keys count (using DBSIZE command)
                total_keys = redis_client.dbsize()
            except:
                total_keys = "Error counting"
            st.write(f"Total keys: {total_keys}")
            
            # Show number of chat entries
            chat_count = len(redis_client.smembers("all_chats"))
            st.write(f"Stored chats: {chat_count}")
        except Exception as e:
            st.error(f"Error getting Redis info: {str(e)}")

# Toggle for Redis features
enable_redis_search = st.sidebar.checkbox("Enable similar query search", 
                                         value=embedding_model_loaded, # Changed to only depend on embedding model
                                         disabled=not embedding_model_loaded)

similarity_threshold = st.sidebar.slider("Similarity threshold", 
                                        min_value=0.5, 
                                        max_value=0.95, 
                                        value=0.7,
                                        step=0.05,
                                        disabled=not enable_redis_search)

# Debug mode toggle
debug_redis = st.sidebar.checkbox("Debug Redis operations", 
                                 value=st.session_state.debug_redis)
st.session_state.debug_redis = debug_redis

# Override the st.button function within the Recent Queries expander to make sure it works
def create_clickable_query_buttons(queries_data, source="redis"):
    """Create clickable buttons for cached queries that work reliably"""
    if not queries_data:
        st.info("No cached queries found.")
        return
        
    # Sort by timestamp (most recent first)
    queries_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Create a unique container for the buttons
    for i, chat in enumerate(queries_data[:5]):  # Limit to 5 most recent
        query = chat.get("query", "Unknown query")
        timestamp = chat.get("timestamp", "")
        timestamp_display = timestamp.split("T")[0] if "T" in timestamp else timestamp
        
        # Create a horizontal layout with query and timestamp
        cols = st.columns([3, 1])
        with cols[0]:
            # Use a unique key based on query content
            query_key = f"query_{source}_{i}_{hash(query)}"
            if st.button(
                f"{query[:30] + '...' if len(query) > 30 else query}", 
                key=query_key,
                use_container_width=True
            ):
                st.session_state.reuse_query = query
                st.rerun()
        with cols[1]:
            st.caption(timestamp_display)
            
        # Add a small separator
        st.markdown("<hr style='margin: 2px 0'>", unsafe_allow_html=True)
    
    if len(queries_data) > 5:
        st.info(f"Showing 5 of {len(queries_data)} cached queries.")

# Display cached queries in sidebar for quick access
with st.sidebar.expander("Recent Queries", expanded=True):
    # Get chat count from Redis or in-memory
    chat_count = 0
    if redis_connected:
        try:
            chat_count = len(redis_client.smembers("all_chats"))
        except:
            chat_count = "Error counting"
    else:
        chat_count = len(st.session_state.memory_storage)
    
    st.write(f"Total stored queries: {chat_count}")
    
    # Add clear cache button
    if st.button("Clear All Cache", help="Delete all cached queries and responses"):
        # Show confirmation
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = True
            st.warning("Are you sure? This will delete all cached data.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Clear All"):
                    # Clear Redis cache
                    if redis_connected:
                        try:
                            # Get all chat keys
                            all_chat_ids = redis_client.smembers("all_chats")
                            
                            # Delete each chat
                            for chat_id in all_chat_ids:
                                redis_client.delete(chat_id)
                            
                            # Clear the index
                            redis_client.delete("all_chats")
                            
                            st.success("Redis cache cleared!")
                        except Exception as e:
                            st.error(f"Error clearing Redis cache: {str(e)}")
                    
                    # Always clear in-memory storage too
                    st.session_state.memory_storage = []
                    st.success("In-memory cache cleared!")
                    
                    # Reset confirmation state
                    st.session_state.confirm_clear = False
                    
                    # Refresh the page
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()
    
    st.markdown("**Cached queries you can reuse:**")
    # Display cached queries here directly
    if redis_connected:
        try:
            # Get all chat IDs
            all_chat_ids = redis_client.smembers("all_chats")
            
            if not all_chat_ids:
                st.info("No cached queries found.")
            else:
                # Show most recent chats first (sort by timestamp)
                chat_data_list = []
                for chat_id in all_chat_ids:
                    try:
                        chat_data = redis_client.hgetall(chat_id)
                        if "query" in chat_data:  # Make sure the query field exists
                            chat_data["id"] = chat_id
                            chat_data_list.append(chat_data)
                    except Exception as e:
                        if st.session_state.debug_redis:
                            st.error(f"Error retrieving chat {chat_id}: {str(e)}")
                        continue
                
                # If no valid chats were found
                if not chat_data_list:
                    st.info("No valid cached queries found.")
                else:
                    create_clickable_query_buttons(chat_data_list, "redis")
        except Exception as e:
            st.warning(f"Error displaying cached queries: {str(e)}")
            if st.session_state.debug_redis:
                st.error(f"Detailed error: {str(e)}")
    else:
        # Show in-memory cache
        memory_storage = st.session_state.memory_storage
        if not memory_storage or all("query" not in chat for chat in memory_storage):
            st.info("No cached queries found.")
        else:
            valid_memory = [chat for chat in memory_storage if "query" in chat]
            create_clickable_query_buttons(valid_memory, "memory")

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
        role = message["role"]
        timestamp = datetime.now().strftime("%H:%M") # Normally you'd store this with each message
        
        if role == "user":
            st.markdown(f"""
            <div class='chat-message user-message'>
                <div>{message["content"]}</div>
                <div class='timestamp'>{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Bot message with expandable details
            is_cached = message.get("is_cached", False)
            cached_label = " (From Cache)" if is_cached else ""
            
            st.markdown(f"""
            <div class='chat-message bot-message'>
                <div>{message["content"]}</div>
                <div class='timestamp'>{timestamp}{cached_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create an expander for the details
            with st.expander("View search details"):
                # If response is cached, show original query
                if is_cached:
                    st.markdown(f"**Original Query:** {message.get('original_query', 'Unknown')}")
                
                # Display search decision
                if "search_decision" in message:
                    display_search_decision(message["search_decision"])
                
                # SQL Queries
                if "sql_queries" in message:
                    display_sql_queries(message["sql_queries"])
                
                # Pinecone Queries
                if "pinecone_queries" in message:
                    display_pinecone_queries(message["pinecone_queries"])

def process_user_input(query):
    """Process the user input and update the chat history"""
    if not query:
        return
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Find similar queries in Redis if enabled
    similar_queries = []
    if enable_redis_search:  # Check if similar query search is enabled
        try:
            # Look for similar queries before processing
            similar_queries = find_similar_queries(query, threshold=similarity_threshold)
            
            # Log the search results for debugging
            if st.session_state.debug_redis:
                st.sidebar.write(f"Found {len(similar_queries)} similar queries")
                if similar_queries:
                    st.sidebar.write(f"Top match similarity: {similar_queries[0].get('similarity', 0):.2f}")
            
            # If similar queries are found, show them and let the user choose
            if similar_queries:
                st.session_state.similar_queries = similar_queries
                st.session_state.show_similar_dialog = True
                return  # Return without processing to show the similar query dialog
        except Exception as e:
            st.sidebar.error(f"Error finding similar queries: {str(e)}")
            similar_queries = []
    
    # If no similar queries or similar query search is disabled, process the query directly
    result = process_query(query)
    
    # Save to Redis if connected
    if redis_connected:
        try:
            redis_id = save_chat_to_redis(query, result)
            if st.session_state.debug_redis and redis_id:
                st.sidebar.success(f"Saved to Redis with ID: {redis_id}")
        except Exception as e:
            st.sidebar.error(f"Failed to save to Redis: {str(e)}")
    
    # Add bot response to chat history
    st.session_state.messages.append({
        "role": "bot", 
        "content": result.get("final_response", "I couldn't process your query."),
        "search_decision": result.get("search_decision", {}),
        "sql_queries": result.get("sql_queries", []),
        "pinecone_queries": result.get("pinecone_queries", []),
        "is_cached": False
    })
    
    # Force rerun to update the UI
    st.rerun()

# Check if we need to show the similar query dialog
if "show_similar_dialog" not in st.session_state:
    st.session_state.show_similar_dialog = False
    
if "similar_queries" not in st.session_state:
    st.session_state.similar_queries = []
    
if "reuse_query" not in st.session_state:
    st.session_state.reuse_query = None

# Check if we should process a reused query from the sidebar
if st.session_state.reuse_query:
    query_to_use = st.session_state.reuse_query
    st.session_state.reuse_query = None  # Reset after using it
    
    # Add the query to chat history
    st.session_state.messages.append({"role": "user", "content": query_to_use})
    
    # Look for similar queries first (except the one we just clicked)
    if enable_redis_search:
        try:
            similar_queries = find_similar_queries(query_to_use, threshold=similarity_threshold)
            
            # Filter out the exact query we're using
            similar_queries = [q for q in similar_queries if q.get("query") != query_to_use]
            
            # If we found similar queries (besides the one we clicked)
            if similar_queries:
                st.session_state.similar_queries = similar_queries
                st.session_state.show_similar_dialog = True
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error finding similar queries: {str(e)}")
    
    # Process the query directly if no other similar queries
    result = process_query(query_to_use)
    
    # Save to Redis if connected (avoid duplicating cached queries)
    if redis_connected:
        try:
            redis_id = save_chat_to_redis(query_to_use, result)
        except Exception as e:
            st.sidebar.error(f"Failed to save to Redis: {str(e)}")
    
    # Add bot response to chat history
    st.session_state.messages.append({
        "role": "bot", 
        "content": result.get("final_response", "I couldn't process your query."),
        "search_decision": result.get("search_decision", {}),
        "sql_queries": result.get("sql_queries", []),
        "pinecone_queries": result.get("pinecone_queries", []),
        "is_cached": False
    })
    
    # Rerun to update UI
    st.rerun()

# Show dialog to choose whether to use cached response
if st.session_state.show_similar_dialog and st.session_state.similar_queries:
    # Get the last user message
    last_user_query = st.session_state.messages[-1]["content"]
    
    st.success("Found similar previous queries! Would you like to use a cached response?")
    st.markdown("<div class='sub-header'>Similar Previous Queries</div>", unsafe_allow_html=True)
    
    # Display similar queries with similarity score
    for i, similar in enumerate(st.session_state.similar_queries):
        similarity = similar.get("similarity", 0) * 100
        query = similar.get("query", "Unknown query")
        timestamp = similar.get("timestamp", "")
        timestamp_display = timestamp.split("T")[0] if "T" in timestamp else timestamp
        
        with st.container():
            st.markdown(f"""
            <div class='similar-query'>
                <div class='similar-query-score'>Similarity: {similarity:.1f}%</div>
                <strong>Query:</strong> {query}
                <div class='timestamp'>{timestamp_display}</div>
                <div><em>Response:</em> {similar.get("response", "")[:100]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Ask if user wants to use a cached response
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use cached response", key="use_cached_btn", use_container_width=True):
            # Parse the cached data
            cached_data = st.session_state.similar_queries[0]  # Most similar query
            
            # Create result from cached data
            result = {
                "final_response": cached_data.get("response", "No cached response available."),
                "search_decision": json.loads(cached_data.get("search_decision", "{}")),
                "sql_queries": json.loads(cached_data.get("sql_queries", "[]")),
                "pinecone_queries": json.loads(cached_data.get("pinecone_queries", "[]")),
                "is_cached": True,
                "original_query": cached_data.get("query", "Unknown")
            }
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "bot", 
                "content": result.get("final_response", "I couldn't process your query."),
                "search_decision": result.get("search_decision", {}),
                "sql_queries": result.get("sql_queries", []),
                "pinecone_queries": result.get("pinecone_queries", []),
                "is_cached": True,
                "original_query": result.get("original_query", last_user_query)
            })
            
            # Clear the dialog state
            st.session_state.show_similar_dialog = False
            st.session_state.similar_queries = []
            
            # Refresh the UI
            st.rerun()
            
    with col2:
        if st.button("Search again", key="search_again_btn", use_container_width=True):
            # Process with agentic pipeline
            result = process_query(last_user_query)
            
            # Save to Redis if connected
            if redis_connected:
                try:
                    redis_id = save_chat_to_redis(last_user_query, result)
                except Exception as e:
                    st.sidebar.error(f"Failed to save to Redis: {str(e)}")
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "bot", 
                "content": result.get("final_response", "I couldn't process your query."),
                "search_decision": result.get("search_decision", {}),
                "sql_queries": result.get("sql_queries", []),
                "pinecone_queries": result.get("pinecone_queries", []),
                "is_cached": False
            })
            
            # Clear the dialog state
            st.session_state.show_similar_dialog = False
            st.session_state.similar_queries = []
            
            # Refresh the UI
            st.rerun()
else:
    # Run the main interface
    main_interface()

# Footer (now fixed at bottom with input)
st.markdown("---")
st.markdown("<div class='info-text'>Powered by OpenAI, Pinecone, Redis and Streamlit</div>", unsafe_allow_html=True) 