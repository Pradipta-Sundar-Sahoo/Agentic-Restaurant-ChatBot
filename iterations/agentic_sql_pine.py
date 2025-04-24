import os
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Literal
from pydantic import BaseModel, Field
import openai # Import OpenAI library
import psycopg2
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv(override=True) # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Use a different log file for this pipeline
        logging.FileHandler("openai_agentic_pipeline.log", encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ],
    encoding='utf-8'
)
# Use a different logger name
logger = logging.getLogger("openai_agentic_pipeline")
restaurant_name_list=['Imperio Restaurant','Burger King','Burger Singh - Big Punjabi Burgers',"Domino's Pizza",'Gochick','KFC','House Of Biryan- Biryani, Kepsa And More','Nomad Pizza - Traveller Series','Salad Days','The Burger Club']

# Database connection parameters (Keep the same)
DB_PARAMS = {
    'dbname': 'restaurant_db',
    'user': 'postgres',
    'password': '2112',
    'host': 'localhost',
    'port': '5433'
}

# Schema description (Updated to match schema.sql)
DB_SCHEMA = """
Table: restaurant_menu_flat
Columns:
- item_id (SERIAL PRIMARY KEY)
- item_name (TEXT)
- item_description (TEXT)
- item_price (NUMERIC)
- item_category (TEXT)
- item_is_veg (BOOLEAN)
- restaurant_name (TEXT)
- restaurant_address (TEXT)
- restaurant_phone_numbers (TEXT[])
- restaurant_cuisines (TEXT[])
- restaurant_opening_hours (JSONB)
- restaurant_dining_rating (NUMERIC)
- restaurant_delivery_rating (NUMERIC)
- restaurant_dining_ratings_count (INTEGER)
- restaurant_delivery_ratings_count (INTEGER)
- restaurant_source_url (TEXT)
- description_clean (TEXT)
- city (TEXT)
- created_at (TIMESTAMP WITH TIME ZONE)
- updated_at (TIMESTAMP WITH TIME ZONE)
"""

# Pinecone configuration
PINECONE_INDEX_NAME = "nuggetpine"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
logger.info(f"Using Pinecone environment: {PINECONE_ENVIRONMENT}")

# Initialize sentence transformer model for embeddings
try:
    # Define a local cache directory for the model
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logger.info(f"Created model cache directory at: {cache_dir}")
    
    logger.info(f"Loading embedding model from cache: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_dir)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading embedding model: {str(e)}")
    model = None

# Initialize Pinecone client
try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if pinecone_api_key:
        pinecone = Pinecone(api_key=pinecone_api_key)
        logger.info("Pinecone client initialized")
    else:
        logger.warning("PINECONE_API_KEY not found in environment")
        pinecone = None
except Exception as e:
    logger.error(f"Error initializing Pinecone client: {str(e)}")
    pinecone = None

# Define state schema (Updated to include Pinecone search)
class AgentState(BaseModel):
    user_query: str = Field(description="The original query from the user")
    sql_queries: List[str] = Field(default_factory=list, description="Generated SQL queries")
    sql_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from SQL queries")
    contexts: List[str] = Field(default_factory=list, description="Accumulated context from query results")
    current_iteration: int = Field(default=0, description="Current iteration count")
    final_response: Optional[str] = Field(default=None, description="Final response to the user")
    is_complete: bool = Field(default=False, description="Whether the process is complete")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning about next steps")
    
    # New fields for Pinecone vector search
    pinecone_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Queries for Pinecone vector search")
    pinecone_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from Pinecone vector search")
    combined_context: Optional[str] = Field(default=None, description="Combined context from SQL and vector search")
    pinecone_retry_count: int = Field(default=0, description="Counter for Pinecone query retries")
    
    # New field for the judge's decision
    search_decision: Dict[str, int] = Field(default_factory=dict, description="Judge's decision on which search methods to use")
    
    # New field for the combined context judgment
    context_judgment: Dict[str, Any] = Field(default_factory=dict, description="Judge's evaluation of the combined context")

    def add_sql_query(self, query: str) -> None:
        self.sql_queries.append(query)

    def add_sql_result(self, result: List[Dict[str, Any]]) -> None:
        self.sql_results.append(result)

        # Convert the result to a readable string format
        result_str = json.dumps(result, indent=2, default=str)
        self.contexts.append(f"SQL Query #{len(self.sql_queries)}:\n{self.sql_queries[-1]}\n\nResults:\n{result_str}")

    def add_pinecone_query(self, query: Dict[str, Any]) -> None:
        self.pinecone_queries.append(query)
    
    def add_pinecone_result(self, result: Dict[str, Any]) -> None:
        self.pinecone_results.append(result)
        
        # Convert the result to a readable string format
        result_str = json.dumps(result, indent=2, default=str)
        self.contexts.append(f"Vector Search #{len(self.pinecone_queries)}:\nQuery: {json.dumps(self.pinecone_queries[-1], indent=2)}\n\nResults:\n{result_str}")

    def increment_iteration(self) -> None:
        self.current_iteration += 1

# --- Modified LLM Client for OpenAI ---
class LLMClient:
    def __init__(self, model_name: str = "gpt-4o"):
        # Fetch the API key from environment variables
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not found in environment variable OPENAI_API_KEY")
            raise ValueError("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")

        # Initialize the OpenAI client (using the recommended way for openai>=1.0.0)
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model_name = model_name
            logger.info(f"OpenAI LLM Client initialized with model: {self.model_name}")
        except AttributeError:
             # Fallback for older openai versions < 1.0.0
             openai.api_key = self.api_key
             self.client = openai.ChatCompletion # Use the class directly
             self.model_name = model_name
             logger.warning("Using fallback initialization for older OpenAI library version (<1.0.0). Consider upgrading.")
             logger.info(f"OpenAI LLM Client initialized with model: {self.model_name}")
        except Exception as e:
             logger.error(f"Failed to initialize OpenAI client: {e}")
             raise

    def generate_sql(self, query: str, iteration: int, contexts: List[str]) -> str:
        start_time = time.time()
        logger.info(f"Generating SQL queries for iteration {iteration} using {self.model_name}")

        # Construct the messages list for OpenAI API
        messages = []

        # System prompt defining the role and context
        system_prompt = f"""You are an AI assistant that helps users query a database based on their natural language questions.
Database Schema:
{DB_SCHEMA}
restaurant_name: {restaurant_name_list} USE THIS EXACTLY AS IT IS for restaurant name in sql queries
Your task is to generate SQL queries based on the user's question and previous query results (if any).
- For the first query (iteration 0), generate one or more SQL queries to answer the initial question. Separate multiple queries with '---'.
- For subsequent queries, analyze the provided context (previous queries and results) and the original user question.
- If more information is needed, generate ONE additional SQL query.
- If enough information is available, respond with "END" followed by a comprehensive natural language answer based on all collected data.
- Always use LIMIT 5 in your SQL queries unless the question specifically asks for more results.
- Respond ONLY with the SQL query(s) or "END" followed by the answer. Do not include explanations unless it's part of the final answer after "END".
"""
        messages.append({"role": "system", "content": system_prompt})

        # Add context from previous iterations if available
        if iteration > 0:
            contexts_text = "\n\n".join(contexts)
            context_message = f"""So far, we have executed the following queries and got these results:
{contexts_text}

Original User Question: {query}

Now, decide if you need another SQL query or if you can provide the final answer."""
            messages.append({"role": "user", "content": context_message})
        else:
            # First iteration, just add the user query
             messages.append({"role": "user", "content": f"User Question: {query}"})

        try:
            # Check how the client was initialized to call the API correctly
            if isinstance(self.client, openai.OpenAI): # For openai >= 1.0.0
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2 # Lower temperature for more deterministic SQL generation
                )
                response = completion.choices[0].message.content
            else: # Fallback for openai < 1.0.0
                 completion = self.client.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2
                )
                 response = completion.choices[0].message.content

            execution_time = time.time() - start_time
            logger.info(f"OpenAI response generated in {execution_time:.2f}s")

            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise

# --- SQL Query Executor (Keep the same) ---
async def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute SQL query asynchronously and return results"""
    logger.info(f"Executing SQL query: {query}")
    start_time = time.time()

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, lambda: _execute_query_sync(query))
        execution_time = time.time() - start_time
        logger.info(f"SQL query executed in {execution_time:.2f}s, returned {len(results)} results")
        # Remove or comment out debug prints if not needed
        # print("s9hsh",results)
        return results
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return [{"error": str(e)}]

def _execute_query_sync(query: str) -> List[Dict[str, Any]]:
    """Synchronous implementation of SQL query execution"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        cursor.execute(query)

        # Handle cases where query might not return results (e.g., INSERT, UPDATE)
        if cursor.description:
            column_names = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                result_dict = {}
                for i, value in enumerate(row):
                    # Attempt to decode bytea/memoryview if necessary
                    if isinstance(value, (bytes, memoryview)):
                        try:
                            value = bytes(value).decode('utf-8')
                        except UnicodeDecodeError:
                            logger.warning(f"Could not decode byte sequence for column {column_names[i]}. Keeping as bytes.")
                            # Keep as bytes or handle appropriately
                            pass
                    result_dict[column_names[i]] = value
                results.append(result_dict)
        else:
            results = [] # No results to fetch

        cursor.close()
        conn.close()
        return results
    except psycopg2.Error as db_err: # Catch specific psycopg2 errors
        logger.error(f"Database error: {db_err}")
        raise # Re-raise the specific error for potentially better handling upstream
    except Exception as e:
        logger.error(f"Unexpected error during sync query execution: {e}")
        raise

# --- Pinecone Vector Search Functions ---

async def generate_pinecone_query(state: AgentState) -> AgentState:
    """Generate Pinecone vector search query"""
    logger.info(f"Generating Pinecone query for iteration {state.current_iteration}")
    
    # Generate vector query
    query_params = await _generate_vector_query(state)
    state.add_pinecone_query(query_params)
    logger.info(f"Generated Pinecone query parameters: {query_params}")
    
    return state

async def _generate_vector_query(state: AgentState) -> Dict[str, Any]:
    """Generate Pinecone query parameters based on user query using LLM"""
    logger.info("Generating vector search parameters")
    
    llm_client = LLMClient()
    system_prompt = f"""You are an AI assistant that helps create search parameters for a vector database.
The user has a database of restaurant menu items with the following fields:
- item_id (SERIAL PRIMARY KEY): Unique identifier for the menu item
- item_name (TEXT): Name of the menu item
- item_description (TEXT): Description of the menu item
- item_price (NUMERIC): Price of the item
- item_category (TEXT): Category of the menu item  
- item_is_veg (BOOLEAN): Whether the item is vegetarian
- restaurant_name (TEXT): Name of the restaurant {restaurant_name_list} USE THIS EXACTLY AS IT IS for restaurant name in sql queries
- restaurant_address (TEXT): Physical address of the restaurant
- restaurant_phone_numbers (TEXT[]): Contact information as an array
- restaurant_cuisines (TEXT[]): Types of cuisine as an array
- restaurant_opening_hours (JSONB): Opening hours in JSON format
- restaurant_dining_rating (NUMERIC): Rating for dining experience
- restaurant_delivery_rating (NUMERIC): Rating for delivery experience
- restaurant_dining_ratings_count (INTEGER): Number of dining ratings
- restaurant_delivery_ratings_count (INTEGER): Number of delivery ratings
- restaurant_source_url (TEXT): Original data source URL
- description_clean (TEXT): Cleaned description text
- city (TEXT): City where restaurant is located

Extract the key search terms for semantic search and create metadata filters based on the schema above.
Respond with a JSON object containing:
1. "query_text": The main text to search for semantic similarity
2. "metadata_filters": Any metadata filters to apply (using exact field names from the schema)

IMPORTANT: Return only the JSON object with no additional markdown formatting, code blocks, or explanations.

Example response:
{{"query_text": "spicy chicken dishes", "metadata_filters": {{"item_is_veg": false}}}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate vector search parameters for this query: {state.user_query}"}
    ]
    
    try:
        if isinstance(llm_client.client, openai.OpenAI): # For openai >= 1.0.0
            completion = llm_client.client.chat.completions.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
                response_format={"type": "json_object"} # Force JSON response format
            )
            response = completion.choices[0].message.content
        else: # Fallback for openai < 1.0.0
            completion = llm_client.client.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.2
            )
            response = completion.choices[0].message.content

        # Clean up the response by removing any code block formatting
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON response
        try:
            query_params = json.loads(cleaned_response)
            logger.info(f"Generated vector search parameters: {query_params}")
            return query_params
        except json.JSONDecodeError:
            logger.error(f"Error parsing vector search parameters JSON: {cleaned_response}")
            # Fallback to using the raw query
            return {"query_text": state.user_query, "metadata_filters": {}}
            
    except Exception as e:
        logger.error(f"Error generating vector search parameters: {str(e)}")
        return {"query_text": state.user_query, "metadata_filters": {}}

# --- Helper functions for conditional routing ---

def check_sql_needed_or_complete(state: AgentState) -> Literal["execute_sql", "generate_final_response", "end_loop"]:
    """
    Decide next step after generating SQL queries.
    - If state is marked complete by generate_queries (LLM returned "END"), go to final response.
    - If new SQL queries were added, execute them.
    - Otherwise, end (safety).
    """
    if state.is_complete:
        logger.info("Routing: SQL generation decided to end. Proceeding to final response.")
        # If Pinecone was also requested, we should probably still run it?
        # Let's reconsider. generate_queries ending means it thinks it has enough info *from SQL context*.
        # If Pinecone was *also* requested, it's better to gather that too before final response.
        # Let's stick to the simpler model for now: if generate_queries says END, it ENDs the query phase.
        return "generate_final_response"
    elif len(state.sql_queries) > len(state.sql_results): # Check if new queries were added
        logger.info("Routing: New SQL queries generated. Proceeding to execution.")
        return "execute_sql"
    else:
        logger.warning("Routing: No new SQL queries and not complete. Ending loop.")
        # This case might indicate an issue where generate_queries didn't produce output or END.
        # Force completion.
        if not state.final_response:
             state.final_response = "Agent loop terminated unexpectedly after query generation."
        state.is_complete = True
        return "generate_final_response" # Go to final response to output the termination message

def decide_next_step_after_sql(state: AgentState) -> Literal["generate_pinecone_query", "generate_queries", "generate_final_response"]:
    """
    Decide route after SQL execution based on completion status and search plan.
    - If state is complete, go to final response.
    - If Pinecone search is planned (part of 'both' path), go to Pinecone generation.
    - If only SQL was planned (and not complete), loop back to generate more SQL.
    """
    if state.is_complete:
        logger.info("Routing: State marked complete after SQL execution. Proceeding to final response.")
        # Before final response, maybe combine contexts if Pinecone ran?
        # No, generate_final_response handles combining or using state.contexts.
        return "generate_final_response"
    elif state.search_decision.get("pinecone") == 1:
        logger.info("Routing: SQL part done, Pinecone search is planned. Proceeding to Pinecone query generation.")
        return "generate_pinecone_query"
    else:
        # SQL-only path, and not yet complete, loop back
        logger.info("Routing: SQL-only path, not complete. Looping back to generate next SQL query.")
        # Check iteration limit before looping back
        if state.current_iteration >= 5:
             logger.warning("Routing: Max iterations reached in SQL loop. Ending.")
             state.final_response = "Reached maximum iterations during SQL query phase."
             state.is_complete = True
             return "generate_final_response"
        return "generate_queries"

async def execute_pinecone_search(state: AgentState) -> AgentState:
    """Execute Pinecone vector search"""
    logger.info(f"Executing Pinecone search for iteration {state.current_iteration}")
    
    num_executed = len(state.pinecone_results)
    new_queries = state.pinecone_queries[num_executed:]
    
    if not new_queries:
        return state
    
    tasks = []
    for query_params in new_queries:
        logger.info(f"Preparing to execute vector search: {query_params}")
        if query_params:
            tasks.append(_execute_vector_search(query_params))
        else:
            logger.warning("Skipping empty vector search parameters.")
    
    if tasks:
        query_results = await asyncio.gather(*tasks)
        for result in query_results:
            state.add_pinecone_result(result)
            
            # Check if we got zero results and should retry with a transformed query
            if state.pinecone_retry_count < 3 and _check_zero_results(result):
                logger.info(f"Pinecone search returned 0 results. Retry count: {state.pinecone_retry_count}. Transforming query...")
                state.pinecone_retry_count += 1
                
                # Get the original query parameters
                original_query = state.pinecone_queries[-1]
                
                # Generate a transformed query based on retry count
                transformed_query = await _transform_pinecone_query(original_query, state.pinecone_retry_count, state.user_query)
                
                # Add the transformed query to the state
                state.add_pinecone_query(transformed_query)
                
                # Execute the transformed query immediately
                retry_result = await _execute_vector_search(transformed_query)
                state.add_pinecone_result(retry_result)
                
                # If we still have zero results after max retries, log a warning
                if state.pinecone_retry_count >= 3 and _check_zero_results(retry_result):
                    logger.warning("Maximum Pinecone query retries reached with zero results.")
    
    return state

def _check_zero_results(result: Dict[str, Any]) -> bool:
    """Check if the Pinecone result has zero matches"""
    if "error" in result:
        return True
    
    if "results" in result and "matches" in result["results"]:
        return len(result["results"]["matches"]) == 0
    
    return True  # Default to True if we can't determine (will trigger retry)

async def _transform_pinecone_query(original_query: Dict[str, Any], retry_count: int, user_query: str) -> Dict[str, Any]:
    """Transform the Pinecone query based on retry count"""
    # Create a copy of the original query to modify
    transformed_query = original_query.copy()
    
    if retry_count == 1:
        # First retry: Remove all metadata filters but keep the original query text
        transformed_query["metadata_filters"] = {}
        logger.info("First retry: Removed all metadata filters")
    elif retry_count >= 2:
        # Second/third retry: Use LLM to reformulate the query text and no filters
        transformed_query["metadata_filters"] = {}
        
        # Use LLM to reformulate the query
        llm_client = LLMClient()
        system_prompt = """You are an AI assistant that helps reformulate search queries when the original query returns no results.
        
Your task is to analyze the original query and create a more general, broader version that might return results.
- Remove specific constraints
- Use more general terms
- Consider synonyms for key terms
- Focus on the core intent behind the query

Respond with ONLY the reformulated query text, nothing else."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original user query: '{user_query}'\nOriginal search text: '{original_query.get('query_text', '')}'\n\nThis query returned zero results. Please reformulate it to be more general and likely to match documents in a restaurant menu items database."}
        ]
        
        try:
            if isinstance(llm_client.client, openai.OpenAI):  # For openai >= 1.0.0
                completion = llm_client.client.chat.completions.create(
                    model=llm_client.model_name,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7  # Higher temperature for more creative reformulation
                )
                response = completion.choices[0].message.content
            else:  # Fallback for openai < 1.0.0
                completion = llm_client.client.create(
                    model=llm_client.model_name,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                )
                response = completion.choices[0].message.content
                
            # Update the query text with the reformulated version
            transformed_query["query_text"] = response.strip()
            logger.info(f"Retry {retry_count}: Reformulated query text to '{transformed_query['query_text']}'")
            
        except Exception as e:
            logger.error(f"Error reformulating query: {str(e)}")
            # Fallback: Just use a more general version of the original query
            transformed_query["query_text"] = f"restaurant food {original_query.get('query_text', 'menu items')}"
            logger.info(f"Retry {retry_count}: Using fallback query text '{transformed_query['query_text']}'")
    
    return transformed_query

async def _execute_vector_search(query_params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute vector search in Pinecone"""
    logger.info(f"Executing vector search with params: {query_params}")
    start_time = time.time()
    
    try:
        if not pinecone or not model:
            logger.warning("Pinecone client or embedding model not available")
            return {"error": "Vector search not available - Pinecone or embedding model not initialized"}
        
        # Connect to the index
        index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Generate embedding for the query text
        query_text = query_params.get("query_text", "")
        query_embedding = model.encode(query_text).tolist()
        
        # Get metadata filters if any
        metadata_filters = query_params.get("metadata_filters", {})
        
        # Validate metadata filter fields against schema
        valid_fields = [
            "item_id", "item_name", "item_description", "item_price",
            "item_category", "item_is_veg", "restaurant_name", "restaurant_address",
            "restaurant_phone_numbers", "restaurant_cuisines", "restaurant_opening_hours",
            "restaurant_dining_rating", "restaurant_delivery_rating",
            "restaurant_dining_ratings_count", "restaurant_delivery_ratings_count", 
            "restaurant_source_url", "description_clean", "city", 
            "created_at", "updated_at"
        ]
        
        cleaned_filters = {}
        for field, value in metadata_filters.items():
            if field in valid_fields:
                # Handle boolean conversions appropriately
                if field == "item_is_veg" and isinstance(value, str):
                    if value.lower() in ["true", "yes", "1"]:
                        cleaned_filters[field] = True
                    elif value.lower() in ["false", "no", "0"]:
                        cleaned_filters[field] = False
                # Handle numeric conversions
                elif field in ["item_price", "restaurant_dining_rating", "restaurant_delivery_rating"] and isinstance(value, str):
                    try:
                        cleaned_filters[field] = float(value)
                    except ValueError:
                        logger.warning(f"Could not convert {field} value '{value}' to number, skipping filter")
                else:
                    cleaned_filters[field] = value
            else:
                logger.warning(f"Ignoring invalid metadata field: {field}")
        
        # If item_is_veg filter exists, ensure it's properly formatted
        if "item_is_veg" in cleaned_filters:
            cleaned_filters["item_is_veg"] = bool(cleaned_filters["item_is_veg"])
        
        logger.info(f"Using cleaned metadata filters: {cleaned_filters}")
        
        # Execute the search with properly formatted filters
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=cleaned_filters if cleaned_filters else None
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Vector search executed in {execution_time:.2f}s")
        
        # Convert Pinecone response to a serializable dict to prevent deepcopy issues
        serializable_results = {}
        if hasattr(results, "to_dict"):
            serializable_results = results.to_dict()
        else:
            # Manual conversion if to_dict is not available
            serializable_results = {
                "matches": []
            }
            if hasattr(results, "matches"):
                for match in results.matches:
                    match_dict = {}
                    if hasattr(match, "id"):
                        match_dict["id"] = match.id
                    if hasattr(match, "score"):
                        match_dict["score"] = float(match.score)
                    if hasattr(match, "metadata"):
                        match_dict["metadata"] = dict(match.metadata)
                    serializable_results["matches"].append(match_dict)
        
        return {"results": serializable_results, "query_text": query_text, "metadata_filters": cleaned_filters}
    
    except Exception as e:
        logger.error(f"Error executing vector search: {str(e)}")
        return {"error": str(e), "query_text": query_params.get("query_text", "")}

# --- Agent Functions ---

async def generate_queries(state: AgentState) -> AgentState:
    """Generate SQL queries based on user query using OpenAI LLM"""
    logger.info(f"Generating queries for iteration {state.current_iteration}")

    # The LLMClient is now the OpenAI one
    llm_client = LLMClient()
    llm_response = llm_client.generate_sql(
        query=state.user_query,
        iteration=state.current_iteration,
        contexts=state.contexts
    )

    # Logic for handling first iteration vs subsequent iterations remains the same
    if state.current_iteration == 0:
        queries = [q.strip() for q in llm_response.split('---') if q.strip()]
        logger.info(f"Initial queries generated: {queries}")
        for query in queries:
            clean_query = query
            if "```sql" in clean_query:
                clean_query = clean_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in clean_query:
                 # Handle potential plain ``` blocks without language specified
                 parts = clean_query.split("```")
                 if len(parts) >= 3: # e.g., ```\nQUERY\n```
                     clean_query = parts[1].strip()
                 else: # If it's just ```QUERY``` or similar, try to strip ```
                     clean_query = clean_query.replace("```", "").strip()

            state.add_sql_query(clean_query)
    else:
        if llm_response.startswith("END"):
            final_response = llm_response[3:].strip() # Get text after "END"
            state.final_response = final_response
            state.is_complete = True
            logger.info("Query generation complete - sufficient information collected.")
        elif llm_response: # Ensure we got a response before treating it as a query
            additional_query = llm_response.strip()
            logger.info(f"Additional query generated: {additional_query}")

            # Clean up the query (remove markdown formatting if present)
            if "```sql" in additional_query:
                additional_query = additional_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in additional_query:
                 parts = additional_query.split("```")
                 if len(parts) >= 3:
                     additional_query = parts[1].strip()
                 else:
                     additional_query = additional_query.replace("```", "").strip()

            state.add_sql_query(additional_query)
            logger.info(f"Added additional query in iteration {state.current_iteration}")
        else:
             # Handle cases where the LLM might return an empty response unexpectedly
             logger.warning(f"LLM returned an empty response in iteration {state.current_iteration}. Assuming completion.")
             state.final_response = "The AI assistant could not determine the next step or final answer based on the available information."
             state.is_complete = True


    return state

async def execute_queries(state: AgentState) -> AgentState:
    """Execute the SQL queries and store results"""
    logger.info(f"Executing queries for iteration {state.current_iteration}")

    num_executed = len(state.sql_results)
    # Only execute queries that haven't been executed yet
    new_queries = state.sql_queries[num_executed:]

    if not new_queries and not state.is_complete:
         logger.warning(f"Iteration {state.current_iteration}: No new queries to execute, but process is not marked complete. Forcing completion.")
         state.is_complete = True # Avoid potential infinite loops if generate_queries fails to produce output or END
         if not state.final_response:
              state.final_response = "Could not generate further queries or determine a final answer."
         return state # Skip execution if no new queries

    tasks = []
    for query in new_queries:
        logger.info(f"Preparing to execute query: {query}")
        if query: # Ensure query is not empty
             tasks.append(execute_sql_query(query))
        else:
             logger.warning("Skipping empty query string.")

    if tasks:
        query_results = await asyncio.gather(*tasks)
        for result in query_results:
             state.add_sql_result(result)

    # Increment iteration counter ONLY if queries were actually processed in this step
    # Or if we generated a final answer in the previous step
    if new_queries or state.is_complete:
        state.increment_iteration()

    return state

async def combine_contexts(state: AgentState) -> AgentState:
    """Combine contexts from SQL and Pinecone searches"""
    logger.info("Combining contexts from SQL and Pinecone searches")
    
    if not state.sql_results and not state.pinecone_results:
        logger.warning("No results to combine")
        return state
    
    # Create a combined context string
    combined_context = "Combined Search Results:\n\n"
    
    # Add SQL results
    if state.sql_results:
        combined_context += "== SQL Database Results ==\n"
        for i, result in enumerate(state.sql_results):
            combined_context += f"SQL Query #{i+1}:\n{state.sql_queries[i]}\n\n"
            # Limit the number of results shown if there are too many
            result_sample = result[:10] if len(result) > 10 else result
            combined_context += f"Results (showing {len(result_sample)} of {len(result)}):\n{json.dumps(result_sample, indent=2, default=str)}\n\n"
        print(f"SQL Results: {combined_context}")
    
    # Add Pinecone results
    if state.pinecone_results:
        combined_context += "== Vector Database Results ==\n"
        for i, result in enumerate(state.pinecone_results):
            query = state.pinecone_queries[i] if i < len(state.pinecone_queries) else {}
            combined_context += f"Vector Search #{i+1}:\n"
            combined_context += f"Query Text: {query.get('query_text', 'N/A')}\n"
            combined_context += f"Metadata Filters: {json.dumps(query.get('metadata_filters', {}), indent=2)}\n\n"
            
            if "results" in result and "matches" in result["results"]:
                matches = result["results"]["matches"]
                combined_context += f"Top {len(matches)} Matches:\n"
                
                for j, match in enumerate(matches):
                    metadata = match.get("metadata", {})
                    score = match.get("score", 0)
                    combined_context += f"Match #{j+1} (Similarity Score: {score:.4f}):\n"
                    # Add most important fields first
                    combined_context += f"  Restaurant: {metadata.get('restaurant_name', 'N/A')}\n"
                    combined_context += f"  Item: {metadata.get('item_name', 'N/A')}\n"
                    combined_context += f"  Price: {metadata.get('item_price', 'N/A')}\n"
                    combined_context += f"  Category: {metadata.get('item_category', 'N/A')}\n"
                    combined_context += f"  Vegetarian: {metadata.get('item_is_veg', 'N/A')}\n"
                    # Only add description if it exists and isn't empty
                    description = metadata.get('item_description', '') or metadata.get('description_clean', '')
                    if description and description != 'N/A':
                        combined_context += f"  Description: {description}\n"
                    combined_context += "\n"
            elif "error" in result:
                combined_context += f"Error: {result.get('error', 'Unknown error')}\n\n"
            else:
                combined_context += "No matches found in vector search.\n\n"
        print(f"Pinecone Results: {combined_context}")
    
    state.combined_context = combined_context
    logger.info("Contexts combined successfully")
    
    return state

async def judge_combined_context(state: AgentState) -> AgentState:
    """Evaluate the combined context and make final refinements before generating response"""
    logger.info("Judging combined context before final response")
    
    if not state.combined_context:
        logger.warning("No combined context available for judging")
        return state
    
    llm_client = LLMClient()
    
    system_prompt = """You are an AI assistant that evaluates combined search results to determine the most relevant information for the user's query.
    
Review the combined context from both SQL and vector database searches, and determine:
1. Which pieces of information are most relevant to the user's query
2. If any contradictions exist between sources, which source is likely more reliable
3. What additional information might be useful to include in the final response
4. How to best structure the final response for clarity and completeness

Respond with a JSON object containing:
- "relevant_info": Array of the most relevant pieces of information (max 5)
- "contradictions": Any contradictions found between sources
- "structure_recommendation": How to structure the final response
- "additional_notes": Any other insights about the data that should inform the final response

This assessment will be used to generate a more accurate and helpful final response."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {state.user_query}\n\nCombined Context:\n{state.combined_context}\n\nPlease evaluate this combined information and provide guidance for the final response."}
    ]
    
    try:
        if isinstance(llm_client.client, openai.OpenAI): # For openai >= 1.0.0
            completion = llm_client.client.chat.completions.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
                response_format={"type": "json_object"} # Force JSON response format
            )
            response = completion.choices[0].message.content
        else: # Fallback for openai < 1.0.0
            completion = llm_client.client.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3
            )
            response = completion.choices[0].message.content
        
        # Clean and parse the response
        response = response.strip()
        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0].strip()
        elif response.startswith("```"):
            response = response.split("```")[1].split("```")[0].strip()
        
        # Parse the judgment
        judgment = json.loads(response)
        
        # Add the judgment to state
        state.context_judgment = judgment
        logger.info(f"Context judgment completed: {judgment.get('relevant_info', [])[0] if judgment.get('relevant_info') else 'No relevant info identified'}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in judge_combined_context: {str(e)}")
        # Create a default judgment if there's an error
        state.context_judgment = {
            "relevant_info": ["All information appears relevant"],
            "contradictions": "Unable to evaluate contradictions due to error",
            "structure_recommendation": "Present information in a clear, organized manner",
            "additional_notes": f"Error during judgment: {str(e)}"
        }
        return state

async def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response using combined context from SQL and Pinecone"""
    logger.info("Generating final response using combined context")
    
    # Only generate final response if not already complete
    if state.is_complete and state.final_response:
        return state
    
    llm_client = LLMClient()
    
    system_prompt = """You are an AI assistant that helps provide comprehensive answers about restaurants and menu items.
You have access to both structured data from a SQL database and semantic search results from a vector database.
Synthesize the information from both sources to provide the most complete and accurate answer to the user's question.
Focus on being concise, accurate, and helpful. Format the response in a readable way with markdown if appropriate."""

    # Prepare the messages with combined context
    context = state.combined_context if state.combined_context else "\n\n".join(state.contexts)
    
    # Include judgment guidance if available
    judgment_guidance = ""
    if hasattr(state, "context_judgment") and state.context_judgment:
        judgment = state.context_judgment
        judgment_guidance = "Based on analysis of the information, please note:\n"
        
        if "relevant_info" in judgment and judgment["relevant_info"]:
            judgment_guidance += "Most relevant information:\n"
            for info in judgment["relevant_info"]:
                judgment_guidance += f"- {info}\n"
        
        if "contradictions" in judgment and judgment["contradictions"]:
            judgment_guidance += f"\nPotential contradictions: {judgment['contradictions']}\n"
        
        if "structure_recommendation" in judgment and judgment["structure_recommendation"]:
            judgment_guidance += f"\nRecommended structure: {judgment['structure_recommendation']}\n"
        
        if "additional_notes" in judgment and judgment["additional_notes"]:
            judgment_guidance += f"\nAdditional context: {judgment['additional_notes']}\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Question: {state.user_query}\n\nAvailable Information:\n{context}\n\n{judgment_guidance}\nPlease provide a comprehensive answer based on all available information."}
    ]
    
    try:
        if isinstance(llm_client.client, openai.OpenAI): # For openai >= 1.0.0
            completion = llm_client.client.chat.completions.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7 # Higher temperature for more creative response
            )
            response = completion.choices[0].message.content
        else: # Fallback for openai < 1.0.0
            completion = llm_client.client.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            response = completion.choices[0].message.content

        state.final_response = response
        state.is_complete = True
        logger.info(f"Generated final response: {response[:100]}...")
        
    except Exception as e:
        logger.error(f"Error generating final response: {str(e)}")
        state.final_response = f"Error generating response: {str(e)}"
        state.is_complete = True
    
    return state

# should_continue remains the same
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if we should continue with another iteration or end"""
    if state.is_complete:
        logger.info("Process complete, ending agent execution")
        return "end"
    elif state.current_iteration >= 5: # Add a max iteration limit
         logger.warning("Maximum iterations reached (5). Ending process.")
         state.final_response = "Reached maximum iterations. Could not fully complete the request."
         return "end"
    else:
        logger.info(f"Continuing to iteration {state.current_iteration}")
        return "continue"

# --- Add Judge Function to Determine Search Strategy ---
async def judge_query(state: AgentState) -> AgentState:
    """Determine whether to use SQL, Pinecone, both, or neither based on the query"""
    logger.info(f"Judging query: {state.user_query}")
    
    llm_client = LLMClient()
    
    system_prompt = """You are an AI assistant that analyzes user queries to determine the most appropriate search strategy.
You have access to two types of search:

1. SQL Database Search:
   - Good for: Precise queries about specific restaurant menu items, prices, ratings, etc.
   - Schema: {DB_SCHEMA}

2. Vector Database Search (Pinecone):
    - Good for: Semantic similarity, conceptual queries, or queries about dish characteristics
    - Contains the same data as SQL but allows semantic search
    - Use It almost EVERY TIME , Recommended to use it 90 perecent of the time with no metadata filters

Analyze the user query and determine whether SQL, Vector search, both, or neither should be used.
Respond with a JSON object containing:
- "sql": 1 if SQL search should be used, 0 if not
- "pinecone": 1 if Vector search should be used, 0 if not
- "explanation": A brief explanation of your decision

Only respond with the JSON object, no additional text.""".format(DB_SCHEMA=DB_SCHEMA)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {state.user_query}\nWhat search strategy should be used?"}
    ]
    
    try:
        if isinstance(llm_client.client, openai.OpenAI): # For openai >= 1.0.0
            completion = llm_client.client.chat.completions.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.2,
                response_format={"type": "json_object"} # Force JSON response format
            )
            response = completion.choices[0].message.content
        else: # Fallback for openai < 1.0.0
            completion = llm_client.client.create(
                model=llm_client.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.2
            )
            response = completion.choices[0].message.content
        
        # Clean and parse the response
        response = response.strip()
        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0].strip()
        elif response.startswith("```"):
            response = response.split("```")[1].split("```")[0].strip()
        
        decision = json.loads(response)
        sql_decision = int(decision.get("sql", 0))
        pinecone_decision = int(decision.get("pinecone", 0))
        explanation = decision.get("explanation", "No explanation provided")
        
        # Store the decision in the state
        state.search_decision = {
            "sql": sql_decision,
            "pinecone": pinecone_decision,
            "explanation": explanation
        }
        
        logger.info(f"Search decision: SQL={sql_decision}, Pinecone={pinecone_decision}")
        logger.info(f"Explanation: {explanation}")
        
        # If both are 0, set a polite response that the query cannot be answered
        if sql_decision == 0 and pinecone_decision == 0:
            state.final_response = "I apologize, but I don't have enough information to answer your question. The query appears to be outside the scope of our restaurant and menu item database. Would you like to ask something about restaurants, their menu items, prices, or ratings instead?"
            state.is_complete = True
            logger.info("Query cannot be answered with available data sources.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in judge_query: {str(e)}")
        # Default to using both search methods if there's an error
        state.search_decision = {"sql": 1, "pinecone": 1, "explanation": f"Error in decision: {str(e)}"}
        return state

# Route based on judge decision
def route_based_on_decision(state: AgentState) -> Literal["sql_only", "pinecone_only", "both", "none"]:
    """Route the workflow based on the judge's decision"""
    sql = state.search_decision.get("sql", 1)
    pinecone = state.search_decision.get("pinecone", 1)
    
    if sql == 1 and pinecone == 1:
        return "both"
    elif sql == 1:
        return "sql_only"
    elif pinecone == 1:
        return "pinecone_only"
    else:
        return "none"

# --- Build the updated LangGraph with judge node ---
def build_agent():
    checkpoint = MemorySaver()
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("judge_query", judge_query)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("execute_queries", execute_queries)
    workflow.add_node("generate_pinecone_query", generate_pinecone_query)
    workflow.add_node("execute_pinecone_search", execute_pinecone_search)
    workflow.add_node("combine_contexts", combine_contexts)
    workflow.add_node("judge_combined_context", judge_combined_context)  # Add the new judge node
    workflow.add_node("generate_final_response", generate_final_response)

    # Set entry point
    workflow.set_entry_point("judge_query")

    # Routing from Judge
    workflow.add_conditional_edges(
        "judge_query",
        route_based_on_decision,
        {
            "sql_only": "generate_queries",
            "pinecone_only": "generate_pinecone_query",
            "both": "generate_queries",
            "none": END # If judge decides neither, end immediately
        }
    )

    # Routing from SQL Generation
    # Check if SQL generation already decided to end or if queries need execution
    workflow.add_conditional_edges(
        "generate_queries",
        check_sql_needed_or_complete,
        {
            "execute_sql": "execute_queries",
            "generate_final_response": "generate_final_response", # SQL LLM call decided it's finished
            "end_loop": END # Safety break
        }
    )

    # Routing after SQL Execution
    # Decide whether to loop back for more SQL, proceed to Pinecone, or finish
    workflow.add_conditional_edges(
        "execute_queries",
        decide_next_step_after_sql,
        {
            "generate_pinecone_query": "generate_pinecone_query", # For 'both' path
            "generate_queries": "generate_queries",           # For 'sql_only' path loop
            "generate_final_response": "generate_final_response" # If SQL execution led to completion
        }
    )

    # Routing for Pinecone Path (Pinecone-only or Both)
    workflow.add_edge("generate_pinecone_query", "execute_pinecone_search")
    workflow.add_edge("execute_pinecone_search", "combine_contexts") # Always combine after Pinecone search
    
    # Add the new judge after combining contexts
    workflow.add_edge("combine_contexts", "judge_combined_context")
    workflow.add_edge("judge_combined_context", "generate_final_response")

    # Final step
    workflow.add_edge("generate_final_response", END) # End after generating the final response

    return workflow.compile(checkpointer=checkpoint)

# --- Main function to run the agent (Updated for combined SQL and Pinecone search) ---
async def run_agent(query: str) -> Dict[str, Any]:
    """Run the agent on a query and return the result"""
    logger.info(f"Starting OpenAI agent for query: {query}")
    start_time = time.time()

    # Variables to store final results, resilient to extraction issues
    final_sql_queries = []
    final_pinecone_queries = []
    final_answer = "Error: Could not extract final state."
    final_iterations = 0
    search_decision = {}

    try:
        initial_state = AgentState(user_query=query)
        agent = build_agent() # Builds the graph with OpenAI nodes

        thread_id = f"openai_sql_pinecone_agent_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}} 

        # Run the agent without streaming to avoid serialization issues
        try:
            final_state = await agent.ainvoke(initial_state, config=config)
            # Extract results from the final state
            if isinstance(final_state, AgentState):
                final_sql_queries = final_state.sql_queries
                final_pinecone_queries = final_state.pinecone_queries
                final_answer = final_state.final_response if final_state.final_response else "No final response generated."
                final_iterations = final_state.current_iteration
                search_decision = final_state.search_decision
                logger.info(f"Agent completed. Iterations: {final_iterations}. Final Answer: {final_answer[:100]}...")
            elif isinstance(final_state, dict):
                # Handle dict representation
                final_sql_queries = final_state.get("sql_queries", [])
                final_pinecone_queries = final_state.get("pinecone_queries", [])
                final_answer = final_state.get("final_response", "No final response generated.")
                final_iterations = final_state.get("current_iteration", 0)
                search_decision = final_state.get("search_decision", {})
                logger.info(f"Agent completed (dict state). Iterations: {final_iterations}. Final Answer: {final_answer[:100]}...")
        except Exception as e:
            logger.error(f"Error during agent execution: {str(e)}", exc_info=True)
            final_answer = f"Error during agent execution: {str(e)}"

        execution_time = time.time() - start_time
        logger.info(f"Agent execution completed in {execution_time:.2f}s")

        output = {
            "query": query,
            "sql_queries": final_sql_queries,
            "pinecone_queries": final_pinecone_queries,
            "iterations": final_iterations,
            "final_response": final_answer,
            "execution_time": f"{execution_time:.2f}s",
            "search_decision": search_decision
        }

        # Format response if needed (same as before)
        if final_answer and final_answer != "No final response generated.":
             try:
                  formatted_response = final_answer
                  output["final_response"] = formatted_response.replace("\\n", "\n").replace("\\t", "\t")
                  logger.info(f"Using final response: {output['final_response'][:100]}...")
             except Exception as e:
                  logger.error(f"Error formatting response: {str(e)}")
                  output["final_response"] = final_answer # Fallback

        # Default if no response
        if not final_answer or final_answer == "No final response generated.":
            output["final_response"] = "Unable to generate a response. Please try rephrasing your question."
            logger.warning("No valid response generated by the agent.")

        return output

    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        execution_time = time.time() - start_time
        return {
            "query": query,
            "sql_queries": final_sql_queries,
            "pinecone_queries": final_pinecone_queries,
            "iterations": final_iterations,
            "final_response": f"Error during agent execution: {str(e)}",
            "execution_time": f"{execution_time:.2f}s",
            "search_decision": search_decision,
            "error": str(e)
        }

# --- Entry point (Updated to include Pinecone query info) ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Example query
        # query = "List the top 3 restaurants based on delivery ratings count."
        # query = "What are the top 5 most expensive vegetarian dishes and which restaurants serve them?"
        query="spicy burgers below 400"

    # Setup environment variable if running directly and it's not set
    if not os.environ.get("OPENAI_API_KEY"):
         print("Warning: OPENAI_API_KEY not found in environment. Attempting to load from .env file.")
         if not load_dotenv():
              print("Error: Could not load .env file. Please set the OPENAI_API_KEY environment variable.")
              sys.exit(1)
         if not os.environ.get("OPENAI_API_KEY"):
              print("Error: OPENAI_API_KEY still not found after attempting to load .env. Please set the environment variable.")
              sys.exit(1)
    
    # Check for Pinecone API key
    if not os.environ.get("PINECONE_API_KEY"):
         print("Warning: PINECONE_API_KEY not found in environment. Vector search functionality may not be available.")
         load_dotenv() # Try to load again in case it's in .env

    print(f'Running OpenAI Agent for query: "{query}"')
    result = asyncio.run(run_agent(query))

    # Print the result
    print("\n" + "="*80)
    print("ORIGINAL QUERY:")
    print(result.get("query", "N/A"))
    print("\n" + "="*80)
    
    # Print judge's decision if available
    if "search_decision" in result:
        print("SEARCH STRATEGY DECISION:")
        decision = result["search_decision"]
        print(f"SQL Search: {'✓' if decision.get('sql', 0) == 1 else '✗'}")
        print(f"Vector Search: {'✓' if decision.get('pinecone', 0) == 1 else '✗'}")
        if "explanation" in decision:
            print(f"Explanation: {decision['explanation']}")
        print("\n" + "="*80)

    print("SQL QUERIES EXECUTED:")
    sql_queries = result.get("sql_queries", [])
    if sql_queries:
        for i, sql_query in enumerate(sql_queries):
            print(f"\nQuery {i+1}:")
            print(sql_query)
    else:
        print("No SQL queries were executed or recorded.")
    
    print("\n" + "="*80)
    print("PINECONE VECTOR SEARCHES:")
    pinecone_queries = result.get("pinecone_queries", [])
    if pinecone_queries:
        for i, query_params in enumerate(pinecone_queries):
            print(f"\nVector Search {i+1}:")
            print(f"Query Text: {query_params.get('query_text', 'N/A')}")
            print(f"Metadata Filters: {json.dumps(query_params.get('metadata_filters', {}), indent=2)}")
    else:
        print("No vector searches were executed or recorded.")

    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print(result.get("final_response", "N/A"))
    print("\n" + "="*80)
    print(f"Completed in {result.get('execution_time', 'N/A')} with {result.get('iterations', 'N/A')} iterations")
    print("="*80)
    if "error" in result:
         print(f"Error encountered: {result['error']}")
         print("="*80) 