import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from agentic_openai_pipeline import run_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log", encoding='utf-8'),
        logging.StreamHandler()
    ],
    encoding='utf-8'
)
logger = logging.getLogger("restaurant_api")

# Create FastAPI app
app = FastAPI(
    title="Restaurant Search API",
    description="API for searching restaurant and menu information",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or bot)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[str] = Field(None, description="Timestamp of the message")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list, 
        description="Previous chat messages (up to 5 exchanges)"
    )

# Response models
class SearchDecision(BaseModel):
    sql: int = Field(..., description="Whether SQL search was used (0 or 1)")
    pinecone: int = Field(..., description="Whether vector search was used (0 or 1)")
    chat_history_lookup: int = Field(..., description="Whether chat history lookup was used (0 or 1)")
    conversational: int = Field(..., description="Whether conversational response was used (0 or 1)")
    explanation: str = Field(..., description="Explanation of search strategy")
    history_context: Optional[str] = Field(None, description="Extracted context from chat history")
    suggested_response: Optional[str] = Field(None, description="Suggested conversational response")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Response to the query")
    search_decision: SearchDecision = Field(..., description="Search strategy decision")
    execution_time: str = Field(..., description="Time taken to execute the query")
    sql_queries: Optional[List[str]] = Field(None, description="SQL queries executed")
    pinecone_queries: Optional[List[Dict[str, Any]]] = Field(None, description="Vector search queries executed")

# Background tasks
async def process_query_async(query: str, chat_history: List[ChatMessage]) -> Dict[str, Any]:
    """Process user query in the background"""
    try:
        # Format chat history for the agent
        formatted_history = []
        for msg in chat_history:
            formatted_history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp if msg.timestamp else ""
            })
        
        # Add chat history to query if available
        if formatted_history:
            chat_history_json = json.dumps(formatted_history)
            enhanced_query = f"{query}\nContext from previous conversation: {chat_history_json}"
        else:
            enhanced_query = query
            
        # Run the agent
        result = await run_agent(enhanced_query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# API endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a restaurant search query
    
    This endpoint accepts a query and optional chat history,
    processes it through the restaurant search assistant,
    and returns the response.
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        # Limit chat history to the most recent 5 exchanges (10 messages)
        limited_history = request.chat_history
        if len(limited_history) > 10:
            limited_history = limited_history[-10:]
            logger.info(f"Trimmed chat history to last 10 messages")
        
        # Process the query
        result = await process_query_async(request.query, limited_history)
        
        # Extract search decision from the result
        search_decision = result.get("search_decision", {})
        if not search_decision:
            search_decision = {
                "sql": 0,
                "pinecone": 0, 
                "chat_history_lookup": 0,
                "conversational": 1,
                "explanation": "No search strategy provided, defaulting to conversational",
                "suggested_response": None
            }
        
        # Ensure all fields are in the search decision
        for field in ["sql", "pinecone", "chat_history_lookup", "conversational"]:
            if field not in search_decision:
                search_decision[field] = 0
        
        if "explanation" not in search_decision:
            search_decision["explanation"] = "No explanation provided"
        
        # Format the response
        response = {
            "query": request.query,
            "response": result.get("final_response", "I couldn't process your query."),
            "search_decision": search_decision,
            "execution_time": result.get("execution_time", "N/A"),
            "sql_queries": result.get("sql_queries", []),
            "pinecone_queries": result.get("pinecone_queries", [])
        }
        
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Restaurant Search API"}

# Run the FastAPI app with Uvicorn if executed directly
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    ) 