import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import re
from sqlalchemy import inspect
import traceback
from uuid import uuid4
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
load_dotenv()

# Import agent classes from agents.py
from agents import (
    LlamaCppInterface, ModelInterface, GuardAgent, IntentClassificationAgent,
    ContextTrackingAgent, VisualizationAgent, SQLGenerator, DynamicSuggestionAgent,
    ExplanationAgent, FeedbackAgent, DatabaseManager, QueryOrchestrator, GroqInterface
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sql-ai-app")

# Initialize FastAPI app
app = FastAPI(
    title="AI SQL Assistant API",
    description="API for intelligent database querying using natural language",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static file directory for frontend
os.makedirs("static", exist_ok=True)

# Pydantic models for request/response
class ConnectionRequest(BaseModel):
    db_type: str
    host: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    db_path: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class CancelQueryRequest(BaseModel):
    query_id: str

class SuggestionRequest(BaseModel):
    partial_query: str

class FeedbackRequest(BaseModel):
    query_id: str
    rating: str  # "good", "neutral", "bad"

class ModelLoadRequest(BaseModel):
    model_path: str
    device: str = "cuda"  # Use "cpu" if no GPU is available
    max_context_length: int = 4096  # Adjust based on model capabilities

# Global variables
db_manager = None
query_orchestrator = None
active_queries = {}  # Track active queries by their IDs
groq_model_name = os.environ.get("GROQ_MODEL_NAME", "llama3-8b-8192")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Startup check for API key
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment before starting the backend.")

def get_db_manager() -> DatabaseManager:
    """Get the database manager. If not initialized, raise an exception."""
    global db_manager
    if db_manager is None:
        raise HTTPException(
            status_code=400, 
            detail="Database manager not initialized."
        )
    return db_manager

def get_query_orchestrator() -> QueryOrchestrator:
    """Get the query orchestrator. If not initialized, raise an exception."""
    global query_orchestrator
    if query_orchestrator is None:
        raise HTTPException(
            status_code=400, 
            detail="Query orchestrator not initialized. Please connect to a database first."
        )
    return query_orchestrator


@app.post("/database/connect")
async def connect_database(
    request: ConnectionRequest
):
    """
    Connect to a database with the provided credentials.
    """
    global db_manager, query_orchestrator
    try:
        # Check if we already have a connection to the same database
        if (db_manager is not None and 
            db_manager.connection is not None and 
            db_manager.current_db == request.database and 
            db_manager.current_host == request.host and 
            db_manager.current_port == request.port):
            # Reuse existing connection
            return {
                "status": "success",
                "message": "Using existing database connection",
                "schema": db_manager.get_schema(),
                "tables": db_manager.get_tables()
            }
        # If we have an existing connection, close it
        if db_manager is not None and db_manager.connection is not None:
            db_manager.disconnect()
        # Create new database manager if needed
        if db_manager is None:
            db_manager = DatabaseManager()
        # Connect to the database
        success, message = db_manager.connect(
            db_type=request.db_type,
            host=request.host,
            user=request.user,
            password=request.password,
            port=request.port,
            database=request.database,
            db_path=request.db_path
        )
        if not success:
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
        # Initialize the query orchestrator only if it doesn't exist or if we changed databases
        if query_orchestrator is None:
            query_orchestrator = QueryOrchestrator(
                db_manager=db_manager,
                api_key=groq_api_key,
                model_name=groq_model_name
            )
        # Return the database schema
        schema = db_manager.get_schema()
        tables = db_manager.get_tables()
        return {
            "status": "success",
            "message": message,
            "schema": schema,
            "tables": tables
        }
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect to database: {str(e)}"}
        )

@app.get("/database/schema")
async def get_schema(db_manager: DatabaseManager = Depends(get_db_manager)):
    """
    Get the database schema.
    """
    try:
        schema = db_manager.get_schema()
        tables = db_manager.get_tables()
        
        return {
            "schema": schema,
            "tables": tables
        }
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get schema: {str(e)}"}
        )

async def process_query_task(query_id: str, query_text: str, orchestrator: QueryOrchestrator):
    """Process a query in a separate task that can be cancelled."""
    try:
        # Check if query is already cancelled before processing
        if query_id not in active_queries:
            logger.info(f"Query {query_id} was cancelled before processing started")
            return None
        
        # Process the query - directly await the async method
        result = await orchestrator.process_query(query_text)
        
        # Check if query was cancelled during processing
        if query_id not in active_queries:
            logger.info(f"Query {query_id} was cancelled during processing")
            return None
            
        # Clean up when done
        if query_id in active_queries:
            del active_queries[query_id]
            
        return result
    except asyncio.CancelledError:
        logger.info(f"Query {query_id} was cancelled during processing")
        # Clean up when cancelled
        if query_id in active_queries:
            del active_queries[query_id]
        return None
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Clean up on error
        if query_id in active_queries:
            del active_queries[query_id]
        raise

@app.post("/query")
async def process_query(
    request: Request, 
    query_request: QueryRequest, 
    background_tasks: BackgroundTasks,
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator)
):
    """Process a natural language query with enhanced cancellation support."""
    try:
        # Generate a unique ID for this query
        query_id = str(uuid4())
        logger.info(f"Starting new query with ID: {query_id}")
        
        # Create a task for processing the query
        task = asyncio.create_task(process_query_task(query_id, query_request.query, orchestrator))
        
        # Store the task so it can be cancelled later
        active_queries[query_id] = task
        
        # Wait for the task to complete or be cancelled
        try:
            result = await task
            
            # If result is None, the query was cancelled or failed
            if result is None:
                logger.info(f"Query {query_id} returned None (likely cancelled)")
                return JSONResponse(
                    status_code=499,  # Client Closed Request
                    content={"error": "Query was cancelled or failed", "query_id": query_id}
                )
                
            # Add query_id to the result for client reference
            result["query_id"] = query_id
            logger.info(f"Query {query_id} completed successfully")
            return result
            
        except asyncio.CancelledError:
            logger.info(f"Query {query_id} was cancelled")
            return JSONResponse(
                status_code=499,  # Client Closed Request
                content={"error": "Query was cancelled", "query_id": query_id}
            )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel-query")
async def cancel_query(request: CancelQueryRequest):
    """Cancel an ongoing query by ID."""
    query_id = request.query_id
    
    logger.info(f"Received cancellation request for query {query_id}")
    
    if query_id in active_queries:
        logger.info(f"Found active query {query_id}, attempting to cancel")
        
        # Cancel the task
        task = active_queries[query_id]
        task.cancel()
        
        # Give the task a moment to handle the cancellation
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # This is expected - the task was cancelled
            pass
        except Exception as e:
            logger.error(f"Error during cancellation of query {query_id}: {str(e)}")
        
        # Clean up
        if query_id in active_queries:
            del active_queries[query_id]
        
        return {
            "status": "success",
            "message": f"Query {query_id} cancelled successfully"
        }
    else:
        logger.info(f"Query {query_id} not found in active queries")
        return JSONResponse(
            status_code=404,
            content={"error": f"Query {query_id} not found or already completed"}
        )

@app.post("/suggestions")
async def get_suggestions(
    request: SuggestionRequest,
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator)
):
    """
    Get dynamic suggestions based on partial user input.
    Uses the DynamicSuggestionAgent to generate context-aware suggestions.
    """
    try:
        # Get suggestions from the orchestrator's suggestion agent
        suggestions = orchestrator.get_suggestions(request.partial_query)
        
        # Log the suggestions for debugging
        logger.debug(f"Generated suggestions for '{request.partial_query}': {suggestions}")
        
        return {
            "status": "success",
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Failed to get suggestions: {str(e)}",
                "suggestions": []  # Return empty suggestions on error
            }
        )

@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator)
):
    """
    Submit feedback for a query.
    """
    try:
        orchestrator.record_feedback(request.query_id, request.rating)
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to record feedback: {str(e)}"}
        )

@app.post("/execute-sql")
async def execute_sql(query: str = Form(...), db_manager: DatabaseManager = Depends(get_db_manager)):
    """
    Execute a raw SQL query (for testing purposes).
    """
    try:
        # Only allow SELECT statements for security
        if not re.match(r'^\s*SELECT\s+', query, re.IGNORECASE):
            return JSONResponse(
                status_code=400,
                content={"error": "Only SELECT statements are allowed"}
            )
            
        results, error = db_manager.execute_query(query)
        
        if error:
            return JSONResponse(
                status_code=400,
                content={"error": error}
            )
            
        return {
            "results": results
        }
    except Exception as e:
        logger.error(f"Error executing SQL: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to execute SQL: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    status = {
        "status": "healthy",
        "db_connected": db_manager is not None and db_manager.connection is not None,
        "orchestrator_ready": query_orchestrator is not None,
        "active_queries": len(active_queries)
    }
    
    return status

@app.get("/active-queries")
async def get_active_queries():
    """
    Get a list of active query IDs.
    """
    return {
        "active_queries": list(active_queries.keys())
    }

@app.get("/api/feedback/accuracy")
def get_feedback_accuracy():
    feedback_file = r"D:\My App\Backend\feedback.jsonl"
    if not os.path.exists(feedback_file):
        return {"accuracy": 0}
    good = 0
    bad = 0
    with open(feedback_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('rating') == 'good':
                    good += 1
                elif entry.get('rating') == 'bad':
                    bad += 1
            except Exception:
                continue
    total = good + bad
    accuracy = int((good / total) * 100) if total else 0
    return {"accuracy": accuracy}

@app.get("/api/stats")
def get_stats():
    query_times_file = "Backend/query_times.json"
    manual_time_sec = 5 * 60  # 5 minutes in seconds

    if not os.path.exists(query_times_file):
        return {"time_saved_hr": 0, "queries_today": 0}

    try:
        with open(query_times_file, "r") as f:
            entries = json.load(f)
    except Exception:
        return {"time_saved_hr": 0, "queries_today": 0}

    total_time_saved_sec = 0
    for entry in entries:
        exec_time = entry.get("execution_time_sec", 0)
        saved = max(manual_time_sec - exec_time, 0)
        total_time_saved_sec += saved

    total_time_saved_hr = round(total_time_saved_sec / 3600, 1)
    queries_today = len(entries)

    return {
        "time_saved_hr": total_time_saved_hr,
        "queries_today": queries_today
    }

# Add shutdown endpoint to properly clean up resources
@app.on_event("shutdown")
def shutdown_event():
    """
    Clean up resources when shutting down.
    """
    global active_queries
    
    # Cancel all active queries
    for query_id, task in active_queries.items():
        try:
            task.cancel()
            logger.info(f"Cancelled query {query_id} during shutdown")
        except Exception as e:
            logger.error(f"Error cancelling query {query_id}: {str(e)}")
    
    active_queries = {}
    
    if db_manager is not None:
        db_manager.disconnect()
    logger.info("Application shutting down")

# Optional: Serve static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run the application if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)