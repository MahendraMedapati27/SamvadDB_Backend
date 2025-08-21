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
# Load env from current working directory
load_dotenv()
# Additionally load from project root and Backend directory to be robust
try:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir.parent / ".env")
    load_dotenv(base_dir / ".env")
except Exception:
    pass
import time
from datetime import datetime

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

class ConnectionMetadata(BaseModel):
    connection_id: str
    name: str
    db_type: str
    host: str
    database: str
    port: Optional[int] = None
    username: Optional[str] = None
    is_favorite: bool = False
    tags: List[str] = []
    environment: str = "Development"
    last_accessed: Optional[str] = None
    query_count: int = 0
    latency_ms: Optional[int] = None
    table_count: Optional[int] = None
    collection_count: Optional[int] = None
    status: str = "disconnected"  # connected, disconnected, error, slow

class ConnectionUpdateRequest(BaseModel):
    name: Optional[str] = None
    is_favorite: Optional[bool] = None
    tags: Optional[List[str]] = None
    environment: Optional[str] = None

class ConnectionHealthRequest(BaseModel):
    connection_id: str

class QueryRequest(BaseModel):
    query: str
    connection_id: Optional[str] = None  # Add connection ID for tracking
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AssistantMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

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
llamacpp_model_path = os.environ.get("LLAMACPP_MODEL_PATH")

# Startup checks for required environment variables
if not groq_api_key:
    logger.warning("GROQ_API_KEY not set. Required for non-SQL tasks.")
if not llamacpp_model_path:
    logger.warning("LLAMACPP_MODEL_PATH not set. Required for SQL generation.")

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
            detail="Query orchestrator not initialized. Please connect to a database first by going to the Connections page and clicking 'Connect' on your database."
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
        logger.info(f"Received connection request for {request.db_type} database")
        logger.info(f"Connection parameters: host={request.host}, user={request.user}, port={request.port}, database={request.database}")
        
        # Check if we already have a connection to the same database
        if (db_manager is not None and 
            db_manager.connection is not None and 
            db_manager.current_db == request.database and 
            db_manager.current_host == request.host and 
            db_manager.current_port == request.port):
            # Reuse existing connection
            logger.info("Reusing existing database connection")
            return {
                "status": "success",
                "message": "Using existing database connection",
                "schema": db_manager.get_schema(),
                "tables": db_manager.get_tables()
            }
        
        # If we have an existing connection, close it
        if db_manager is not None and db_manager.connection is not None:
            logger.info("Closing existing database connection")
            db_manager.disconnect()
        
        # Create new database manager if needed
        if db_manager is None:
            logger.info("Creating new DatabaseManager instance")
            db_manager = DatabaseManager()
        
        # Connect to the database
        logger.info("Attempting to connect to database...")
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
            logger.error(f"Database connection failed: {message}")
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
        
        logger.info(f"Database connection successful: {message}")
        
        # Initialize the query orchestrator only if it doesn't exist or if we changed databases
        if query_orchestrator is None:
            try:
                logger.info("Initializing QueryOrchestrator...")
                query_orchestrator = QueryOrchestrator(
                    db_manager=db_manager,
                    api_key=groq_api_key,
                    model_name=groq_model_name,
                )
                logger.info("QueryOrchestrator initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing QueryOrchestrator: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to initialize AI components: {str(e)}"}
                )
        
        # Return the database schema
        schema = db_manager.get_schema()
        tables = db_manager.get_tables()
        
        logger.info(f"Connection successful. Found {len(tables)} tables")
        
        response_data = {
            "status": "success",
            "message": message,
            "schema": schema,
            "tables": tables
        }
        
        logger.info("Returning successful response to frontend")
        return response_data
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect to database: {str(e)}"}
        )

@app.post("/database/connection/health")
async def check_connection_health(request: ConnectionHealthRequest):
    """
    Check the health and status of a specific database connection.
    """
    global db_manager
    try:
        if db_manager is None or db_manager.connection is None:
            return {
                "status": "disconnected",
                "latency_ms": None,
                "error": "No active database connection"
            }
        
        # Test connection with a simple query
        start_time = time.time()
        try:
            # Execute a simple health check query
            if db_manager.db_type.lower() in ['mysql', 'postgresql', 'sqlite']:
                result = db_manager.execute_query("SELECT 1")
            elif db_manager.db_type.lower() == 'mongodb':
                result = db_manager.connection.admin.command('ping')
            else:
                result = {"status": "unknown"}
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "connected",
                "latency_ms": latency_ms,
                "last_check": datetime.now().isoformat(),
                "db_type": db_manager.db_type,
                "database": db_manager.current_db
            }
        except Exception as e:
            return {
                "status": "error",
                "latency_ms": None,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error checking connection health: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to check connection health: {str(e)}"}
        )

@app.post("/database/connection/metadata")
async def update_connection_metadata(request: ConnectionMetadata):
    """
    Update metadata for a specific database connection.
    """
    try:
        # For now, we'll just log the update since we don't have Firebase integration in the backend
        # In a production environment, this would update the connection metadata in the database
        logger.info(f"Updating connection metadata for {request.connection_id}: {request.name}")
        logger.info(f"Query count: {request.query_count}, Last accessed: {request.last_accessed}")
        
        return {
            "status": "success",
            "message": "Connection metadata updated successfully",
            "connection_id": request.connection_id
        }
        
    except Exception as e:
        logger.error(f"Error updating connection metadata: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to update connection metadata: {str(e)}"}
        )

async def update_connection_query_count(connection_id: str, increment: bool = True):
    """
    Update the query count for a specific connection.
    This is a placeholder function - in production, it would update the actual connection metadata.
    """
    try:
        logger.info(f"Updating query count for connection {connection_id}, increment: {increment}")
        # In production, this would update the connection's query count in the database
        # For now, we'll just log it
        return True
    except Exception as e:
        logger.error(f"Error updating query count: {str(e)}")
        return False

@app.delete("/database/connection/{connection_id}")
async def delete_connection(connection_id: str):
    """
    Delete a connection and clean up associated data.
    """
    global db_manager
    try:
        logger.info(f"Deleting connection: {connection_id}")
        
        # If this is the currently connected database, disconnect it
        if (db_manager is not None and 
            db_manager.connection is not None and 
            hasattr(db_manager, 'connection_id') and 
            db_manager.connection_id == connection_id):
            logger.info("Disconnecting current database connection")
            db_manager.disconnect()
            db_manager = None
        
        # In a real implementation, you would also clean up Firebase data here
        # For now, we'll return success and let the frontend handle Firebase cleanup
        
        return {
            "status": "success",
            "message": "Connection deleted successfully",
            "connection_id": connection_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting connection: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete connection: {str(e)}"}
        )

@app.get("/database/connection/stats/{connection_id}")
async def get_connection_stats(connection_id: str):
    """
    Get detailed statistics for a specific connection.
    """
    global db_manager
    try:
        # Check if this is the currently connected database
        is_currently_connected = (
            db_manager is not None and 
            db_manager.connection is not None and 
            hasattr(db_manager, 'connection_id') and 
            db_manager.connection_id == connection_id
        )
        
        # Get schema information if connected
        schema_info = None
        table_count = None
        collection_count = None
        
        if is_currently_connected:
            try:
                schema = db_manager.get_schema()
                tables = db_manager.get_tables()
                
                if db_manager.db_type.lower() == 'mongodb':
                    collection_count = len(tables) if tables else 0
                else:
                    table_count = len(tables) if tables else 0
                    
                schema_info = {
                    "tables": tables,
                    "schema": schema
                }
            except Exception as e:
                logger.warning(f"Could not get schema info: {str(e)}")
        
        return {
            "connection_id": connection_id,
            "is_connected": is_currently_connected,
            "db_type": db_manager.db_type if is_currently_connected else None,
            "database": db_manager.current_db if is_currently_connected else None,
            "table_count": table_count,
            "collection_count": collection_count,
            "schema_info": schema_info,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting connection stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get connection stats: {str(e)}"}
        )

@app.post("/database/connection/test")
async def test_connection(request: ConnectionRequest):
    """
    Test a database connection without establishing a permanent connection.
    """
    try:
        logger.info(f"Testing connection for {request.db_type} database")
        
        # Create a temporary database manager for testing
        temp_db_manager = DatabaseManager()
        
        # Attempt to connect
        success, message = temp_db_manager.connect(
            db_type=request.db_type,
            host=request.host,
            user=request.user,
            password=request.password,
            port=request.port,
            database=request.database,
            db_path=request.db_path
        )
        
        if success:
            # Get basic info
            schema = temp_db_manager.get_schema()
            tables = temp_db_manager.get_tables()
            
            # Test a simple query
            start_time = time.time()
            try:
                if request.db_type.lower() in ['mysql', 'postgresql', 'sqlite']:
                    result = temp_db_manager.execute_query("SELECT 1")
                elif request.db_type.lower() == 'mongodb':
                    result = temp_db_manager.connection.admin.command('ping')
                else:
                    result = {"status": "unknown"}
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Disconnect the test connection
                temp_db_manager.disconnect()
                
                return {
                    "status": "success",
                    "message": "Connection test successful",
                    "latency_ms": latency_ms,
                    "table_count": len(tables) if tables else 0,
                    "collection_count": len(tables) if request.db_type.lower() == 'mongodb' and tables else 0,
                    "schema": schema
                }
            except Exception as e:
                temp_db_manager.disconnect()
                return {
                    "status": "partial_success",
                    "message": f"Connected but query test failed: {str(e)}",
                    "table_count": len(tables) if tables else 0,
                    "collection_count": len(tables) if request.db_type.lower() == 'mongodb' and tables else 0,
                    "schema": schema
                }
        else:
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
            
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to test connection: {str(e)}"}
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
        
        # If a connection ID is provided, update the query count
        if query_request.connection_id:
            await update_connection_query_count(query_request.connection_id, increment=True)
            logger.info(f"Updated query count for connection: {query_request.connection_id}")
        
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

@app.get("/database/suggestions")
async def get_database_suggestions(
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator)
):
    """
    Get database-specific suggestions based on the current database schema.
    These suggestions are generated dynamically based on the connected database.
    """
    try:
        # Get suggestions from the orchestrator
        suggestions = orchestrator.get_database_suggestions()
        
        # Log the suggestions for debugging
        logger.debug(f"Generated database suggestions: {suggestions}")
        
        return {
            "status": "success",
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error getting database suggestions: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Failed to get database suggestions: {str(e)}",
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
    db_connected = db_manager is not None and db_manager.connection is not None
    orchestrator_ready = query_orchestrator is not None
    
    status = {
        "status": "healthy" if db_connected and orchestrator_ready else "unhealthy",
        "db_connected": db_connected,
        "orchestrator_ready": orchestrator_ready,
        "active_queries": len(active_queries),
        "connection_info": {
            "db_type": db_manager.db_type if db_manager else None,
            "database": db_manager.current_db if db_manager else None,
            "host": db_manager.current_host if db_manager else None,
            "port": db_manager.current_port if db_manager else None
        } if db_manager else None,
        "message": "Ready to process queries" if db_connected and orchestrator_ready else "Database connection required. Please connect to a database first."
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

@app.delete("/query/history/bulk")
async def delete_multiple_query_history(query_ids: List[str]):
    """
    Delete multiple queries from backend JSON files.
    This endpoint removes the queries from both query_times.json and feedback.jsonl files.
    """
    try:
        deleted_count = 0
        errors = []
        
        for query_id in query_ids:
            try:
                deleted_from_times = False
                deleted_from_feedback = False
                
                # Delete from query_times.json
                query_times_file = "Backend/query_times.json"
                if os.path.exists(query_times_file):
                    try:
                        with open(query_times_file, "r") as f:
                            entries = json.load(f)
                        
                        # Filter out the query to delete
                        original_count = len(entries)
                        entries = [entry for entry in entries if entry.get("query_id") != query_id]
                        
                        if len(entries) < original_count:
                            with open(query_times_file, "w") as f:
                                json.dump(entries, f, indent=2)
                            deleted_from_times = True
                    except Exception as e:
                        logger.error(f"Error deleting {query_id} from query_times.json: {str(e)}")
                
                # Delete from feedback.jsonl
                feedback_file = "Backend/feedback.jsonl"
                if os.path.exists(feedback_file):
                    try:
                        with open(feedback_file, "r") as f:
                            lines = f.readlines()
                        
                        # Filter out the query to delete
                        original_count = len(lines)
                        filtered_lines = []
                        for line in lines:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("query_id") != query_id:
                                    filtered_lines.append(line)
                            except json.JSONDecodeError:
                                # Keep malformed lines
                                filtered_lines.append(line)
                        
                        if len(filtered_lines) < original_count:
                            with open(feedback_file, "w") as f:
                                f.writelines(filtered_lines)
                            deleted_from_feedback = True
                    except Exception as e:
                        logger.error(f"Error deleting {query_id} from feedback.jsonl: {str(e)}")
                
                if deleted_from_times or deleted_from_feedback:
                    deleted_count += 1
                    logger.info(f"Deleted query {query_id} from backend history")
                else:
                    errors.append(f"Query {query_id} not found in backend history files")
                    
            except Exception as e:
                error_msg = f"Error deleting {query_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            "success": True,
            "message": f"Bulk deletion completed. {deleted_count} queries deleted.",
            "deleted_count": deleted_count,
            "total_queries": len(query_ids),
            "errors": errors
        }
            
    except Exception as e:
        logger.error(f"Error in bulk deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform bulk deletion: {str(e)}")

@app.delete("/query/history/{query_id}")
async def delete_query_history(query_id: str):
    """
    Delete query history from backend JSON files.
    This endpoint removes the query from both query_times.json and feedback.jsonl files.
    """
    try:
        deleted_from_times = False
        deleted_from_feedback = False
        
        # Delete from query_times.json
        query_times_file = "Backend/query_times.json"
        if os.path.exists(query_times_file):
            try:
                with open(query_times_file, "r") as f:
                    entries = json.load(f)
                
                # Filter out the query to delete
                original_count = len(entries)
                entries = [entry for entry in entries if entry.get("query_id") != query_id]
                
                if len(entries) < original_count:
                    with open(query_times_file, "w") as f:
                        json.dump(entries, f, indent=2)
                    deleted_from_times = True
                    logger.info(f"Deleted query {query_id} from query_times.json")
            except Exception as e:
                logger.error(f"Error deleting from query_times.json: {str(e)}")
        
        # Delete from feedback.jsonl
        feedback_file = "Backend/feedback.jsonl"
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r") as f:
                    lines = f.readlines()
                
                # Filter out the query to delete
                original_count = len(lines)
                filtered_lines = []
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("query_id") != query_id:
                            filtered_lines.append(line)
                    except json.JSONDecodeError:
                        # Keep malformed lines
                        filtered_lines.append(line)
                
                if len(filtered_lines) < original_count:
                    with open(feedback_file, "w") as f:
                        f.writelines(filtered_lines)
                    deleted_from_feedback = True
                    logger.info(f"Deleted query {query_id} from feedback.jsonl")
            except Exception as e:
                logger.error(f"Error deleting from feedback.jsonl: {str(e)}")
        
        if deleted_from_times or deleted_from_feedback:
            return {
                "success": True,
                "message": f"Query {query_id} deleted from backend history",
                "deleted_from_times": deleted_from_times,
                "deleted_from_feedback": deleted_from_feedback
            }
        else:
            return {
                "success": False,
                "message": f"Query {query_id} not found in backend history files"
            }
            
    except Exception as e:
        logger.error(f"Error deleting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete query history: {str(e)}")

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

@app.post("/assistant/chat")
async def assistant_chat(request: AssistantMessage):
    """
    Chat with the AI assistant using Groq API.
    The assistant can help with:
    - Website features and usage
    - Database connection help
    - Query writing assistance
    - General platform guidance
    """
    try:
        # Initialize Groq interface
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
        groq_interface = GroqInterface(api_key=groq_key, model_name="llama3-8b-8192")
        
        # Create a comprehensive system prompt for the assistant
        system_prompt = """You are Samvad AI, a helpful and knowledgeable assistant for the SamvadDB platform. 

SamvadDB is a natural language database querying platform that allows users to:
1. Connect to various databases (MySQL, PostgreSQL, MongoDB, etc.)
2. Write queries in natural language
3. Create visualizations and dashboards
4. Manage database connections
5. View query history and performance

Your role is to:
- Help users understand how to use the platform
- Explain features and capabilities
- Guide users through database connections
- Help with query writing and optimization
- Provide tips and best practices
- Answer any questions about the platform

Be friendly, helpful, and provide practical, actionable advice. If you don't know something specific about the platform, suggest where they might find the information or offer general database best practices.

Current user message: {user_message}"""

        # Create the full prompt
        full_prompt = system_prompt.format(user_message=request.message)
        
        # Generate response using Groq
        response = groq_interface.generate(
            prompt=full_prompt,
            temperature=0.7,
            max_tokens=1024
        )
        
        logger.info(f"Assistant response generated for user: {request.user_id}")
        
        return {
            "success": True,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in assistant chat: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate assistant response: {str(e)}")

# Optional: Serve static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run the application if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)