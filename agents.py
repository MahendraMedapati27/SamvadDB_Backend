import os
import re
import json
import time
import uuid
import logging
import sqlite3
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, date
import traceback
import functools
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import requests
# import google.generativeai as genai  # Removed - no longer using Gemini
from llama_cpp import Llama
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine, Connection, ResultProxy
from sqlalchemy.sql import text as sql_text
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexFlatIP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
try:
    from groq import Groq, AuthenticationError
except ImportError:
    Groq = None
    # Fall back to base Exception type for environments without groq installed
    class AuthenticationError(Exception):
        pass

# Forward declaration for type hints
class ContextTrackingAgent:
    pass

# Custom JSON encoder to handle date objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Decimal):
            return float(obj)  # Convert Decimal to float for JSON serialization
        return super().default(obj)

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('agents_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sql-ai-agents")

def debug_logger(func):
    """Decorator for detailed function logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name} with args: {args}, kwargs: {kwargs}")
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"Exiting {func_name} with result: {result}")
            logger.debug(f"{func_name} execution time: {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

# ===============================================================
# MODEL INTERFACES
# ===============================================================

class ModelInterface:
    """Base class for all model interfaces."""
    
    def __init__(self):
        logger.debug("Initializing ModelInterface")
        self.model_loaded = False
    
    @debug_logger
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model_loaded
    
    @debug_logger
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    @debug_logger
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        raise NotImplementedError("Subclasses must implement this method.")

class HuggingFaceInterface(ModelInterface):
    """Interface for locally loaded HuggingFace models."""
    
    @debug_logger
    def __init__(self, model_name_or_path: str, device: str = "cuda", max_context_length: int = 4096):
        """
        Initialize the HuggingFace model interface.
        
        Args:
            model_name_or_path (str): Path to the model or model name from HuggingFace Hub
            device (str): Device to run the model on ('cuda' or 'cpu')
            max_context_length (int): Maximum context length for the model
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing HuggingFaceInterface with model: {model_name_or_path}")
        
        try:
            # Set device
            self.torch_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
            self.logger.debug(f"Using device: {self.torch_device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Set generation config
            self.model.generation_config.do_sample = True  # Enable sampling
            self.model.generation_config.temperature = 0.1  # Set temperature for sampling
            self.model.generation_config.max_new_tokens = 512
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            
            self.max_context_length = max_context_length
            self.logger.info("HuggingFace model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing HuggingFace model: {str(e)}")
            raise
    
    def _chunk_prompt(self, prompt: str, chunk_overlap: int = 100) -> List[str]:
        """Split a long prompt into chunks that fit within the model's context length.
        
        Args:
            prompt (str): The prompt to chunk.
            chunk_overlap (int): The number of tokens to overlap between chunks.
            
        Returns:
            List[str]: A list of prompt chunks.
        """
        # Tokenize the prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Calculate the maximum tokens per chunk (leaving room for generation)
        max_tokens_per_chunk = self.max_context_length - 512  # Reserve 512 tokens for generation
        
        # If the prompt fits within the context window, return it as is
        if len(tokens) <= max_tokens_per_chunk:
            return [prompt]
        
        # Split into chunks
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + max_tokens_per_chunk, len(tokens))
            
            # Get chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to the next chunk, with overlap
            start_idx = end_idx - chunk_overlap
        
        logger.debug(f"Split prompt into {len(chunks)} chunks")
        return chunks
    
    @debug_logger
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the HuggingFace model.
        
        Args:
            prompt (str): The prompt to generate a response for.
            **kwargs: Additional arguments to pass to the model.
        
        Returns:
            str: The generated response.
        """
        try:
            import torch
            
            logger.debug(f"Generating response with prompt length: {len(prompt)}")
            logger.debug(f"Generation parameters: {kwargs}")
            
            # Set default parameters
            max_new_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.95)
            top_k = kwargs.get("top_k", 40)
            
            # Check if prompt needs to be chunked
            prompt_chunks = self._chunk_prompt(prompt)
            
            # If we have multiple chunks, process them and combine results
            if len(prompt_chunks) > 1:
                logger.debug(f"Processing {len(prompt_chunks)} prompt chunks")
                all_responses = []
                
                for i, chunk in enumerate(prompt_chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(prompt_chunks)}")
                    
                    # Add specific instructions for chunked processing
                    if i == 0:
                        chunk += "\n\n[Note: This is the first part of a longer prompt. Please analyze it but wait for the complete context before final output.]"
                    elif i < len(prompt_chunks) - 1:
                        chunk += f"\n\n[Note: This is part {i+1} of a longer prompt. Please continue analyzing but wait for the complete context.]"
                    else:
                        chunk += "\n\n[Note: This is the final part of the prompt. Please provide a complete response based on all context.]"
                    
                    # Generate response for this chunk
                    inputs = self.tokenizer(chunk, return_tensors="pt").to(self.torch_device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,  # Use smaller token count for intermediate chunks
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    # Decode the response
                    response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    all_responses.append(response_text)
                
                # For the final response, use the entire context and generate a coherent answer
                final_prompt = f"""
                I've analyzed a large prompt in chunks. Here are my intermediate analyses:
                
                {' '.join(all_responses)}
                
                Now, based on all the information above, please generate a final, coherent response to the original request:
                
                {prompt_chunks[-1]}
                """
                
                inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.torch_device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode the final response
                final_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                logger.debug("Response generated successfully from chunked prompt")
                return final_response
            
            # Process single prompt (fits within context window)
            else:
                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.torch_device)
                
                # Generate the response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode the response
                response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                logger.debug("Response generated successfully")
                return response_text
            
        except Exception as e:
            logger.error(f"Error generating response from HuggingFace model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @debug_logger
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text.
        
        Args:
            text (str): The text to generate an embedding for.
        
        Returns:
            List[float]: The embedding vector.
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # For embeddings, we'll use the model's hidden states
            # This is a simple approach - for better embeddings, consider using a dedicated embedding model
            
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.torch_device)
            
            # Generate embeddings from the model's hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            # Convert to list and return
            return embeddings[0].cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

# class GeminiInterface(ModelInterface):
#     """Interface for the Gemini API."""
#     
#     @debug_logger
#     def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
#         """Initialize the Gemini interface.
#         
#         Args:
#             api_key (str): The API key for Gemini.
#             model_name (str): The model name to use.
#         """
#         super().__init__()
#         logger.debug(f"Initializing GeminiInterface with model: {model_name}")
#         self.api_key = api_key
#         self.model_name = model_name
#         
#         try:
#             # Initialize the Gemini API
#             genai.configure(api_key=api_key)
#             logger.debug("Gemini API configured successfully")
#             
#             # Set up the model
#             self.model = genai.GenerativeModel(model_name)
#             self.model_loaded = True
#             logger.info(f"Gemini model {model_name} initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize Gemini model: {str(e)}")
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             raise
#     
#     @debug_logger
#     def generate(self, prompt: str, **kwargs) -> str:
#         """Generate a response using the Gemini API."""
#         try:
#             logger.debug(f"Generating response with prompt length: {len(prompt)}")
#             logger.debug(f"Generation parameters: {kwargs}")
#             
#             generation_config = {
#                 "temperature": kwargs.get("temperature", 0.7),
#                 "top_p": kwargs.get("top_p", 0.95),
#                 "top_k": kwargs.get("top_k", 40),
#                 "max_output_tokens": kwargs.get("max_tokens", 1024),
#             }
#             logger.debug(f"Using generation config: {generation_config}")
#             
#             response = self.model.generate_content(
#                 prompt,
#                 generation_config=generation_config
#             )
#             logger.debug("Response generated successfully")
#             
#             response_text = response.text
#             logger.debug(f"Response text length: {len(response_text)}")
#             return response_text
#             
#         except Exception as e:
#             logger.error(f"Error generating response from Gemini: {str(e)}")
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             raise


class LlamaCppInterface(ModelInterface):
    """Interface for the llama.cpp library, used primarily for SQL generation."""
    
    @debug_logger
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
        """Initialize the llama.cpp interface.
        
        Args:
            model_path (str): Path to the model file.
            n_gpu_layers (int): Number of GPU layers to use. -1 means all.
            n_ctx (int): Context size.
        """
        super().__init__()
        logger.debug(f"Initializing LlamaCppInterface with model path: {model_path}")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        try:
            # Load the model
            logger.debug("Loading LlamaCpp model...")
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            self.model_loaded = True
            logger.info(f"LlamaCpp model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading LlamaCpp model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @debug_logger
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the llama.cpp model.
        
        Args:
            prompt (str): The prompt to generate a response for.
            **kwargs: Additional arguments to pass to the model.
        
        Returns:
            str: The generated response.
        """
        try:
            logger.debug(f"Generating response with prompt length: {len(prompt)}")
            logger.debug(f"Generation parameters: {kwargs}")
            
            # Default parameters
            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.95)
            top_k = kwargs.get("top_k", 40)
            repeat_penalty = kwargs.get("repeat_penalty", 1.1)
            
            logger.debug(f"Using generation parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, repeat_penalty={repeat_penalty}")
            
            # Generate the response
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                echo=False
            )
            logger.debug("Response generated successfully")
            
            # Extract and return the response text
            response_text = output["choices"][0]["text"].strip()
            logger.debug(f"Response text length: {len(response_text)}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response from LlamaCpp: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @debug_logger
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text.
        
        Args:
            text (str): The text to generate an embedding for.
        
        Returns:
            List[float]: The embedding vector.
        """
        try:
            # Generate the embedding
            embedding = self.llm.embed(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

class GroqInterface(ModelInterface):
    """Interface for the Groq API."""
    @debug_logger
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192"):
        """Initialize the Groq interface.
        Args:
            api_key (str): The API key for Groq. If None, uses GROQ_API_KEY env var.
            model_name (str): The model name to use (e.g., 'llama3-8b-8192').
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        if Groq is None:
            raise ImportError("groq package is not installed. Please install it with 'pip install groq'.")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Provide a valid key via env or parameter.")
        self.client = Groq(api_key=self.api_key)
        # Validate API key with a minimal call to fail fast on invalid keys
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model=model_name,
                temperature=0.0,
                max_tokens=1,
            )
        except AuthenticationError as e:
            logger.error(f"Groq authentication failed during initialization: {e}")
            raise
        except Exception:
            # Ignore other transient errors; actual generation will handle them
            pass
        self.model_loaded = True
        logger.info(f"GroqInterface initialized with model: {self.model_name}")

    @debug_logger
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the Groq API.
        Args:
            prompt (str): The prompt to generate a response for.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
        Returns:
            str: The generated response.
        """
        try:
            logger.debug(f"Generating response with prompt length: {len(prompt)}")
            logger.debug(f"Generation parameters: {kwargs}")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            logger.debug(f"Groq response: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating response from Groq: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

class EmbeddingModel:
    """Interface for sentence transformers embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model.
        
        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding for the given text.
        
        Args:
            text (str): The text to get the embedding for.
        
        Returns:
            np.ndarray: The embedding vector.
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise


# ===============================================================
# DATABASE MANAGEMENT
# ===============================================================

class DatabaseManager:
    """Manages database connections and executes queries."""
    
    @debug_logger
    def __init__(self):
        """Initialize the database manager."""
        logger.debug("Initializing DatabaseManager")
        self.engine = None
        self.connection = None
        self.metadata = None
        self.db_type = None
        self.current_db = None
        self.current_host = None
        self.current_port = None
    
    @debug_logger
    def connect(self, db_type: str, host: Optional[str] = None, user: Optional[str] = None, 
               password: Optional[str] = None, port: Optional[int] = None, 
               database: Optional[str] = None, db_path: Optional[str] = None) -> Tuple[bool, str]:
        """Connect to a database.
        
        Args:
            db_type (str): The type of database (sqlite, mysql, postgresql).
            host (str, optional): The database host.
            user (str, optional): The database username.
            password (str, optional): The database password.
            port (int, optional): The database port.
            database (str, optional): The database name.
            db_path (str, optional): The path to the SQLite database file.
        
        Returns:
            Tuple[bool, str]: A tuple containing a success flag and a message.
        """
        try:
            logger.debug(f"Attempting to connect to {db_type} database")
            logger.debug(f"Connection parameters: host={host}, user={user}, port={port}, database={database}, db_path={db_path}")
            
            # Disconnect from any existing connection
            self.disconnect()
            
            # Create the connection URL based on the database type
            if db_type.lower() == "sqlite":
                if not db_path:
                    logger.error("SQLite database path is required")
                    return False, "SQLite database path is required"
                
                # Create the SQLite connection URL
                connection_url = f"sqlite:///{db_path}"
                logger.debug(f"SQLite connection URL: {connection_url}")
            
            elif db_type.lower() == "mysql":
                if not all([host, user, password, database]):
                    logger.error("MySQL connection requires host, user, password, and database")
                    return False, "MySQL connection requires host, user, password, and database"
                
                # Set default port if not provided
                if not port:
                    port = 3306
                
                # Create the MySQL connection URL
                connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                logger.debug(f"MySQL connection URL created (password hidden)")
            
            elif db_type.lower() == "postgresql":
                if not all([host, user, password, database]):
                    logger.error("PostgreSQL connection requires host, user, password, and database")
                    return False, "PostgreSQL connection requires host, user, password, and database"
                
                # Set default port if not provided
                if not port:
                    port = 5432
                
                # Create the PostgreSQL connection URL
                connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                logger.debug(f"PostgreSQL connection URL created (password hidden)")
            
            else:
                logger.error(f"Unsupported database type: {db_type}")
                return False, f"Unsupported database type: {db_type}"
            
            # Create the SQLAlchemy engine
            logger.debug("Creating SQLAlchemy engine...")
            self.engine = create_engine(connection_url)
            
            # Create a connection
            logger.debug("Establishing database connection...")
            self.connection = self.engine.connect()
            
            # Store the database type
            self.db_type = db_type.lower()
            
            # Reflect the database schema
            logger.debug("Reflecting database schema...")
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
            # After successful connection, store connection info
            self.current_db = database
            self.current_host = host
            self.current_port = port
            
            logger.info(f"Successfully connected to {db_type} database")
            return True, f"Successfully connected to {db_type} database"
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Failed to connect to database: {str(e)}"
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.connection:
            self.connection.close()
            self.connection = None
        
        if self.engine:
            self.engine.dispose()
            self.engine = None
        
        self.metadata = None
        self.db_type = None
        self.current_db = None
        self.current_host = None
        self.current_port = None
    
    def get_tables(self) -> List[str]:
        """Get a list of tables in the database.
        
        Returns:
            List[str]: A list of table names.
        """
        if not self.metadata:
            return []
        
        return list(self.metadata.tables.keys())
    
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Get the database schema.
        
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with table names as keys and
                lists of column dictionaries as values.
        """
        if not self.metadata:
            return {}
        
        schema = {}
        
        for table_name, table in self.metadata.tables.items():
            columns = []
            for column in table.columns:
                columns.append({
                    "name": column.name,
                    "type": str(column.type),
                    "nullable": column.nullable,
                    "primary_key": column.primary_key
                })
            
            schema[table_name] = columns
        
        return schema
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of rows from a table.
        
        Args:
            table_name (str): The name of the table.
            limit (int, optional): The maximum number of rows to return. Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries with row data.
        """
        if not self.connection or table_name not in self.metadata.tables:
            return []
        
        try:
            # Create the query
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            
            # Execute the query
            result = self.connection.execute(sql_text(query))
            
            # Get column names
            columns = result.keys()
            
            # Convert the result to a list of dictionaries
            rows = []
            for row in result:
                # Create a dictionary with column names as keys and row values
                row_dict = {col: val for col, val in zip(columns, row)}
                rows.append(row_dict)
            
            return rows
            
        except Exception as e:
            logger.error(f"Error getting table sample: {str(e)}")
            return []
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute a SQL query.
        
        Args:
            query (str): The SQL query to execute.
        
        Returns:
            Tuple[List[Dict[str, Any]], Optional[str]]: A tuple containing the results and an error message.
        """
        if not self.connection:
            return [], "Database connection not established"
        
        try:
            # Execute the query
            result = self.connection.execute(sql_text(query))
            
            # Get column names
            columns = result.keys()
            
            # Convert the result to a list of dictionaries
            rows = []
            for row in result:
                # Create a dictionary with column names as keys and row values
                row_dict = {col: val for col, val in zip(columns, row)}
                rows.append(row_dict)
            
            return rows, None
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return [], str(e)
    
    def get_schema_description(self) -> str:
        """Get a textual description of the database schema.
        
        Returns:
            str: A description of the database schema.
        """
        if not self.metadata:
            return "No database schema available"
        
        schema_desc = []
        
        for table_name, table in self.metadata.tables.items():
            schema_desc.append(f"Table: {table_name}")
            
            for column in table.columns:
                pk_indicator = "PRIMARY KEY" if column.primary_key else ""
                nullable = "NULL" if column.nullable else "NOT NULL"
                schema_desc.append(f"  - {column.name} ({column.type}) {nullable} {pk_indicator}")
            
            schema_desc.append("")
        
        return "\n".join(schema_desc)
    
    def get_table_relationships(self) -> Dict[str, List[Dict[str, str]]]:
        """Get the relationships between tables.
        
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with table names as keys and
                lists of relationship dictionaries as values.
        """
        if not self.metadata:
            return {}
        
        relationships = {}
        
        for table_name, table in self.metadata.tables.items():
            table_relationships = []
            
            for fk in table.foreign_keys:
                target_table = fk.column.table.name
                target_column = fk.column.name
                source_column = fk.parent.name
                
                table_relationships.append({
                    "source_column": source_column,
                    "target_table": target_table,
                    "target_column": target_column
                })
            
            if table_relationships:
                relationships[table_name] = table_relationships
        
        return relationships


# ===============================================================
# AGENT IMPLEMENTATIONS
# ===============================================================

class GuardAgent:
    """
    Guard Agent: Classifies whether the user input is relevant to the database.
    Uses RAG + embeddings to compare user input with schema.
    """
    
    def __init__(self, model_interface: ModelInterface, db_manager: DatabaseManager):
        """Initialize the guard agent.
        
        Args:
            model_interface (ModelInterface): The model interface to use.
            db_manager (DatabaseManager): The database manager.
        """
        self.model = model_interface
        self.db_manager = db_manager
        self.embedding_model = EmbeddingModel()
        self.schema_embeddings = None
        self.schema_texts = None
        self.index = None
        
        # Initialize schema embeddings
        self._initialize_schema_embeddings()
    
    def _initialize_schema_embeddings(self) -> None:
        """Initialize schema embeddings for similarity comparison."""
        if self.db_manager.metadata is None:
            logger.warning("Database schema not available. Schema embeddings not initialized.")
            return
        
        # Create textual descriptions of each table
        schema_texts = []
        
        for table_name, table in self.db_manager.metadata.tables.items():
            # Create a description of the table
            table_desc = f"Table {table_name} with columns: "
            column_desc = ", ".join([f"{col.name} ({col.type})" for col in table.columns])
            schema_texts.append(table_desc + column_desc)
            
            # Add descriptions for individual columns
            for column in table.columns:
                schema_texts.append(f"Column {column.name} in table {table_name} with type {column.type}")
        
        # Add relationships if available
        relationships = self.db_manager.get_table_relationships()
        for table_name, table_relationships in relationships.items():
            for rel in table_relationships:
                rel_desc = (f"Table {table_name} has foreign key {rel['source_column']} "
                           f"referring to {rel['target_table']}.{rel['target_column']}")
                schema_texts.append(rel_desc)
        
        # Generate embeddings for schema texts
        if schema_texts:
            embeddings = [self.embedding_model.get_embedding(text) for text in schema_texts]
            self.schema_embeddings = np.array(embeddings)
            self.schema_texts = schema_texts
            
            # Create a FAISS index for cosine similarity
            dimension = self.schema_embeddings.shape[1]
            self.index = IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.schema_embeddings)
            self.index.add(self.schema_embeddings)
            
            logger.info(f"Initialized schema embeddings with {len(schema_texts)} entries")
    
    def _use_model_for_relevance(self, query: str) -> Tuple[str, str, Optional[List[str]]]:
        """Use the LLM to determine if a query is relevant to the database.
        
        Args:
            query (str): The user query.
            
        Returns:
            Tuple[str, str, Optional[List[str]]]: A tuple with:
                - classification: "RELEVANT", "NOT_RELEVANT", or "NEED_MORE_INFO"
                - reason: The reason for the classification
                - suggestions: List of suggestions if NEED_MORE_INFO
        """
        # Get schema information
        schema_desc = self.db_manager.get_schema_description()
        tables = self.db_manager.get_tables()
        
        # Craft a prompt using various prompt engineering techniques
        prompt = f"""You are a database query analyzer. Your task is to carefully analyze if a user's query can be answered using the database schema.

DATABASE SCHEMA:
{schema_desc}

AVAILABLE TABLES:
{', '.join(tables)}

Let's analyze the query step by step:

1. First, check if the query is valid English and makes sense:
   - Is it gibberish or random characters?
   - Is it a complete sentence or phrase?
   - Does it contain any database-related terms?

2. Then, check if it's related to the database:
   - Does it mention any tables from the schema?
   - Does it ask for information that could be in the database?
   - Is it a question about the data?

3. Finally, check if it needs more information:
   - Is it too vague or incomplete?
   - Does it need specific details about tables or columns?
   - Would you need to ask the user for clarification?

Here are some examples:

Query: "show me the"
Analysis: This is incomplete and needs more information
Classification: NEED_MORE_INFO
Reason: The query is incomplete and doesn't specify what information to show
Suggestions: 
- "Show me the orders from last month"
- "Show me the total sales by product"
- "Show me the customer details for order #123"

Query: "asdfghjkl"
Analysis: This is random characters with no meaning
Classification: NOT_RELEVANT
Reason: The query is gibberish and not a valid question

Query: "what is the meaning of life or embedding systems"
Analysis: This is a question about the meaning of life and embedding systems
Classification: NOT_RELEVANT
Reason: The query is not related to database 

Query: "Count the number of orders for each month in 2024"
Analysis: This is a clear, complete query about orders and dates
Classification: RELEVANT
Reason: The query is clear and can be answered using the orders table

Now, analyze this query:

USER QUERY:
{query}

Think through your analysis step by step:
1. Is it valid English?
2. Is it related to the database?
3. Does it need more information?

IMPORTANT: Your response MUST start with exactly one of these three lines:
CLASSIFICATION: RELEVANT
CLASSIFICATION: NOT_RELEVANT
CLASSIFICATION: NEED_MORE_INFO

Then provide:
REASON: <your detailed reasoning>
SUGGESTIONS: <list of specific suggestions if NEED_MORE_INFO>
"""
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.1)
        
        # Parse the response
        classification = "NOT_RELEVANT"
        reason = "Query is not related to the database"
        suggestions = None
        
        try:
            # First, try to find the classification in the first few lines
            first_lines = response.split('\n')[:5]
            for line in first_lines:
                line = line.strip().upper()
                if "RELEVANT" in line:
                    if "NOT" in line:
                        classification = "NOT_RELEVANT"
                    else:
                        classification = "RELEVANT"
                    break
                elif "NEED_MORE_INFO" in line:
                    classification = "NEED_MORE_INFO"
                    break
            
            # If not found in first lines, try regex patterns
            if classification == "NOT_RELEVANT":
                classification_patterns = [
                    r'(?:CLASSIFICATION|CLASSIFIACTION|CLASSIFICAATION)\s*:\s*(RELEVANT|NOT_RELEVANT|NEED_MORE_INFO)',
                    r'(?:CLASSIFICATION|CLASSIFIACTION|CLASSIFICAATION)\s*:\s*(\w+)',
                    r'CLASSIFICATION\s*:\s*(\w+)',
                    r'CLASSIFIACTION\s*:\s*(\w+)',
                    r'CLASSIFICAATION\s*:\s*(\w+)'
                ]
                
                for pattern in classification_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        found_class = match.group(1).strip().upper()
                        if found_class in ["RELEVANT", "NOT_RELEVANT", "NEED_MORE_INFO"]:
                            classification = found_class
                            break
            
            # Extract reason
            reason_match = re.search(r'REASON:\s*(.*?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            
            # Extract suggestions if NEED_MORE_INFO
            if classification == "NEED_MORE_INFO":
                suggestions_match = re.search(r'SUGGESTIONS:\s*(.*?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
                if suggestions_match:
                    suggestions_text = suggestions_match.group(1).strip()
                    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
            
            # Log the full response for debugging
            logger.debug(f"Full model response:\n{response}")
            logger.debug(f"Parsed classification: {classification}, reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            logger.error(f"Response text: {response}")
        
        return classification, reason, suggestions
    
    def is_relevant(self, query: str, threshold: float = 0.65) -> Tuple[str, str, Optional[List[str]]]:
        """Check if the query is relevant to the database.
        
        Args:
            query (str): The user query.
            threshold (float, optional): The similarity threshold. Defaults to 0.65.
            
        Returns:
            Tuple[str, str, Optional[List[str]]]: A tuple with:
                - classification: "RELEVANT", "NOT_RELEVANT", or "NEED_MORE_INFO"
                - reason: The reason for the classification
                - suggestions: List of suggestions if NEED_MORE_INFO
        """
        # If no schema embeddings are available, default to using the model
        if self.schema_embeddings is None or len(self.schema_embeddings) == 0:
            return self._use_model_for_relevance(query)
        
        # Get the embedding for the query
        query_embedding = self.embedding_model.get_embedding(query)
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Reshape for FAISS
        query_embedding_reshaped = np.array([query_embedding])
        
        # Find the closest schema element using cosine similarity
        similarities, indices = self.index.search(query_embedding_reshaped, 3)
        
        # Get the maximum similarity
        max_similarity = similarities[0].max()
        
        # If similarity is high enough, consider it relevant
        if max_similarity >= threshold:
            best_match_idx = indices[0][similarities[0].argmax()]
            reason = f"Query is relevant to schema: {self.schema_texts[best_match_idx]}"
            return "RELEVANT", reason, None
        
        # If similarity is low, use the model as a backup
        return self._use_model_for_relevance(query)


class IntentClassificationAgent:
    """
    Intent Classification Agent: Identifies the user's intent.
    (e.g., question, aggregation, update, clarification, explanation)
    """
    
    def __init__(self, model_interface: ModelInterface, context_agent: Optional[ContextTrackingAgent] = None):
        """Initialize the intent classification agent.
        
        Args:
            model_interface (ModelInterface): The model interface to use.
            context_agent (ContextTrackingAgent, optional): The context tracking agent to use for conversation history.
        """
        self.model = model_interface
        self.context_agent = context_agent
    
    def classify(self, query: str) -> Dict[str, Any]:
        """Classify the user's intent.
        
        Args:
            query (str): The user query.
        
        Returns:
            Dict[str, Any]: A dictionary with the intent classification.
        """
        # Get context-enhanced query if context agent is available
        enhanced_query = query
        context_info = ""
        if self.context_agent and self.context_agent.history:
            enhanced_query = self.context_agent.get_context_enhanced_query(query)
            # Get the last 3 interactions for context
            recent_history = self.context_agent.history[-3:]
            context_info = "\n\nPrevious Context:\n" + "\n".join([
                f"User: {item['query']}\nSystem: {item['response'].get('explanation', '')}"
                for item in recent_history
            ])
        
        # Craft a prompt for the model using various prompting techniques
        prompt = f"""You are an expert SQL query analyzer with deep understanding of database operations and user intentions. Your role is to carefully analyze user queries and classify their intent with high accuracy.

Let's analyze this query step by step:

1. First, understand the context:
   - Is this a new query or asking about previous results?
   - Does it reference any previous operations?
   - Is it asking for clarification or explanation?
   {context_info}

2. Then, identify the operation type:
   - Is it retrieving data (SELECT)?
   - Is it modifying data (INSERT/UPDATE/DELETE)?
   - Is it aggregating data (GROUP BY, COUNT, SUM, etc.)?
   - Is it comparing data (JOIN, WHERE conditions)?
   - Is it asking for explanation of previous results?

3. Finally, determine the complexity:
   - Does it need multiple tables?
   - Does it require complex calculations?
   - Does it need temporal analysis?

Here are some examples:

Query: "Show me the total sales by product"
Analysis: This is a data aggregation query
Intent: Aggregation
Confidence: 0.95
Operations: ["SELECT", "GROUP BY", "SUM"]
Requires Write: false

Query: "Add a new customer with name John Smith"
Analysis: This is a data modification query
Intent: Update
Confidence: 0.90
Operations: ["INSERT"]
Requires Write: true

Query: "Compare sales between regions"
Analysis: This is a data comparison query
Intent: Comparison
Confidence: 0.85
Operations: ["SELECT", "GROUP BY", "JOIN"]
Requires Write: false

Query: "Can you explain the results from my last query?"
Analysis: This is asking for explanation of previous results
Intent: Explanation
Confidence: 0.95
Operations: ["SELECT"]
Requires Write: false

Query: "What tables are available?"
Analysis: This is asking about database structure
Intent: Schema Request
Confidence: 0.90
Operations: []
Requires Write: false

Now, analyze this query:

USER QUERY:
{enhanced_query}

Think through your analysis step by step:
1. What is the user trying to achieve?
2. What type of operation is needed?
3. How complex is the query?
4. Does it need write access?
5. How does the previous context affect the intent?

Return your analysis as a JSON object with these fields:
- intent: The category name (Information Retrieval, Aggregation, Update, Comparison, Explanation, Schema Request)
- confidence: A number between 0 and 1 indicating confidence
- requires_write_access: Boolean indicating if the query requires write access
- operations: List of potential SQL operations (SELECT, INSERT, UPDATE, DELETE, etc.)
- reasoning: A brief explanation of your classification

Format your response as valid JSON only.
"""
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.1)
        
        # Parse the JSON response
        try:
            # Find and extract the JSON object
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                intent_data = json.loads(json_str)
            else:
                # If no JSON found, create a default response
                intent_data = {
                    "intent": "Information Retrieval",
                    "confidence": 0.7,
                    "requires_write_access": False,
                    "operations": ["SELECT"],
                    "reasoning": "Default classification based on query analysis"
                }
                
            return intent_data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default response
            logger.error(f"Failed to parse intent classification response: {response}")
            return {
                "intent": "Information Retrieval",
                "confidence": 0.5,
                "requires_write_access": False,
                "operations": ["SELECT"],
                "reasoning": "Error parsing classification response"
            }


class ContextTrackingAgent:
    """
    Context Tracking Agent: Maintains conversation context and determines if a new input
    is related to a previous one.
    """
    
    def __init__(self, model_interface: ModelInterface, max_history: int = 5):
        """Initialize the context tracking agent.
        
        Args:
            model_interface (ModelInterface): The model interface to use.
            max_history (int, optional): Maximum number of conversation turns to keep.
        """
        self.model = model_interface
        self.max_history = max_history
        self.history = []
    
    def add_interaction(self, query: str, response: Dict[str, Any]) -> None:
        """Add an interaction to the conversation history.
        
        Args:
            query (str): The user query.
            response (Dict[str, Any]): The system response.
        """
        # Add to history
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if it exceeds the maximum length
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_enhanced_query(self, query: str) -> str:
        """Enhance the query with context from the conversation history.
        
        Args:
            query (str): The current user query.
        
        Returns:
            str: The context-enhanced query.
        """
        # If no history, just return the original query
        if not self.history:
            return query
        
        # Create a prompt for the model
        history_text = "\n".join([
            f"User: {item['query']}\nSystem: {item['response'].get('explanation', '')}"
            for item in self.history[-3:]  # Just use the last 3 interactions
        ])
        
        prompt = f"""
        You are an AI assistant that enhances user queries by considering conversation context.
        
        CONVERSATION HISTORY:
        {history_text}
        
        CURRENT QUERY:
        {query}
        
        Based on the conversation history, enhance the current query by:
        1. Resolving ambiguous references (e.g., "it", "that", "those")
        2. Adding relevant context from previous queries
        3. Incorporating information from previous responses if needed
        
        Return the enhanced query that can stand alone without requiring the conversation history.
        Only include the enhanced query text in your response, with no explanations or markdown.
        If the query is already complete and doesn't need enhancement, return it unchanged.
        """
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.1)
        
        # Clean up the response to get just the enhanced query
        enhanced_query = response.strip()
        
        # If the response is too long or appears to be an explanation rather than a query,
        # just return the original query
        if len(enhanced_query) > len(query) * 3 or enhanced_query.startswith("I "):
            return query
            
        return enhanced_query
    
    def is_related_to_previous(self, query: str) -> Tuple[bool, Optional[int]]:
        """Check if the query is related to a previous interaction.
        
        Args:
            query (str): The user query.
        
        Returns:
            Tuple[bool, Optional[int]]: A tuple with a boolean indicating relatedness and
                the index of the related interaction.
        """
        # If no history, it's not related
        if not self.history:
            return False, None
        
        # Create a prompt for the model
        history_text = "\n".join([
            f"Query {i+1}: {item['query']}"
            for i, item in enumerate(self.history)
        ])
        
        prompt = f"""
        You are an AI assistant that determines if a new query is related to previous queries.
        
        PREVIOUS QUERIES:
        {history_text}
        
        NEW QUERY:
        {query}
        
        Is this new query related to any of the previous queries?
        If yes, specify the query number it's most related to.
        If no, state that it's not related.
        
        Return your answer as:
        "RELATED: <query_number>" or "NOT_RELATED"
        """
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.1)
        
        # Parse the response
        if "RELATED:" in response.upper():
            try:
                query_number = int(response.split("RELATED:", 1)[1].strip().split()[0])
                # Adjust to 0-based index
                return True, query_number - 1
            except (ValueError, IndexError):
                return True, len(self.history) - 1
        else:
            return False, None
        
class VisualizationAgent:
    """
    Visualization Agent: Analyzes table output and suggests visualizations.
    """
    
    def __init__(self, model_interface: ModelInterface):
        """Initialize the visualization agent.
        
        Args:
            model_interface (ModelInterface): The model interface to use.
        """
        self.model = model_interface
    
    def suggest_visualizations(self, data: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
        """Suggest visualizations for the data.
        
        Args:
            data (List[Dict[str, Any]]): The data to visualize.
            columns (List[str]): The columns in the data.
        
        Returns:
            Dict[str, Any]: A dictionary with visualization suggestions.
        """
        # If no data or columns, return empty suggestions
        if not data or not columns:
            return {
                "suitable_visualizations": [],
                "recommended_chart": None,
                "x_axis_candidates": [],
                "y_axis_candidates": []
            }
        
        # Prepare a sample of the data for the model
        sample_data = data[:5]
        sample_json = json.dumps(sample_data, default=str)
        
        # Identify data types
        column_types = {}
        for col in columns:
            # Skip if the column doesn't exist in the data
            if not data or col not in data[0]:
                continue
                
            # Check the type of the column values
            values = [row[col] for row in data if col in row and row[col] is not None]
            if not values:
                column_types[col] = "unknown"
                continue
                
            # Check if all values are numeric
            if all(isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '', 1).isdigit()) for val in values):
                column_types[col] = "numeric"
            # Check if all values are dates
            elif all(isinstance(val, str) and bool(re.match(r'\d{4}-\d{2}-\d{2}', val)) for val in values):
                column_types[col] = "date"
            # Otherwise, consider it categorical
            else:
                column_types[col] = "categorical"
        
        # Craft a prompt for the model
        prompt = f"""
        You are an AI assistant that suggests visualizations for data.
        
        DATA SAMPLE:
        {sample_json}
        
        COLUMNS AND THEIR TYPES:
        {json.dumps(column_types, indent=2)}
        
        Based on the data, suggest appropriate visualizations. Return your suggestions as a JSON object with these fields:
        - suitable_visualizations: List of suitable chart types (bar, line, pie, scatter, etc.)
        - recommended_chart: The most suitable chart type
        - x_axis_candidates: List of columns suitable for the x-axis
        - y_axis_candidates: List of columns suitable for the y-axis
        - explanation: Brief explanation of why these are suitable
        
        Format your response as valid JSON only.
        """
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.2)
        
        try:
            # Find and extract the JSON object
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                visualization_data = json.loads(json_str)
            else:
                # If no JSON found, create a default response
                visualization_data = {
                    "suitable_visualizations": ["bar", "table"],
                    "recommended_chart": "bar",
                    "x_axis_candidates": [col for col, type_ in column_types.items() if type_ in ["categorical", "date"]],
                    "y_axis_candidates": [col for col, type_ in column_types.items() if type_ == "numeric"],
                    "explanation": "Default visualization based on column types"
                }
            
            return visualization_data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default response
            logger.error(f"Failed to parse visualization suggestion response: {response}")
            return {
                "suitable_visualizations": ["bar", "table"],
                "recommended_chart": "bar",
                "x_axis_candidates": [col for col, type_ in column_types.items() if type_ in ["categorical", "date"]],
                "y_axis_candidates": [col for col, type_ in column_types.items() if type_ == "numeric"],
                "explanation": "Default visualization based on column types"
            }
    
    def is_visualizable(self, results):
        # Simple check: results must be a list of dicts with at least 2 columns
        if not results or not isinstance(results, list):
            return False
        if not isinstance(results[0], dict):
            return False
        return len(results[0].keys()) >= 2
    
    def generate_summary(self, results):
        """
        Generate a simple summary of the results for visualization.
        This is a placeholder. You can make it more advanced as needed.
        """
        if not results or not isinstance(results, list) or not results[0]:
            return []
        # Example: summarize by showing the first row as a sample
        summary = []
        first_row = results[0]
        for key, value in first_row.items():
            summary.append(f"{key}: {value}")
        return summary


class DynamicSuggestionAgent:
    """
    Generates dynamic suggestions based on user input and context.
    Uses advanced prompting techniques for better suggestions.
    """
    
    def __init__(self, model_interface: ModelInterface, db_manager: DatabaseManager):
        """Initialize the suggestion agent.
        
        Args:
            model_interface (ModelInterface): The model interface.
            db_manager (DatabaseManager): The database manager.
        """
        self.model = model_interface
        self.db_manager = db_manager
        self.suggestion_history = []
        self.suggestion_cache = {}
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get suggestions based on partial user input.
        
        Args:
            partial_query (str): The partial user query.
            
        Returns:
            List[str]: A list of suggestions.
        """
        # If query is too short, return empty suggestions
        if len(partial_query.strip()) < 2:
            return []
        
        # Check cache first
        cache_key = partial_query.strip().lower()
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]
        
        try:
            # Get schema information
            schema = self.db_manager.get_schema()
            tables = self.db_manager.get_tables()
            
            # Prepare schema information for the model
            schema_info = []
            for table_name in tables:
                table_info = f"Table: {table_name}\nColumns: "
                if table_name in schema:
                    columns = [f"{col['name']} ({col['type']})" for col in schema[table_name]]
                    table_info += ", ".join(columns)
                schema_info.append(table_info)
            
            schema_text = "\n\n".join(schema_info)
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(partial_query, schema_text)
            
            # Generate suggestions using the model
            response = self.model.generate(prompt, max_tokens=500, temperature=0.7)
            
            # Parse and validate suggestions
            suggestions = self._parse_suggestions(response)
            
            # Cache the result
            self.suggestion_cache[cache_key] = suggestions
            
            # Update suggestion history
            self.suggestion_history.append({
                'input': partial_query,
                'suggestions': suggestions,
                'timestamp': datetime.now().isoformat()
            })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return []
    
    def clear_cache(self) -> None:
        """Clear the suggestion cache."""
        self.suggestion_cache = {}
    
    def _build_enhanced_prompt(self, user_input: str, schema_info: str) -> str:
        """Build an enhanced prompt using multiple prompting techniques.
        
        Args:
            user_input (str): The user's input text.
            schema_info (str): The schema information as a string.
            
        Returns:
            str: The enhanced prompt.
        """
        # Role Prompt
        role_prompt = """You are an expert SQL query suggestion system with deep knowledge of database operations and natural language processing. 
Your goal is to help users formulate effective database queries by providing relevant and helpful suggestions."""

        # Emotion Prompt
        emotion_prompt = """Approach each suggestion with empathy and understanding. Consider the user's potential frustration or confusion 
when dealing with complex data queries. Provide suggestions that are not only technically correct but also user-friendly and encouraging."""

        # Chain of Thought Prompt
        cot_prompt = """Let's think through this step by step:
1. Analyze the user's input and identify key concepts
2. Consider the database schema and available tables/columns
3. Think about common query patterns and best practices
4. Generate suggestions that are both relevant and helpful
5. Ensure suggestions are clear and actionable"""

        # Few-Shot Examples
        few_shot_examples = """
Example 1:
User Input: "Show me sales data"
Thought Process: User wants to see sales information, likely needs aggregation and filtering
Suggestion: "Show me total sales by product category for the last quarter"

Example 2:
User Input: "Who are our top customers?"
Thought Process: User needs customer analysis with ranking and aggregation
Suggestion: "List the top 10 customers by total purchase amount in the last year"

Example 3:
User Input: "Compare performance"
Thought Process: User needs comparative analysis, should suggest time-based comparison
Suggestion: "Compare monthly sales performance between this year and last year"
"""

        # Context Analysis
        context_analysis = ""
        if self.suggestion_history:
            recent_queries = [item['input'] for item in self.suggestion_history[-3:]]
            if recent_queries:
                context_analysis = f"\nRecent user queries:\n" + "\n".join(f"- {q}" for q in recent_queries)

        # Build the complete prompt
        prompt = f"""{role_prompt}

{emotion_prompt}

{cot_prompt}

{few_shot_examples}

Database Schema:
{schema_info}

{context_analysis}

Current User Input: {user_input}

Based on the above information, generate 3-5 relevant and helpful query suggestions. 
Each suggestion should be:
1. Clear and specific
2. Relevant to the user's input
3. Feasible with the available database schema
4. Written in natural language
5. Different from previous suggestions

Suggestions:"""

        return prompt
        
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse and validate the model's response into a list of suggestions.
        
        Args:
            response (str): The model's response text.
            
        Returns:
            List[str]: A list of validated suggestions.
        """
        try:
            # Split response into lines and clean up
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Extract suggestions (assuming they're numbered or bulleted)
            suggestions = []
            for line in lines:
                # Remove numbering or bullets
                clean_line = re.sub(r'^\d+\.\s*|^-\s*', '', line)
                if clean_line and len(clean_line) > 10:  # Basic validation
                    suggestions.append(clean_line)
            
            # Limit to 5 suggestions
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error parsing suggestions: {str(e)}")
            return []
    
    def get_database_suggestions(self) -> List[str]:
        """Get database-specific suggestions based on the current database schema.
        
        Returns:
            List[str]: A list of relevant query suggestions for the database.
        """
        try:
            # Get schema information
            schema = self.db_manager.get_schema()
            tables = self.db_manager.get_tables()
            
            if not tables:
                return [
                    "No tables found in the database",
                    "Please check your database connection"
                ]
            
            # Prepare schema information for the model
            schema_info = []
            for table_name in tables:
                table_info = f"Table: {table_name}\nColumns: "
                if table_name in schema:
                    columns = [f"{col['name']} ({col['type']})" for col in schema[table_name]]
                    table_info += ", ".join(columns)
                schema_info.append(table_info)
            
            schema_text = "\n\n".join(schema_info)
            
            # Build prompt for database-specific suggestions
            prompt = self._build_database_suggestions_prompt(schema_text, tables)
            
            # Generate suggestions using the model
            response = self.model.generate(prompt, max_tokens=500, temperature=0.7)
            
            # Parse and validate suggestions
            suggestions = self._parse_suggestions(response)
            
            # If no suggestions generated, provide fallback suggestions
            if not suggestions:
                suggestions = self._generate_fallback_suggestions(tables)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating database suggestions: {str(e)}")
            # Return fallback suggestions if there's an error
            return self._generate_fallback_suggestions(tables)
    
    def _generate_fallback_suggestions(self, tables: List[str]) -> List[str]:
        """Generate fallback suggestions when AI generation fails.
        
        Args:
            tables (List[str]): List of table names.
            
        Returns:
            List[str]: Fallback suggestions.
        """
        if not tables:
            return ["No tables available for querying"]
        
        # Generate basic suggestions based on table names
        suggestions = []
        
        for table in tables[:3]:  # Limit to first 3 tables
            table_lower = table.lower()
            if 'customer' in table_lower or 'user' in table_lower:
                suggestions.append(f"Show me all {table}")
                suggestions.append(f"Find {table} by name")
            elif 'order' in table_lower or 'transaction' in table_lower:
                suggestions.append(f"Show recent {table}")
                suggestions.append(f"Calculate total {table} amount")
            elif 'product' in table_lower or 'item' in table_lower:
                suggestions.append(f"List all {table}")
                suggestions.append(f"Find {table} by category")
            else:
                suggestions.append(f"Show me the {table} data")
                suggestions.append(f"What's in the {table} table?")
        
        # Add some general suggestions
        if len(suggestions) < 4:
            suggestions.extend([
                "Show me a sample of the data",
                "What tables are available?",
                "Generate a summary report"
            ])
        
        return suggestions[:4]  # Return max 4 suggestions
    
    def _build_database_suggestions_prompt(self, schema_info: str, tables: List[str]) -> str:
        """Build a prompt for generating database-specific suggestions.
        
        Args:
            schema_info (str): The schema information as a string.
            tables (List[str]): List of table names.
            
        Returns:
            str: The prompt for generating suggestions.
        """
        prompt = f"""You are an expert database query suggestion system. Based on the following database schema, generate 4 relevant and helpful query suggestions that users might want to ask.

DATABASE SCHEMA:
{schema_info}

AVAILABLE TABLES: {', '.join(tables)}

Generate 4 natural language query suggestions that:
1. Are relevant to the actual database structure
2. Would provide useful insights or information
3. Are written in clear, natural language
4. Cover different types of queries (SELECT, aggregation, filtering, etc.)
5. Are specific enough to be actionable but general enough to be useful

Examples of good suggestions:
- "Show me the top 10 customers by total order value"
- "What's the monthly sales trend for this year?"
- "List products with inventory below 10 units"
- "Find customers who haven't placed an order in the last 30 days"

Return your suggestions as a JSON array of strings, e.g.:
["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4"]

Format your response as valid JSON only."""

        return prompt


class ExplanationAgent:
    """
    Explanation Agent: Generates explanation for the SQL query and result in natural language.
    """
    
    def __init__(self, model_interface: ModelInterface):
        """Initialize the explanation agent.
        
        Args:
            model_interface (ModelInterface): The model interface to use.
        """
        self.model = model_interface
    
    def generate_explanation(self, query: str, sql: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an explanation for the SQL query and result.
        
        Args:
            query (str): The original user query.
            sql (str): The generated SQL query.
            results (List[Dict[str, Any]]): The query results.
            
        Returns:
            Dict[str, Any]: A dictionary containing summary and detailed explanation.
        """
        # Prepare a sample of the results
        results_sample = results[:5] if results else []
        results_json = json.dumps(results_sample, default=str)
        
        # Count total results
        total_results = len(results)
        
        # Craft a prompt for the model using multiple prompting techniques
        prompt = f"""You are an expert data analyst and SQL educator with a passion for making complex data concepts accessible to everyone. Your role is to explain SQL queries and their results in a clear, engaging, and helpful way.

Let's analyze this step by step using chain-of-thought reasoning:

1. First, understand the user's intent:
   - What was the user trying to find out?
   - What specific information were they looking for?
   - What context might they need to understand the results?

2. Then, analyze the SQL query:
   - What tables are being queried?
   - What conditions are being applied?
   - What aggregations or transformations are happening?
   - How does the query structure affect the results?

3. Next, examine the results:
   - What patterns or trends do you see?
   - Are there any notable values or outliers?
   - What insights can be drawn from the data?
   - How do the results relate to the user's original question?

4. Finally, craft two types of explanations:
   a) A brief summary (2-3 bullet points) highlighting key insights
   b) A detailed explanation connecting everything together


Here are some examples of good explanations:

Example 1:
Query: "Show me the total sales by product"
SQL: "SELECT product_name, SUM(sales_amount) as total_sales FROM sales GROUP BY product_name ORDER BY total_sales DESC;"
Results: [{{"product_name": "Widget A", "total_sales": 5000}}, {{"product_name": "Widget B", "total_sales": 3000}}]
Summary: [
    "Widget A leads sales with $5,000, 67% higher than Widget B",
    "Total sales across products amount to $8,000"
]
Detailed: "I've analyzed the sales data across all products. Widget A is our top performer with total sales of $5,000, followed by Widget B at $3,000. This gives us a clear picture of which products are driving our revenue."

Example 2:
Query: "What's the average order value by customer type?"
SQL: "SELECT customer_type, AVG(order_value) as avg_order FROM orders GROUP BY customer_type;"
Results: [{{"customer_type": "Premium", "avg_order": 150.50}}, {{"customer_type": "Standard", "avg_order": 75.25}}]
Summary: [
    "Premium customers spend 2x more per order than Standard customers",
    "Average order value gap: $75.25 between customer segments"
]
Detailed: "Looking at the average order values, Premium customers spend significantly more per order ($150.50) compared to Standard customers ($75.25). This suggests that our Premium customers are more valuable to the business on a per-order basis."


Example 3:
Query: "Show me the monthly sales trend"
SQL: "SELECT month, SUM(sales) as total_sales FROM sales GROUP BY month ORDER BY month;"
Results: [{{"month": "2024-01", "total_sales": 10000}}, {{"month": "2024-02", "total_sales": 12000}}]
Summary: [
    "20% month-over-month growth in sales",
    "February sales reach $12,000, highest in the period"
]
Detailed: "The sales data shows a positive trend, with February's sales ($12,000) showing a 20% increase compared to January ($10,000). This upward trend suggests growing business performance."


Now, let's analyze this query:

USER QUERY:
{query}

SQL QUERY:
{sql}

QUERY RESULTS (sample of {min(5, total_results)} out of {total_results} total):
{results_json}

Remember to:
1. Be clear and concise
2. Focus on what the user wants to know
3. Highlight important patterns or insights
4. Use natural, conversational language
5. Avoid technical jargon unless necessary
6. Connect the results back to the user's original question

Please provide both a brief summary (2-3 bullet points) and a detailed explanation that helps the user understand both what the query did and what the results mean.

Format your response as a JSON object with two fields:
- summary: An array of 2-3 bullet points highlighting key insights
- detailed: A comprehensive explanation of the query and results

"""
        
        # Get the model's response
        response = self.model.generate(prompt, temperature=0.3)
        
        try:
            # Find and extract the JSON object
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                explanation_data = json.loads(json_str)
            else:
                # If no JSON found, create a default response
                explanation_data = {
                    "summary": ["Could not generate summary"],
                    "detailed": f"Could not generate explanation: {response}"
                }
            
            return explanation_data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default response
            logger.error(f"Failed to parse explanation response: {response}")
            return {
                "summary": ["Could not generate summary"],
                "detailed": f"Could not generate explanation: {response}"
            }


class FeedbackAgent:
    """
    Records and processes user feedback on query responses.
    """
    
    def __init__(self, feedback_file: str = "feedback.jsonl"):
        """Initialize the feedback agent.
        
        Args:
            feedback_file (str, optional): Path to the feedback file.
        """
        self.feedback_file = feedback_file
        
        # Ensure the feedback directory exists
        feedback_dir = os.path.dirname(feedback_file)
        if feedback_dir and not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir, exist_ok=True)
    
    def record_feedback(self, query_id: str, query: str, sql: str, 
                        results: List[Dict[str, Any]], rating: str) -> None:
        """Record user feedback.
        
        Args:
            query_id (str): The unique ID of the query.
            query (str): The original user query.
            sql (str): The generated SQL query.
            results (List[Dict[str, Any]]): The query results.
            rating (str): The user rating (good, neutral, bad).
        """
        # Create feedback entry
        feedback_entry = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "sql": sql,
            "results_count": len(results),
            "rating": rating
        }
        
        # Write to feedback file
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of collected feedback.
        
        Returns:
            Dict[str, Any]: A summary of the feedback.
        """
        # Check if feedback file exists
        if not os.path.exists(self.feedback_file):
            return {
                "total_feedback": 0,
                "ratings": {"good": 0, "neutral": 0, "bad": 0},
                "percentage": {"good": 0, "neutral": 0, "bad": 0}
            }
        
        # Load feedback entries
        entries = []
        with open(self.feedback_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        
        # Count ratings
        ratings = {"good": 0, "neutral": 0, "bad": 0}
        for entry in entries:
            rating = entry.get("rating", "").lower()
            if rating in ratings:
                ratings[rating] += 1
        
        # Calculate percentages
        total = sum(ratings.values())
        percentage = {
            key: round((count / total) * 100) if total > 0 else 0
            for key, count in ratings.items()
        }
        
        return {
            "total_feedback": total,
            "ratings": ratings,
            "percentage": percentage
        }


class SQLGenerator:
    """
    Generates SQL queries from natural language using the model.
    """
    
    def __init__(self, model_interface: ModelInterface, db_manager: DatabaseManager):
        """Initialize the SQL generator."""
        self.model = model_interface
        self.db_manager = db_manager
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum seconds between requests
    
    def _get_date_format_function(self) -> str:
        """Get the appropriate date formatting function based on the database type."""
        if self.db_manager.db_type == 'sqlite':
            return "STRFTIME"
        elif self.db_manager.db_type == 'mysql':
            return "DATE_FORMAT"
        elif self.db_manager.db_type == 'postgresql':
            return "TO_CHAR"
        else:
            return "STRFTIME"  # Default to SQLite format
    
    def _clean_sql_query(self, sql: str) -> str:
        """Clean up the generated SQL query by removing any extraneous text."""
        # First, remove any markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove any common prefixes
        prefixes = [
            "SQL QUERY:",
            "Here's the SQL query:",
            "The SQL query is:",
            "Generated SQL:",
            "Query:",
            "SQL:"
        ]
        
        for prefix in prefixes:
            if sql.lower().startswith(prefix.lower()):
                sql = sql[len(prefix):].strip()
        
        # Remove any numbered instructions (e.g., "1.", "2.", etc.)
        sql = re.sub(r'^\d+\.\s*', '', sql, flags=re.MULTILINE)
        
        # Remove any lines that look like instructions
        instruction_patterns = [
            r'^If you are unable to convert the query.*$',
            r'^If there is any ambiguity.*$',
            r'^If you have any questions.*$',
            r'^If you are unsure.*$',
            r'^Please provide.*$',
            r'^Please explain.*$',
            r'^Please clarify.*$'
        ]
        
        for pattern in instruction_patterns:
            sql = re.sub(pattern, '', sql, flags=re.MULTILINE)
        
        # Split into lines and keep only lines that look like SQL
        sql_lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line and not any(line.startswith(prefix) for prefix in ['If', 'Please', 'You']):
                sql_lines.append(line)
        
        sql = ' '.join(sql_lines)
        
        # Ensure the query starts with a SQL command
        sql_commands = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        for cmd in sql_commands:
            if sql.upper().startswith(cmd):
                break
        else:
            # If no SQL command found at start, try to find it in the text
            for cmd in sql_commands:
                if cmd in sql.upper():
                    sql = sql[sql.upper().index(cmd):]
                    break
        
        # Clean up whitespace and ensure semicolon
        sql = ' '.join(sql.split())  # Normalize whitespace
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
    
    def _handle_rate_limit(self):
        """Handle rate limiting by adding delay between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_sql(self, query: str) -> str:
        """Generate a SQL query from natural language."""
        try:
            # Handle rate limiting
            self._handle_rate_limit()
            
            # Get schema information
            schema_desc = self.db_manager.get_schema_description()
            
            # Get table samples
            tables = self.db_manager.get_tables()
            table_samples = {}
            for table in tables[:5]:  # Limit to 5 tables to avoid prompt size issues
                sample = self.db_manager.get_table_sample(table, limit=3)
                if sample:
                    table_samples[table] = sample
            
            # Format table samples as text using custom JSON encoder
            sample_text = ""
            for table, samples in table_samples.items():
                sample_text += f"Table: {table}\n"
                if samples:
                    try:
                        sample_text += json.dumps(samples[:3], indent=2, cls=CustomJSONEncoder) + "\n\n"
                    except Exception as e:
                        logger.error(f"Error serializing samples for table {table}: {str(e)}")
                        continue
            
            # Get the appropriate date formatting function
            date_func = self._get_date_format_function()
            
            # Craft a prompt for the model
            prompt = f"""You are an expert SQL assistant that converts natural language queries into SQL. Let's solve this step by step.

Database Schema:
{schema_desc}

Sample Data:
{sample_text}

Database Type: {self.db_manager.db_type.upper()}
Date Format Function: {date_func}

User Query: "{query}"

Let's analyze this step by step:

1. Schema Understanding:
   - What tables are relevant to this query?
   - What columns do we need?
   - What relationships between tables are important?

2. Query Requirements:
   - What data is the user asking for?
   - What conditions need to be applied?
   - What aggregations or groupings are needed?

3. SQL Construction:
   - Start with the SELECT clause
   - Add necessary JOINs
   - Apply WHERE conditions
   - Add GROUP BY if needed
   - Add ORDER BY if needed
   - Add LIMIT if needed

4. Validation:
   - Check if all referenced tables exist
   - Verify column names are correct
   - Ensure proper JOIN conditions
   - Validate the query logic

Now, write the SQL query that answers the user's request.
Return ONLY the SQL query without any explanations or markdown formatting.
Use proper SQL syntax for {self.db_manager.db_type.upper()}.
For date formatting, use {date_func} with appropriate format.
End the query with a semicolon.

SQL:"""
            
            # Get the model's response
            response = self.model.generate(prompt, temperature=0.1)
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(response)
            
            # Validate that the response is actually a SQL query
            if not any(sql_query.upper().startswith(cmd) for cmd in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                raise ValueError("Generated response is not a valid SQL query")
            
            # Log the cleaned query for debugging
            logger.debug(f"Generated SQL query: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error in generate_sql: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class QueryOrchestrator:
    """
    Orchestrates the query processing pipeline, coordinating between different agents.
    Uses llama.cpp for SQL generation and Groq for other tasks.
    Includes support for query cancellation.
    """
    
    @debug_logger
    def __init__(
        self,
        db_manager: DatabaseManager,
        api_key: str = None,
        model_name: str = "llama3-8b-8192",
        provider: Optional[str] = None,
    ):
        """Initialize the query orchestrator.
        
        Args:
            db_manager (DatabaseManager): The database manager.
            api_key (str, optional): The API key for Groq (used for non-SQL tasks).
            model_name (str, optional): The model name for Groq.
            provider (str, optional): Ignored - now uses llama.cpp for SQL, Groq for others.
        """
        logger.debug("Initializing QueryOrchestrator...")
        self.db_manager = db_manager

        # Initialize llama.cpp for SQL generation
        logger.info("Initializing llama.cpp for SQL generation...")
        try:
            model_path = os.environ.get("LLAMACPP_MODEL_PATH")
            if not model_path:
                raise RuntimeError("LLAMACPP_MODEL_PATH is not set. Required for SQL generation.")
            self.llama_model = LlamaCppInterface(model_path=model_path)
            logger.info("llama.cpp initialized successfully for SQL generation")
        except Exception as e:
            logger.error(f"Failed to initialize llama.cpp: {e}")
            raise

        # Initialize Groq for other tasks (guard, context, intent, explanation, visualization, suggestions)
        logger.info("Initializing Groq for other tasks...")
        try:
            groq_key = api_key or os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY is not set. Required for non-SQL tasks.")
            self.groq_model = GroqInterface(api_key=groq_key, model_name=model_name)
            logger.info("Groq initialized successfully for other tasks")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            raise

        # Initialize agents with appropriate models
        logger.debug("Initializing agents...")
        try:
            # Use Groq for non-SQL tasks
            self.guard_agent = GuardAgent(self.groq_model, db_manager)
            self.context_agent = ContextTrackingAgent(self.groq_model)
            self.intent_agent = IntentClassificationAgent(self.groq_model, self.context_agent)
            self.explanation_agent = ExplanationAgent(self.groq_model)
            self.visualization_agent = VisualizationAgent(self.groq_model)
            self.suggestion_agent = DynamicSuggestionAgent(self.groq_model, db_manager)
            
            # Use llama.cpp for SQL generation
            self.sql_generator = SQLGenerator(self.llama_model, db_manager)
            
            self.feedback_agent = FeedbackAgent()
            logger.debug("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @debug_logger
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return the results with cancellation support.
        
        Args:
            query (str): The natural language query.
            
        Returns:
            Dict[str, Any]: The query results and metadata.
        """
        logger.debug(f"Processing query: {query}")
        start_time = time.time()
        
        # Generate a unique query ID
        query_id = str(uuid.uuid4())
        logger.debug(f"Generated query ID: {query_id}")
        
        # Step 1: Check if the query is relevant to the database (Guard Agent)
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Checking query relevance...")
            classification, reason, suggestions = self.guard_agent.is_relevant(query)
            logger.debug(f"Query classification: {classification}, reason: {reason}")
            
            if classification == "NOT_RELEVANT":
                logger.warning(f"Query not relevant to database: {reason}")
                return {
                    "query_id": query_id,
                    "status": "error",
                    "message": f"Query not relevant to the database: {reason}",
                    "sql_query": None,
                    "results": None,
                    "explanation": None,
                    "summary": None,
                    "visualizable": False,
                    "processing_time": time.time() - start_time
                }
            elif classification == "NEED_MORE_INFO":
                logger.info(f"Query needs more information: {reason}")
                # Generate more specific suggestions if none were provided
                if not suggestions:
                    suggestions = self._generate_specific_suggestions(query)
                return {
                    "query_id": query_id,
                    "status": "need_more_info",
                    "message": reason,
                    "suggestions": suggestions,
                    "sql_query": None,
                    "results": None,
                    "explanation": None,
                    "summary": None,
                    "visualizable": False,
                    "processing_time": time.time() - start_time
                }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during relevance check: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query_id": query_id,
                "status": "error",
                "message": f"Error during relevance check: {str(e)}",
                "sql_query": None,
                "results": None,
                "explanation": None,
                "summary": None,
                "visualizable": False,
                "processing_time": time.time() - start_time
            }
        
        # Step 2: Classify the intent (Intent Classification Agent)
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Classifying query intent...")
            intent_data = self.intent_agent.classify(query)
            logger.debug(f"Intent classification: {intent_data}")
            
            # Handle explanation intent by reusing previous query results
            if intent_data["intent"] == "Explanation" and self.context_agent.history:
                logger.debug("Handling explanation intent...")
                # Get the last successful query from history
                for item in reversed(self.context_agent.history):
                    if item["response"].get("status") == "success":
                        # Generate a new explanation for the previous results
                        # Check for cancellation first
                        if asyncio.current_task().cancelled():
                            raise asyncio.CancelledError()
                            
                        explanation_data = self.explanation_agent.generate_explanation(
                            query,
                            item["response"].get("sql_query", ""),
                            item["response"].get("results", [])
                        )
                        return {
                            "query_id": query_id,
                            "status": "success",
                            "message": "Explanation generated for previous query",
                            "sql_query": item["response"].get("sql_query"),
                            "results": item["response"].get("results"),
                            "explanation": explanation_data["detailed"],
                            "summary": explanation_data["summary"],
                            "visualizable": item["response"].get("visualizable", False),
                            "visualization_suggestions": item["response"].get("visualization_suggestions"),
                            "processing_time": time.time() - start_time
                        }
                # If no previous successful query found
                return {
                    "query_id": query_id,
                    "status": "error",
                    "message": "No previous query results available to explain",
                    "sql_query": None,
                    "results": None,
                    "explanation": None,
                    "summary": None,
                    "visualizable": False,
                    "processing_time": time.time() - start_time
                }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during intent classification: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query_id": query_id,
                "status": "error",
                "message": f"Failed to classify intent: {str(e)}",
                "sql_query": None,
                "results": None,
                "explanation": None,
                "summary": None,
                "visualizable": False,
                "processing_time": time.time() - start_time
            }
        
        # Step 3: Enhance the query with context if needed (Context Tracking Agent)
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Checking query context...")
            is_related, related_idx = self.context_agent.is_related_to_previous(query)
            logger.debug(f"Query related to previous: {is_related}, index: {related_idx}")
            
            if is_related:
                logger.debug("Enhancing query with context...")
                enhanced_query = self.context_agent.get_context_enhanced_query(query)
                logger.debug(f"Enhanced query: {enhanced_query}")
            else:
                enhanced_query = query
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during context enhancement: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Use original query if context enhancement fails
            enhanced_query = query
        
        # Step 4: Generate SQL (SQL Generator)
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Generating SQL query...")
            sql = self.sql_generator.generate_sql(enhanced_query)
            logger.debug(f"Generated SQL: {sql}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate SQL: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query_id": query_id,
                "status": "error",
                "message": f"Failed to generate SQL: {str(e)}",
                "sql_query": None,
                "results": None,
                "explanation": None,
                "summary": None,
                "visualizable": False,
                "processing_time": time.time() - start_time
            }
        
        # Step 5: Execute the query (Database Manager)
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Executing SQL query...")
            results, error = self.db_manager.execute_query(sql)
            
            if error:
                logger.error(f"Error executing query: {error}")
                return {
                    "query_id": query_id,
                    "status": "error",
                    "message": f"SQL Error: {error}",
                    "sql_query": sql,
                    "results": None,
                    "explanation": None,
                    "summary": None,
                    "visualizable": False,
                    "processing_time": time.time() - start_time
                }
            logger.debug(f"Query executed successfully, got {len(results)} results")
            
            # Check for cancellation after execution
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query_id": query_id,
                "status": "error",
                "message": f"Failed to execute query: {str(e)}",
                "sql_query": sql,
                "results": None,
                "explanation": None,
                "summary": None,
                "visualizable": False,
                "processing_time": time.time() - start_time
            }
        
        # Step 6: Generate explanation and other post-processing
        try:
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            logger.debug("Generating explanation...")
            explanation_data = self.explanation_agent.generate_explanation(query, sql, results)
            logger.debug(f"Generated explanation: {explanation_data}")
            
            # Check for cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            # Step 7: Generate visualization suggestions (Visualization Agent)
            logger.debug("Generating visualization suggestions...")
            if results and len(results) > 0:
                columns = list(results[0].keys())
                visualization_suggestions = self.visualization_agent.suggest_visualizations(results, columns)
                logger.debug(f"Generated visualization suggestions: {visualization_suggestions}")
                visualizable = self.visualization_agent.is_visualizable(results)
                
                # Generate summary insights
                summary = self.visualization_agent.generate_summary(results) if results else []
            else:
                visualization_suggestions = None
                visualizable = False
                summary = []
                logger.debug("No results available for visualization suggestions")
            
            # Step 8: Add to context history (Context Tracking Agent)
            logger.debug("Adding to context history...")
            self.context_agent.add_interaction(query, {
                "sql": sql,
                "results": results[:5] if results else [],
                "explanation": explanation_data["detailed"],
                "summary": summary,
                "status": "success",
                "visualizable": visualizable,
                "visualization_suggestions": visualization_suggestions
            })
            
            # Calculate processing time
            processing_time = time.time() - start_time

            # Log execution time to Backend/query_times.json
            try:
                import json, os
                log_file = os.path.join(os.path.dirname(__file__), 'query_times.json')
                log_entry = {
                    'query_id': query_id,
                    'execution_time_sec': processing_time,
                    'timestamp': int(time.time())
                }
                if os.path.exists(log_file):
                    with open(log_file, 'r+') as f:
                        try:
                            data = json.load(f)
                        except Exception:
                            data = []
                        data.append(log_entry)
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                else:
                    with open(log_file, 'w') as f:
                        json.dump([log_entry], f, indent=2)
            except Exception as e:
                logger.error(f"Failed to log query execution time: {str(e)}")

            # Return the results in the format expected by the frontend
            return {
                "query_id": query_id,
                "status": "success",
                "message": "Query processed successfully",
                "sql_query": sql,
                "results": results,
                "explanation": explanation_data["detailed"],
                "summary": summary,
                "visualizable": visualizable,
                "visualization_suggestions": visualization_suggestions,
                "processing_time": processing_time
            }
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during post-processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Even if explanation/visualization fails, return the results
            return {
                "query_id": query_id,
                "status": "partial",
                "message": f"Query executed but failed during post-processing: {str(e)}",
                "sql_query": sql,
                "results": results,
                "explanation": None,
                "summary": [],
                "visualizable": False,
                "visualization_suggestions": None,
                "processing_time": time.time() - start_time
            }
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get suggestions based on partial user input.
        
        Args:
            partial_query (str): The partial user query.
        
        Returns:
            List[str]: A list of suggestions.
        """
        return self.suggestion_agent.get_suggestions(partial_query)
    
    def record_feedback(self, query_id: str, rating: str) -> None:
        """Record user feedback.
        
        Args:
            query_id (str): The unique ID of the query.
            rating (str): The user rating (good, neutral, bad).
        """
        # We don't have the original query and results here, so just record the ID and rating
        self.feedback_agent.record_feedback(
            query_id=query_id,
            query="",  # We don't have this information at this point
            sql="",    # We don't have this information at this point
            results=[],
            rating=rating
        )

    def _generate_specific_suggestions(self, query: str) -> List[str]:
        """Generate specific suggestions for a query that needs more information.
        
        Args:
            query (str): The user query that needs clarification.
            
        Returns:
            List[str]: A list of specific suggestions.
        """
        # Get schema information for context
        schema_desc = self.db_manager.get_schema_description()
        tables = self.db_manager.get_tables()
        
        # Craft a prompt for the model
        prompt = f"""
        You are an AI assistant that helps users clarify their database queries.
        
        DATABASE SCHEMA:
        {schema_desc}
        
        USER QUERY:
        {query}
        
        Based on the database schema and the user's query, generate 2-3 specific suggestions for how the user could make their query more precise.
        Each suggestion should be a complete, natural language query that the user could use.
        
        Return your suggestions as a JSON array of strings, e.g.:
        ["suggestion 1", "suggestion 2", "suggestion 3"]
        
        Format your response as valid JSON only.
        """
        
        try:
            # Get the model's response
            response = self.model.generate(prompt, temperature=0.3)
            
            # Find and extract the JSON array
            json_match = re.search(r'(\[.*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                suggestions = json.loads(json_str)
                
                # Limit to 3 suggestions
                suggestions = suggestions[:3]
                
                return suggestions
            else:
                # If no JSON found, return generic suggestions
                return [
                    f"Please specify which table you want to query from {', '.join(tables)}",
                    "Please provide more details about what information you're looking for",
                    "Please specify any conditions or filters you want to apply"
                ]
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            # Return generic suggestions if there's an error
            return [
                f"Please specify which table you want to query from {', '.join(tables)}",
                "Please provide more details about what information you're looking for",
                "Please specify any conditions or filters you want to apply"
            ]

    def get_database_suggestions(self) -> List[str]:
        """Get database-specific suggestions based on the current database schema.
        
        Returns:
            List[str]: A list of relevant query suggestions for the database.
        """
        return self.suggestion_agent.get_database_suggestions()
