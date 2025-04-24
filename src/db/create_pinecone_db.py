import pandas as pd
import os
import uuid
import logging # Added logging
from pinecone import Pinecone, ServerlessSpec, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm.auto import tqdm # For progress bar

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # You could add a FileHandler here as well:
        # logging.FileHandler("vector_db_creation.log")
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# Load environment variables from .env file
load_dotenv(override=True)

# --- Configuration ---
CSV_FILE_PATH = "data/restaurants_menu_data_cleaned.csv"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")# Ensure API key and environment are set
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable not set.")
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Get Pinecone environment or use default
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
logger.info(f"Using Pinecone environment: {PINECONE_ENVIRONMENT}")

INDEX_NAME = "nuggetpine" # Choose a name for your Pinecone index
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Make sure this matches the model used
# Get embedding dimension from the model
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()
logger.info(f"Model loaded. Embedding dimension: {EMBEDDING_DIMENSION}")
BATCH_SIZE = 100 # Pinecone recommends batch sizes of 100 or less for upserts
# --- End Configuration ---

# Initialize Pinecone connection
logger.info("Initializing Pinecone connection...")
try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    logger.exception("Failed to initialize Pinecone connection.")
    exit()
logger.info("Pinecone connection initialized.")

# Check if the index exists
logger.info("Listing available Pinecone indexes...")
try:
    available_indexes = pinecone.list_indexes() # Get the IndexList object
    # Handle potential API changes or empty list scenarios
    index_names = []
    if available_indexes and hasattr(available_indexes, 'names'):
        index_names = available_indexes.names()     # Get the list of names by calling the method
    logger.info(f"Found indexes: {index_names}")
except Exception as e:
    logger.exception("Failed to list Pinecone indexes.")
    exit()


if INDEX_NAME not in index_names:
    logger.info(f"Index '{INDEX_NAME}' not found. Creating a new index...")
    # Determine if the environment suggests a serverless or pod-based index
    try:
        # Check if we're using serverless environment (gcp-starter is a pod-based environment)
        if "serverless" in PINECONE_ENVIRONMENT:
            # Extract cloud and region for serverless
            parts = PINECONE_ENVIRONMENT.split("-")
            cloud = parts[0] if len(parts) > 0 else "aws"
            region = "-".join(parts[1:]) if len(parts) > 1 else "us-west-2"
            
            logger.info(f"Creating serverless index (Cloud: {cloud}, Region: {region}) with dimension {EMBEDDING_DIMENSION}...")
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
        else:
            logger.info(f"Creating pod-based index (Environment: {PINECONE_ENVIRONMENT}) with dimension {EMBEDDING_DIMENSION}...")
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=PodSpec(environment=PINECONE_ENVIRONMENT)
            )
        logger.info(f"Index '{INDEX_NAME}' created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create Pinecone index '{INDEX_NAME}'.")
        exit()
else:
    logger.info(f"Using existing index: '{INDEX_NAME}'")

# Connect to the index
try:
    logger.info(f"Connecting to index '{INDEX_NAME}'...")
    index = pinecone.Index(INDEX_NAME)
    logger.info(f"Connected to index '{INDEX_NAME}'.")
except Exception as e:
    logger.exception(f"Failed to connect to Pinecone index '{INDEX_NAME}'.")
    exit()

# Load data from CSV
try:
    logger.info(f"Loading data from {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH)
    logger.info(f"Loaded {len(df)} rows from {CSV_FILE_PATH}.")
except FileNotFoundError:
    logger.error(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    logger.exception(f"Error loading CSV file: {CSV_FILE_PATH}")
    exit()

# --- Data Preparation and Upsert ---
logger.info("Preparing and upserting data to Pinecone...")

# Ensure 'description_clean' exists and handle potential NaN values
if 'description_clean' not in df.columns:
    logger.error("'description_clean' column not found in the CSV.")
    exit()

# Fill NaN in description_clean with an empty string or a placeholder
df['description_clean'] = df['description_clean'].fillna('')
logger.info("Filled NaN values in 'description_clean' column.")

# Using tqdm for progress bar, logging start/end and errors
logger.info(f"Starting upsert process in batches of {BATCH_SIZE}...")
total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Upserting batches"):
    batch_df = df.iloc[i:i + BATCH_SIZE]

    # Get descriptions for embedding
    descriptions = batch_df['description_clean'].astype(str).tolist()

    # Create embeddings
    try:
        embeddings = model.encode(descriptions).tolist()
    except Exception as e:
        logger.exception(f"Error encoding descriptions for batch starting at index {i}. Skipping batch.")
        continue # Skip this batch

    # Prepare batch for upsert
    batch_vectors = []
    try:
        for idx, (index_row, row) in enumerate(batch_df.iterrows()):
            metadata = row.drop('description_clean').to_dict()
            pinecone_metadata = {}
            for k, v in metadata.items():
                if pd.isna(v):
                    continue
                if isinstance(v, (str, int, float, bool)):
                    pinecone_metadata[k] = v
                elif isinstance(v, list):
                    if all(isinstance(item, str) for item in v):
                        pinecone_metadata[k] = v
                    else:
                        pinecone_metadata[k] = [str(item) for item in v]
                else:
                    pinecone_metadata[k] = str(v)

            unique_id = f"row_{index_row}_{uuid.uuid4()}"
            batch_vectors.append((unique_id, embeddings[idx], pinecone_metadata))
    except Exception as e:
        logger.exception(f"Error preparing metadata/vectors for batch starting at index {i}. Skipping batch.")
        continue # Skip this batch

    # Upsert batch to Pinecone
    if batch_vectors:
        try:
            index.upsert(vectors=batch_vectors)
            # logger.info(f"Upserted batch {i // BATCH_SIZE + 1}/{total_batches}") # Log successful batch upsert if needed
        except Exception as e:
            logger.exception(f"Error upserting batch {i // BATCH_SIZE + 1} (starts at index {i}).")
            # Consider more robust error handling: retry, log failed IDs, etc.

# Check the number of vectors in the index
try:
    index_stats = index.describe_index_stats()
    vector_count = index_stats.get('total_vector_count', 0)
except Exception as e:
    logger.exception("Failed to get index stats after upserting.")
    vector_count = "unknown"

logger.info("\n--------------------")
logger.info(f"Data upsert process completed.")
logger.info(f"Pinecone index '{INDEX_NAME}' now contains approximately {vector_count} vectors.")
logger.info("--------------------")

# Example Query (Requires adapting query logic for Pinecone)
logger.info("\nRunning a test query...")
# try:
#     query_embedding = model.encode("cheesy burger").tolist()
#     results = index.query(
#         vector=query_embedding,
#         top_k=2,
#         include_metadata=True
#     )
#     logger.info("Test query results:")
#     logger.info(results)
# except Exception as e:
#     logger.exception("Failed to run test query.") 