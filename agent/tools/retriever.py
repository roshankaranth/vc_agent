import os
import logging
from dotenv import load_dotenv
from langchain.tools import tool

from azure.cosmos import CosmosClient
from openai import OpenAI

azure_logger = logging.getLogger("azure.cosmos")
azure_logger.setLevel(logging.WARNING)
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

COSMOS_HOST = os.getenv("COSMOS_HOST")
COSMOS_KEY = os.getenv("COSMOS_KEY")
EMBEDDINGS_MODEL = "text-embedding-3-small"
DATABASE_NAME = "vectordb"
CONTAINER_NAME = "vc_docs"

openai_client = OpenAI()
cosmos_client = CosmosClient(COSMOS_HOST, COSMOS_KEY)

db = cosmos_client.get_database_client(DATABASE_NAME)
container = db.get_container_client(CONTAINER_NAME)

def vector_search(queries):
    """
    Performs a vector similarity search in Cosmos DB for a list of query strings.

    Args:
        queries (List[str]): A list of search queries to be embedded and searched.

    Returns:
        List[dict]: A list of document dictionaries retrieved from the database, sorted by similarity score.
    """
    logging.info(f"Performing vector search for user query.")
    results = []

    response = openai_client.embeddings.create(
            input=queries,
            model=EMBEDDINGS_MODEL,
        )
    
    embedding = response.data[0].embedding
     
    query = "SELECT TOP 20 c.id, c.text, c.metadata, VectorDistance(c.embedding, @query_vector) AS SimilarityScore FROM c ORDER BY VectorDistance(c.embedding, @query_vector)"
    result = list(container.query_items(
        query=query,
        parameters=[
            {"name": "@query_vector", "value": embedding}
        ],
        enable_cross_partition_query=True
    ))

    
    logging.info(f"Vector search retrieved {len(result)} total documents across all queries.")

    return result

@tool("rag_retrieve", return_direct=True)
def retriever(user_query : str):
    """
    Orchestrates the entire retrieval process from a user query to a final list of documents.

    Args:
        user_query (str): The original query from the user.

    Returns:
        List[dict]: The final, ranked list of retrieved documents to be used as context.
    """

    docs = vector_search(queries=[user_query])
    
    logging.info(f"Retriever finished. Sending back {len(docs)} documents to LLM.")

    formatted = [
        {
            "id": doc.get("id"),
            "text": doc.get("text"),
            "metadata": doc.get("metadata"),
            "similarity": doc.get("SimilarityScore"),
        }
        for doc in docs
    ]

    return {
        "result_count": len(formatted),
        "documents": formatted
    }
