import json
import os
import threading
from dotenv import load_dotenv
from pinecone import Pinecone
from unidecode import unidecode
from unstructured.chunking.basic import chunk_elements
from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
from unstructured.partition.text import partition_text

# Load environment variables
load_dotenv()

# Constants and Configuration
DATA_FILE = 'data/research_pubs.jsonl'
UTF_8_ENCODING = 'utf-8'
MIN_YEAR = 2021

# Initialize embedding encoder
embedding_encoder = OpenAIEmbeddingEncoder(
    config=OpenAIEmbeddingConfig(
        api_key=os.getenv('OPENAI_EMBEDDING_API_KEY'), 
        model_name=os.getenv('OPENAI_EMBEDDING_MODEL')
    )
)

# Initialize Pinecone client and index
pc = Pinecone()
index = pc.Index(host=os.getenv('PINECONE_HOST'))
index_stats_response = index.describe_index_stats()
print(f'Index stats: {index_stats_response}')

def process_text(text: str) -> list:
    """Process and chunk the text into smaller chunks."""
    elements = partition_text(text=unidecode(text))
    return chunk_elements(elements, max_characters=40_000)

def embed_chunks_in_thread(chunks: list, timeout: int = 120) -> bool:
    """Invoke embedding operation in a separate thread to avoid blocking."""
    success = False

    def embed_target():
        nonlocal success
        try:
            embedding_encoder.embed_documents(chunks)
            success = True
        except Exception as e:
            print(f"Error embedding chunks: {e}")

    thread = threading.Thread(target=embed_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"Error: Embedding timed out after {timeout} seconds.")
        return False
    return success

def upsert_to_pinecone(title: str, src: str, published_at: str, chunks: list) -> None:
    """Upsert chunks into Pinecone index."""
    vectors = [{
        'id': f'{src}|{i}',
        'values': chunk.embeddings,
        'metadata': {
            'title': title,
            'src': src,
            'published_at': published_at or str(MIN_YEAR),
            'text': chunk.text
        }
    } for i, chunk in enumerate(chunks, start=1)]

    try:
        rsp = index.upsert(vectors=vectors)
        print(f"Index response: {rsp}")
    except Exception as e:
        handle_upsert_error(e, vectors)

def handle_upsert_error(exception: Exception, vectors: list) -> None:
    """Handle errors during upsert operations."""
    error_message = str(exception)
    print(f"Error upserting vectors: {error_message}")

    if 'exceeds the maximum supported size' in error_message:
        print("Splitting vectors and retrying...")
        split_vectors(vectors)
    elif 'exceeds the limit of 40960 bytes per vector' in error_message:
        print("Upserting vectors one by one...")
        upsert_one_by_one(vectors)
    else:
        raise exception

def split_vectors(vectors: list) -> None:
    """Split vectors into two halves and upsert each part."""
    mid = len(vectors) // 2
    try:
        index.upsert(vectors=vectors[:mid])
        index.upsert(vectors=vectors[mid:])
    except Exception as e:
        print(f"Error during vector splitting: {e}")

def upsert_one_by_one(vectors: list) -> None:
    """Upsert vectors individually to avoid size limits."""
    for i, vector in enumerate(vectors):
        try:
            index.upsert(vectors=[vector])
        except Exception as e:
            print(f"Error upserting vector[{i}] to index: {e}")
            print(f"Error vector details: {vector['id']} {vector['metadata']}")
            raise

def load_to_vector_store(title: str, src: str, published_at: str, text: str) -> None:
    """Load content to Pinecone vector store."""
    chunks = process_text(text)
    if not embed_chunks_in_thread(chunks):
        print("Retrying embedding...")
        embed_chunks_in_thread(chunks, timeout=240)
    
    total_embeddings = sum(len(chunk.embeddings) for chunk in chunks)
    print(f"Processed {len(chunks)} chunks, Total Embeddings: {total_embeddings} for source: {src}")
    
    upsert_to_pinecone(title, src, published_at, chunks)

def process_data_file():
    """Process the input data file and load each record into Pinecone."""
    START_LINE = 1
    with open(DATA_FILE, 'r', encoding=UTF_8_ENCODING) as data_file:
        for i, line in enumerate(data_file, start=1):
            if i < START_LINE:
                continue
            print(f"Processing line {i} ...")
            
            try:
                jsonl = json.loads(line)
                url = jsonl['url']
                published_at = jsonl.get('published_at')
                title = jsonl['title']
                contents = jsonl['contents']
                load_to_vector_store(title, url, published_at, contents)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line {i}: {e}")
            except KeyError as e:
                print(f"Missing expected field {e} in line {i}")

if __name__ == '__main__':
    process_data_file()
