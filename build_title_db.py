import json
import os

from pinecone_loader import DATA_FILE, UTF_8_ENCODING

OUT_DIR = 'db'
OUT_FILE = f'{OUT_DIR}/articles.jsonl'

# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

def process_line(line: str) -> dict:
    """Process a single line of the input data to extract necessary fields."""
    try:
        jsonl = json.loads(line)
        return {
            'url': jsonl.get('url', ''),
            'published_at': jsonl.get('published_at', ''),
            'title': jsonl.get('title', ''),
            'authors': jsonl.get('authors', '')
        }
    except json.JSONDecodeError:
        # Handle invalid JSON lines gracefully, log or raise if necessary
        print("Warning: Skipping invalid JSON line.")
        return None

def write_to_file(out_file, data: dict):
    """Write processed data to the output file."""
    out_file.write(json.dumps(data, ensure_ascii=False))
    out_file.write('\n')

if __name__ == '__main__':
    try:
        with open(DATA_FILE, 'r', encoding=UTF_8_ENCODING) as data_file, \
             open(OUT_FILE, 'w', encoding=UTF_8_ENCODING) as out_file:
            
            for i, line in enumerate(data_file, start=1):
                processed_data = process_line(line)
                if processed_data:
                    write_to_file(out_file, processed_data)
                    
            print(f"Data successfully written to {OUT_FILE}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
