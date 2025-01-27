from arcticdb import Arctic
from datetime import datetime
import tiktoken  # for token usage

# config. artic db
MONGO_URI = "mongodb://localhost:27017/" 
COLLECTION_NAME = "llm_usage_logs"

# initialize Arctic DB
arctic_store = Arctic(MONGO_URI)
if COLLECTION_NAME not in arctic_store.list_libraries():
    arctic_store.create_library(COLLECTION_NAME)
arctic_library = arctic_store[COLLECTION_NAME]


# function to calculate token count
def calculate_token_count(prompt, model="gpt-3.5-turbo"):
    try:
        tokenizer = tiktoken.encoding_for_model(model)
        tokens = tokenizer.encode(prompt)
        return len(tokens)
    except Exception:
        return 0


# function to log data to Arctic DB
def log_data_to_arctic(api_name, prompt, response, response_time, token_count):
    log_entry = {
        "api_name": api_name,
        "timestamp": datetime.utcnow(),
        "prompt": prompt,
        "response": response,
        "response_time_ms": response_time,
        "token_count": token_count,
    }
    arctic_library.write(log_entry["timestamp"].isoformat(), log_entry)
