import tiktoken
from transformers import DistilBertTokenizer
import datetime
import os
import json

# function to calculate token count
def calculate_token_count(prompt, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo" or model == "deepseek-chat":
        # using tiktoken for OpenAI models
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 tokenizer
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    elif model == "distilbert-base-cased":
        # Using Hugging Face tokenizer for DistilBERT
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        tokens = tokenizer.encode(prompt)
        return len(tokens)
    
    else:
        raise ValueError("Unsupported model for token count calculation.")

# function to log data to ArcticDB (placeholder)
def log_data_to_arctic(api_name, prompt, response, response_time, token_count):
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "api_name": api_name,
        "prompt": prompt,
        "response": response,
        "response_time_ms": response_time, 
        "token_count": token_count,
    }

    # saving log entry as a JSON file for now
    log_file = f"logs/{api_name}_log.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
