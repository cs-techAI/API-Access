import streamlit as st
import pandas as pd
import json
import time
from transformers import pipeline
from arctic_db_utils import log_data_to_arctic, calculate_token_count

# excel data processing
def preprocess_excel(file):
    workbook = pd.ExcelFile(file)
    data = {}
    for sheet in workbook.sheet_names:
        df = workbook.parse(sheet)
        data[sheet] = df.to_dict(orient="records")
    return workbook.sheet_names, data

# Markdown format
def convert_to_markdown(dataframe):
    return dataframe.to_markdown(index=False, tablefmt="grid")

# summarize
def summarize_data(data, max_rows=10, max_columns=None):
    summarized_data = {}

    # Loop through each sheet and summarize data
    for sheet_name, rows in data.items():
        # If max_columns is specified, limit columns
        if max_columns:
            df = pd.DataFrame(rows)
            rows = df.iloc[:, :max_columns].to_dict(orient="records")  # Limit columns
        
        # Limit the rows based on max_rows
        summarized_data[sheet_name] = rows[:max_rows]
    
    return summarized_data

# DeepSeek model
def deepseek_model(api_key, data, user_query):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Summarize data
    summarized_data = summarize_data(data)

    # Create prompt
    prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"

    # Calculate token count
    token_count = calculate_token_count(prompt, model="deepseek-chat")
    if token_count > 100:
        raise ValueError("Payload exceeds maximum token limit. Summarize or filter the data.")

    # Send API request
    start_time = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query},
            {"role": "user", "content": f"Data: {json.dumps(summarized_data)}"},
        ],
    )
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    # Log response
    response_content = response["choices"][0]["message"]["content"]
    log_data_to_arctic(
        api_name="DeepSeek",
        prompt=prompt,
        response=response_content,
        response_time=response_time_ms,
        token_count=token_count,
    )
    return response_content

# OpenAI (ChatGPT) model
def openai_model(api_key, data, user_query):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Summarize data
    summarized_data = summarize_data(data)

    # Create prompt
    prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"

    # Calculate token count
    token_count = calculate_token_count(prompt, model="gpt-3.5-turbo")

    # Send API request
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query},
            {"role": "user", "content": f"Data: {json.dumps(summarized_data)}"},
        ],
    )
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    # Log response
    response_content = response["choices"][0]["message"]["content"]
    log_data_to_arctic(
        api_name="ChatGPT",
        prompt=prompt,
        response=response_content,
        response_time=response_time_ms,
        token_count=token_count,
    )
    return response_content

# Hugging Face model
def huggingface_model(data, user_query):
    # Summarize data
    summarized_data = summarize_data(data)

    # Combine data into context
    context = " ".join(
        [
            f"Sheet: {sheet_name}, " + ", ".join([f"{k}: {v}" for row in rows for k, v in row.items()])
            for sheet_name, rows in summarized_data.items()
        ]
    )

    # Calculate token count
    token_count = calculate_token_count(context, model="distilbert-base-cased")

    # Hugging Face pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased")

    start_time = time.time()
    response = qa_pipeline(question=user_query, context=context)
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    # Log response
    log_data_to_arctic(
        api_name="Hugging Face",
        prompt=user_query,
        response=response["answer"],
        response_time=response_time_ms,
        token_count=token_count,
    )
    return response["answer"]

# Streamlit UI
st.title("Excel-based QnA using LLMs with Token Calculation")
st.write("Upload an Excel file to explore its data, calculate token usage, and ask questions using AI.")

file = st.file_uploader("Upload an Excel File", type=["xlsx", "xls"])

if file:
    st.success("File uploaded successfully!")
    sheet_names, data = preprocess_excel(file)

    selected_sheet = st.radio("Select a sheet to process:", options=sheet_names)

    if selected_sheet:
        dataframe = pd.DataFrame(data[selected_sheet])
        markdown_data = convert_to_markdown(dataframe)

        st.markdown(f"### Data for '{selected_sheet}' (Markdown Format)")
        st.text_area("Marked Down Data", value=markdown_data, height=400)
        st.markdown("### Data Preview:")
        st.dataframe(dataframe)

        llm_options = ["ChatGPT", "DeepSeek", "Hugging Face"]
        llm_choice = st.selectbox("Choose a language model:", llm_options)

        if llm_choice:
            user_query = st.text_input("Enter your question:")
            if user_query:
                # Pass the selected sheet data as a dictionary to summarize_data
                summarized_data = summarize_data({selected_sheet: data[selected_sheet]})
                
                # Generate the prompt for LLM
                prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"

                # Calculate token count
                token_count = calculate_token_count(prompt, model=llm_choice.lower() + "-chat")

                # Display token count and the summarized data
                st.info(f"Number of tokens for {llm_choice}: {token_count}")
                st.markdown(f"### Summarized Data (First {len(summarized_data[selected_sheet])} rows):")
                st.json(summarized_data[selected_sheet])  # Display summarized data for the selected sheet

                # After displaying the summarized data, ask for the API key
                api_key = st.text_input(f"Enter your {llm_choice} API key:", type="password")

                if api_key and st.button("Submit Query"):
                    try:
                        if llm_choice == "ChatGPT":
                            response = openai_model(api_key, data[selected_sheet], user_query)
                        elif llm_choice == "DeepSeek":
                            response = deepseek_model(api_key, data[selected_sheet], user_query)
                        elif llm_choice == "Hugging Face":
                            response = huggingface_model(data[selected_sheet], user_query)
                        st.success(f"Response from {llm_choice}: {response}")
                    except Exception as e:
                        st.error(f"Error querying {llm_choice}: {e}")
