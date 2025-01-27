import streamlit as st
import pandas as pd
import json
import time
from transformers import pipeline

# for Arctic DB utilities
from arctic_db_utils import log_data_to_arctic, calculate_token_count


# excel data processing
def preprocess_excel(file):
    workbook = pd.ExcelFile(file)
    data = {}
    for sheet in workbook.sheet_names:
        df = workbook.parse(sheet)
        data[sheet] = df.to_dict(orient="records")
    return workbook.sheet_names, data


# function for markdown
def convert_to_markdown(dataframe):
    return dataframe.to_markdown(index=False, tablefmt="grid")


# deepseek model
def deepseek_model(api_key, data, user_query):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    prompt = f"{user_query}\nData: {json.dumps(data)}"

    # calculate token count
    token_count = calculate_token_count(prompt, model="deepseek-chat")

    # for response time
    start_time = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query},
            {"role": "user", "content": f"Data: {json.dumps(data)}"},
        ],
    )
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000  # converting to milliseconds

    response_content = response["choices"][0]["message"]["content"]

    # log to Arctic DB
    log_data_to_arctic(
        api_name="DeepSeek",
        prompt=prompt,
        response=response_content,
        response_time=response_time_ms,
        token_count=token_count,
    )

    return response_content


# OpenAi
def openai_model(api_key, data, user_query):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = f"{user_query}\nData: {json.dumps(data)}"

    token_count = calculate_token_count(prompt, model="gpt-3.5-turbo")

    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query},
            {"role": "user", "content": f"Data: {json.dumps(data)}"},
        ],
    )
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000  

    response_content = response["choices"][0]["message"]["content"]

    log_data_to_arctic(
        api_name="ChatGPT",
        prompt=prompt,
        response=response_content,
        response_time=response_time_ms,
        token_count=token_count,
    )

    return response_content


# huggingface mode;l
def huggingface_model(data, user_query):
    # combining the data into a single context
    context = " ".join(
        [
            f"Sheet: {sheet_name}, " + ", ".join([f"{k}: {v}" for row in rows for k, v in row.items()])
            for sheet_name, rows in data.items()
        ]
    )

    token_count = calculate_token_count(context, model="distilbert-base-cased")

    # pipeline for huggingface
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased")

    start_time = time.time()
    response = qa_pipeline(question=user_query, context=context)
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000  

    log_data_to_arctic(
        api_name="Hugging Face",
        prompt=user_query,
        response=response["answer"],
        response_time=response_time_ms,
        token_count=token_count,
    )

    return response["answer"]


# for UI
st.title("Excel-based QnA using LLMs with Markdown Support")
st.write("Upload an Excel file to explore its data and ask questions using AI.")

file = st.file_uploader("Upload an Excel File", type=["xlsx", "xls"])

if file:
    st.success("File uploaded successfully!")
    sheet_names, data = preprocess_excel(file)


    selected_sheet = st.radio("Select a sheet to process:", options=sheet_names)

    if selected_sheet:
        dataframe = pd.DataFrame(data[selected_sheet])

      
        markdown_data = convert_to_markdown(dataframe)

        st.markdown(f"### Data for '{selected_sheet}' (Markdown Format)")
        st.text_area(
            label="Marked Down Data",
            value=markdown_data,
            height=400, 
        )

       
        st.markdown("### Data Preview:")
        st.dataframe(dataframe)

       
        llm_options = ["ChatGPT", "DeepSeek", "Hugging Face"]
        llm_choice = st.selectbox("Choose a language model:", llm_options)

        if llm_choice:
            if llm_choice == "ChatGPT":
                api_key = st.text_input("Enter your ChatGPT API key:", type="password")
                if api_key:
                    user_query = st.text_input("Enter your question:")
                    if user_query:
                        st.write("Processing your query...")
                        try:
                            response = openai_model(api_key, data[selected_sheet], user_query)
                            st.success(f"Response from ChatGPT: {response}")
                            st.info("Token usage, response time, and payload logged to Arctic DB.")
                        except Exception as e:
                            st.error(f"Error querying ChatGPT: {e}")
            elif llm_choice == "DeepSeek":
                api_key = st.text_input("Enter your DeepSeek API key:", type="password")
                if api_key:
                    user_query = st.text_input("Enter your question:")
                    if user_query:
                        st.write("Processing your query...")
                        try:
                            response = deepseek_model(api_key, data[selected_sheet], user_query)
                            st.success(f"Response from DeepSeek: {response}")
                            st.info("Token usage, response time, and payload logged to Arctic DB.")
                        except Exception as e:
                            st.error(f"Error querying DeepSeek: {e}")
            elif llm_choice == "Hugging Face":
                user_query = st.text_input("Enter your question:")
                if user_query:
                    st.spinner("Processing your query...")
                    try:
                        response = huggingface_model(data, user_query)
                        st.success(f"Response from Hugging Face: {response}")
                        st.info("Token usage, response time, and payload logged to Arctic DB.")
                    except Exception as e:
                        st.error(f"Error querying Hugging Face: {e}")
