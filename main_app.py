import streamlit as st
import pandas as pd   # processing and reading excel data
import json       # converts data into a structured format for ai models
import time
from transformers import pipeline
from db_utils import log_data_to_arctic, calculate_token_count  # for log and token calculation



# excel data processing
def preprocess_excel(file):
    workbook = pd.ExcelFile(file)  # stores all the sheets in the file
    data = {}
    for sheet in workbook.sheet_names:  # iterates through each sheet
        df = workbook.parse(sheet)      # reads the current sheet into a df
        data[sheet] = df.to_dict(orient="records")  # Converts the df into a list of dictionaries
    return workbook.sheet_names, data

# markdown format
def convert_to_markdown(dataframe):     # from 157
    return dataframe.to_markdown(index=False, tablefmt="grid")  # converts to markdown

# summarize data
def summarize_data(data, max_rows=10, max_columns=None):
    summarized_data = {}             # limits the no. of rows sent to the llm i have set only 10
    for sheet_name, rows in data.items():  # iterates over each sheet and each rows in sheet    
        if max_columns:
            df = pd.DataFrame(rows)
            rows = df.iloc[:, :max_columns].to_dict(orient="records")   # converts only max_column and df back to list of dict
        summarized_data[sheet_name] = rows[:max_rows] # limits the max rows     
    return summarized_data    # dictionaries 

# deepSeek Model
def deepseek_model(api_key, summarized_data, user_query): 

    from openai import OpenAI     # imports openai class interacted through deepseek

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # sends req to deepseek

    prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"
    token_count = calculate_token_count(prompt, model="deepseek-chat")


    if token_count > 200:
        raise ValueError("Payload exceeds maximum token limit. Summarize or filter the data.")

    start_time = time.time()  # starts time before sending the request
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[      # these are the context provided to the model
            {"role": "system", "content": "You are a helpful assistant."},  # this sets the role of AI
            {"role": "user", "content": user_query},
            {"role": "user", "content": f"Data: {json.dumps(summarized_data)}"},
        ],
    )
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000
    response_content = response.choices[0].message.content

    
    log_data_to_arctic("DeepSeek", prompt, response_content, response_time_ms, token_count)  # tracks api usage
    return response_content, token_count, summarized_data

# openAI Model
def openai_model(api_key, summarized_data, user_query):


    from openai import OpenAI   

    client = OpenAI(api_key=api_key)

    prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"
    token_count = calculate_token_count(prompt, model="gpt-3.5-turbo")

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
    response_content = response.choices[0].message.content


    
    log_data_to_arctic("ChatGPT", prompt, response_content, response_time_ms, token_count)
    return response_content, token_count, summarized_data

# Hugging Face Model
def huggingface_model(summarized_data, user_query):

    context = " ".join(   # here we format the context 
        [
            f"Sheet: {sheet_name}, " + ", ".join([f"{k}: {v}" for row in rows for k, v in row.items()])
            for sheet_name, rows in summarized_data.items()  
        ]
    )
    token_count = calculate_token_count(context, model="distilbert-base-cased")

    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased")
    start_time = time.time()

    response = qa_pipeline(question=user_query, context=context)

    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    log_data_to_arctic("Hugging Face", user_query, response["answer"], response_time_ms, token_count)
    return response["answer"], token_count, summarized_data

# Streamlit UI
st.set_page_config(page_title="Excel QnA with LLMs", layout="wide")
st.title("📊 Excel-based QnA using LLMs")

# create tabs for navigation
tabs = st.tabs(["Upload Data", "View Data", "Markdown", "Ask AI"])

# sidebar for llm Selection and API Key input
with st.sidebar:

    llm_options = ["ChatGPT", "DeepSeek", "Hugging Face"]
    llm_choice = st.selectbox("Select an AI Model:", llm_options)
    api_key = st.text_input(f"Enter your {llm_choice} API key:", type="password")

with tabs[0]:  # Upload Data Tab

    st.header("📂 Upload Excel File")
    file = st.file_uploader("Upload an Excel File", type=["xlsx", "xls"])
    if file:
        st.success("File uploaded successfully!")

with tabs[1]:  # View Data Tab
    st.header("📋 View Data")
    if file:
        sheet_names, data = preprocess_excel(file)
        selected_sheet = st.radio("Select a sheet to process:", options=sheet_names, key="view_data_sheet")
        if selected_sheet:
            dataframe = pd.DataFrame(data[selected_sheet])
            st.dataframe(dataframe)

    else:
        st.warning("Please upload an Excel file first.")



with tabs[2]:  # Markdown Tab
    st.header("📄 Markdown Preview")
    if file:
        sheet_names, data = preprocess_excel(file)
        selected_sheet = st.radio("Select a sheet to preview in Markdown format:", options=sheet_names, key="markdown_sheet")
        if selected_sheet:
            dataframe = pd.DataFrame(data[selected_sheet])
            markdown_data = convert_to_markdown(dataframe)
            st.text_area("Marked Down Data", value=markdown_data, height=400, key="markdown_markdown")
    else:
        st.warning("Please upload an Excel file first.")


with tabs[3]:  # Ask AI 
    st.header("💡 Ask AI")
    if file:
        user_query = st.text_input("Enter your question:")


        if user_query:
            summarized_data = summarize_data(data)  # display token count after user enters the question
            prompt = f"{user_query}\nData: {json.dumps(summarized_data)}"
            token_count = calculate_token_count(prompt, model="gpt-3.5-turbo")

    
            st.success(f"Token count: {token_count}")

            if api_key:  # rpocess the query only if the api key is provided
                if llm_choice == "ChatGPT":
                    response_content, _, _ = openai_model(api_key, summarized_data, user_query)
                elif llm_choice == "DeepSeek":
                    response_content, _, _ = deepseek_model(api_key, summarized_data, user_query)
                elif llm_choice == "Hugging Face":
                    response_content, _, _ = huggingface_model(summarized_data, user_query)

                # display the AI response
                st.text_area("AI Response", value=response_content, height=200, key="ai_response")
            else:
                st.error("No API Key Provided.")

        else:
            st.warning("Please enter a question.")

    else:
        st.warning("Please upload an Excel file first.")


