# Excel QnA with LLMs

## 📌 Overview
This project is a **Streamlit** application that allows users to upload an **Excel file** and query the data using different **LLMs (Large Language Models)**, including:
- **OpenAI's ChatGPT**
- **DeepSeek Chat**
- **Hugging Face's DistilBERT**

The application processes the uploaded Excel data and interacts with the selected LLM to provide insightful answers to user queries.

## 🚀 Features
- **Upload Excel File**: Accepts `.xlsx` and `.xls` file formats.
- **View Data**: Allows users to preview sheets from the uploaded file.
- **Markdown Preview**: Converts sheet data to Markdown format.
- **Ask AI**: Queries an AI model based on summarized Excel data.
- **Token Calculation**: Displays token count before sending queries.
- **Logging**: Tracks API usage, response time, and token count.

## 📦 Installation
### Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### Install Dependencies
Run the following command to install required Python packages:
```sh
pip install streamlit pandas openai transformers tiktoken
```

## ▶️ Usage
### 1. Run the Streamlit App
```sh
streamlit run app.py
```

### 2. Upload an Excel File
- Click the `Upload Data` tab.
- Select an Excel file (`.xlsx` or `.xls`).
- The file will be processed automatically.

### 3. View Data
- Navigate to the `View Data` tab.
- Select a sheet using **radio buttons**.
- Preview data in a table format.

### 4. Markdown Preview
- Go to the `Markdown` tab.
- Select a sheet to view its **Markdown-formatted** data.

### 5. Ask AI
- Navigate to the `Ask AI` tab.
- Enter a **question** in the input field.
- Choose an AI model from the sidebar (**ChatGPT, DeepSeek, or Hugging Face**).
- Provide the **API key** (except for Hugging Face).
- View the AI-generated response.

## 🛠️ Project Structure
```
📂 Excel-QnA-LLM
│── app.py                 # Main Streamlit app
│── db_utils.py            # Functions for logging and token calculation
│── requirements.txt       # Dependencies
│── logs/                  # Stores API logs
│── README.md              # Project documentation
```

## ⚙️ Configuration
- **API Keys**: The application requires an **API key** for OpenAI and DeepSeek models. The user must enter the key in the **sidebar**.
- **Token Limit**: Queries exceeding **200 tokens** will prompt the user to **filter or summarize** the data.

## 📝 License
This project is **open-source** and available for modification and distribution.

---
Made using **Streamlit & LLMs**

