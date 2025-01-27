# Excel-based QnA using LLMs with Arctic DB Logging

This Streamlit app allows users to upload Excel files, explore the data, and ask questions using different language models (ChatGPT, DeepSeek, and Hugging Face). It logs token usage, response time, and payload to Arctic DB for analytics and monitoring.

---

## Features
- **Excel Data Exploration:** Upload Excel files, view data in Markdown format, and preview it in a DataFrame.
- **Question-Answering with LLMs:**
  - **ChatGPT**: Queries OpenAI's `gpt-3.5-turbo` model.
  - **DeepSeek**: Custom model using OpenAI API.
  - **Hugging Face**: Uses the `distilbert-base-cased` model for question-answering.
- **Arctic DB Integration:** Logs API usage metrics, including prompt, response, response time, and token count.
- **Markdown Conversion:** Displays sheet data in a readable Markdown table format.

---

## Installation

### Prerequisites
- Python 3.8 or above
- MongoDB (for Arctic DB)

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/excel-llm-qna.git
   cd excel-llm-qna
