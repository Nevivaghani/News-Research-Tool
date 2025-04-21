# 📰 News Research Tool

The **News Research Tool** is a **Streamlit-based** web application that enables users to **extract, process, and analyze** news articles from given URLs. It leverages **Google Generative AI (Gemini)** for question answering and **FAISS vector store** for efficient document retrieval.
## ✨ Features

- ✅ Accepts multiple news article URLs as input.
- ✅ Extracts and preprocesses text content.
- ✅ Splits documents into manageable chunks.
- ✅ Generates embeddings using **Google Generative AI**.
- ✅ Stores processed data in a **FAISS vector database**.
- ✅ Enables users to ask questions and retrieve relevant information.


## 🛠️ Technologies Used

- **Python 3.11+**
- **Streamlit** - Web UI framework
- **LangChain** - Document processing
- **FAISS** - Vector storage
- **Google Generative AI (Gemini)** - Embeddings & Q&A
- **dotenv** - API key management
## 📦 Installation

### 🔹 Prerequisites

- Ensure you have **Python 3.11+** installed.

### 🔹 Steps to Install

#### 1️⃣ Clone the repository:

```sh
git clone https://github.com/your-repo/news-research-tool.git
cd news-research-tool


2️⃣ Set up a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install dependencies:

poetry install

4️⃣ Set up API key:
Create a .env file and add your Google Generative AI API key:

GEMINI_API_KEY=your_api_key_here

5️⃣ Run the application:

Poetry run streamlit run news_research_tool.py

```


## 🚀 Usage Guide

1️⃣ Enter the news article URLs in the sidebar.

2️⃣ Click "Process URLs" to fetch and process the data.

3️⃣ Ask a question in the input box.

4️⃣ View the AI-generated answer along with relevant sources.
## Screenshots

![App Screenshot][def]

![App Screenshot][def2]

![App Screenshot][def3]

[def]: ./assets/news1.png

[def2]: ./assets/news2.png

[def3]: ./assets/news3.png
