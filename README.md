Overview
The News Research Tool is a Streamlit-based web application that enables users to extract, process, and analyze news articles from given URLs. It leverages Google Generative AI (Gemini) for question answering and FAISS vector store for efficient document retrieval.

Features
Accepts multiple news article URLs as input.

Extracts and preprocesses text content.

Splits documents into manageable chunks.

Generates embeddings using Google Generative AI.

Stores processed data in a FAISS vector database.

Enables users to ask questions and retrieve relevant information.

Technologies Used
Python 3.11+

Streamlit for the web UI

LangChain for document processing

FAISS for vector storage

Google Generative AI (Gemini) for embeddings and Q&A

dotenv for API key management

Installation
Prerequisites
Ensure you have Python 3.11+ installed.

Steps
Clone the repository:

git clone https://github.com/your-repo/news-research-tool.git
cd news-research-tool
Set up a virtual environment:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:


pip install -r requirements.txt
Or use Poetry:


poetry install
Set up your .env file with your Google Generative AI API key:


GEMINI_API_KEY=your_api_key_here
Run the application:


streamlit run news_research_tool.py
Usage
Enter the news article URLs in the sidebar.

Click "Process URLs" to fetch and process the data.

Ask a question in the input box.

View the AI-generated answer along with sources.