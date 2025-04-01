import time
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Initialize LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.9,
    max_tokens=500,
)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# Collect URLs from user input
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
vectorstore_path = "faiss_store_genai"

main_placeholder = st.empty()

if process_url_clicked:
    # Validate URLs: Remove empty and incorrectly formatted URLs
    valid_urls = [url.strip() for url in urls if url.strip().startswith(("http://", "https://"))]

    if not valid_urls:
        st.error("Please enter at least one valid URL starting with 'http://' or 'https://'.")
    else:
        try:
            # Load data from valid URLs
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("🔄 Data Loading... Please wait.")
            data = loader.load()

            if not data:
                st.error(" No content extracted from the provided URLs. The pages might be empty or inaccessible.")
            else:
                # Split data into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                main_placeholder.text("🔄 Splitting text into chunks...")
                docs = text_splitter.split_documents(data)

                if not docs:
                    st.error(" No text content available after splitting. Check the URLs or try different sources.")
                else:
                    # Generate embeddings
                    main_placeholder.text("🔄 Generating embeddings...")
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=os.getenv("GEMINI_API_KEY"),
                    )

                    # Create FAISS vector store
                    vectorstore_genai = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text(" Embedding Vector Store Created Successfully!")

                    # Save FAISS index
                    vectorstore_genai.save_local(vectorstore_path)

                    st.success("Data processing completed! FAISS store saved.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(vectorstore_path):
        # Ensure embeddings are defined before loading FAISS
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        # Load the question-answering chain
        chain = load_qa_chain(llm, chain_type="stuff")
        retriever = vectorstore.as_retriever()

        # Perform the query
        docs = retriever.get_relevant_documents(query)
        result = chain.run(input_documents=docs, question=query)

        st.header("Answer")
        # st.subheader(result)
        # st.write(result["answer"])
        st.text(result)

        # if isinstance(result, dict) and "sources" in result:
        #     sources = result["sources"]
        # else:
        #     sources = ""

        # if sources:
        #     st.subheader("Sources:")
        #     sources_list = sources.split("\n")
        #     for source in sources_list:
        #         st.write(source)

        # sources = result.get("sucess", "")

        # if sources:
        #     st.subheader("Sources:")
        #     sources_list = sources
        #     sources_list = sources.split('\n')
            
        #     for source in sources_list:
        #         st.write(source)