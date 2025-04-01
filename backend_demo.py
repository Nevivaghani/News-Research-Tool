# from langchain.document_loaders import TextLoader

# loader = TextLoader("nvidia_news.txt")

# data = loader.load()

# print(data[0].page_content)

# from langchain.document_loaders import CSVLoader
# loader = CSVLoader("movies.csv", source_column="title")
# data = loader.load()
# # print(len(data))\
# print(data[0].page_content)


# from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAI
import os 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import langchain

load_dotenv()
llm =  GoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.9, max_tokens = 500)

loader = UnstructuredURLLoader(urls = [
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
    ]
)
 
data = loader.load()
# print(len(data))
# print(data[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    # separators = ["\n\n", "\n", " "],
    chunk_size = 1000,
    chunk_overlap = 200
)

docs = text_splitter.split_documents(data)
print(len(docs))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorindex_genai = FAISS.from_documents(docs, embeddings)
print(vectorindex_genai)

file_path = "vector_index.pkl"
with open(file_path, "wb") as f:
    # pickle.dump(vectorindex_genai, f)
    vectorindex_genai.save_local("faiss_index")

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        # vectorIndex = pickle.load(f)
        # vectorIndex = FAISS.load_local("faiss_index", embeddings)
        vectorIndex = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorIndex.as_retriever())
# print(chain)

query = "what is the price of Tiago iCNG?"

langchain.debug = True

chain({"question" : query}, return_only_outputs =True)

