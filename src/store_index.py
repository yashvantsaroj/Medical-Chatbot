import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings, filter_to_minimal_docs
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found in environment variables.")
    exit(1)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Data Processing
print("Loading PDFs...")
extracted_data = load_pdf_file("Data")

print("Filtering Documents...")
# Optional: Use minimal docs if you want to reduce payload size
# minimal_docs = filter_to_minimal_docs(extracted_data) 
# We'll use the full extracted data for now to ensure we don't lose context, 
# but you can switch to minimal_docs if needed.
text_chunks = text_split(extracted_data) 

print("Downloading Embeddings...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

# Check if index exists, create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1") 
    )
else:
    print(f"Index '{index_name}' already exists.")

# Upsert to Pinecone
print("Upserting to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
print("Indexing Complete!")
