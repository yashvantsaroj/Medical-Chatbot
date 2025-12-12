from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import time
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted
from src.prompts import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    print("Error: Missing API Keys")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

# Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize LLM
# Switching to gemini-2.5-flash as it is available in your list and should have better rate limits.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, convert_system_message_to_human=True)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful medical assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
)

# LCEL RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Retry logic for Rate Limiting
def get_response_with_retry(input_text, max_retries=3):
    for i in range(max_retries):
        try:
            return rag_chain.invoke(input_text)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = (i + 1) * 10  # Exponentialish backoff: 10s, 20s, 30s
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return "I am currently experiencing high traffic. Please try again in 1 minute."

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(f"User Input: {input_text}")
    
    response = get_response_with_retry(input_text)
    
    print("Response:", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

