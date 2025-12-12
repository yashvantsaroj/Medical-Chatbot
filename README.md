# Medical Chatbot 🩺

A medical chatbot powered by **Generative AI**, **LangChain**, and **Pinecone Vector Database**. This project helps users answer medical queries using a Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Features
- **Retrieval-Augmented Generation (RAG)**: Fetches relevant medical context from PDFs.
- **Google Gemini Integration**: Uses `gemini-2.5-flash` for high-speed, accurate responses.
- **Vector Search**: Utilizes Pinecone and HuggingFace embeddings for efficient similarity search.
- **Robust Rate Limiting**: Built-in exponential backoff to handle API quota limits gracefully.
- **Modern Stack**: Built with Python 3.12, LangChain v0.2+ (LCEL), and Flask.

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Framework**: Flask
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **Vector DB**: Pinecone
- **Orchestration**: LangChain (LCEL)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)

## 📂 Project Structure
```
.
├── app.py               # Main Flask Application
├── src/
│   ├── helper.py        # PDF loading & Text Splitting
│   ├── store_index.py   # ETL Pipeline (PDF -> Pinecone)
│   ├── prompts.py       # Prompt Templates
├── templates/
│   └── chat.html        # Front-end UI
├── static/
│   └── style.css        # Styles
├── Data/                # Place your medical PDF files here
├── requirements.txt     # Python Dependencies
└── .env                 # API Keys Configuration
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yashvantsaroj/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 5. Build the Index (ETL)
This script processes the PDF files in the `Data/` folder and uploads vectors to Pinecone.
```bash
python src/store_index.py
```

### 6. Run the Application
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:8080`

## ⚠️ Notes on Rate Limits
The application uses the **Google Gemini Free Tier**. If you encounter speed issues or quota errors, the app includes an **automatic retry mechanism**.
- **Model Used**: `gemini-2.5-flash` (Optimized for performance and availability).
- **Fallback**: If you have access issues, you can run `python list_models.py` to see available models and update `app.py` accordingly.

## 🤝 Contributing
Feel free to open issues or submit pull requests.

## 📄 License
MIT License
