# 📚 Document Q&A System

A powerful AI-powered document question-answering system built with FastAPI, OpenAI, and modern web technologies. Upload your documents and ask questions - get intelligent answers backed by relevant sources!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 📄 **Multiple Input Methods**
  - Manual text input with title and content
  - File upload support (PDF, DOCX, TXT, MD)
  - Drag-and-drop interface
  - Bulk file uploads

- 🤖 **AI-Powered Q&A**
  - Semantic search using OpenAI embeddings
  - Context-aware answers using GPT-3.5
  - Source attribution with similarity scores
  - Top-K retrieval for relevant context

- 🔍 **Vector Search**
  - Text chunking with overlapping windows
  - Cosine similarity matching
  - In-memory vector storage
  - Fast retrieval and indexing

- 💻 **Modern Web Interface**
  - Beautiful gradient UI design
  - Real-time API status monitoring
  - Document management (view, delete)
  - Progress indicators and error handling

## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- OpenAI API - Embeddings and GPT-3.5
- scikit-learn - Cosine similarity calculations
- PyPDF2 - PDF text extraction
- python-docx - DOCX text extraction

**Frontend:**
- HTML5, CSS3, JavaScript
- Responsive design
- Drag-and-drop file handling

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package installer)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-qa-system.git
cd document-qa-system
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```


### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
```

### 5. Create Project Structure

```bash
document-qa-system/
├── app.py                 # FastAPI backend
├── .env                    # Environment variables
├── req.txt                # Python dependencies
├── README.md              # This file
└── static/
    └── index.html         # Frontend interface
```

## 🎯 Running the Application

### Start the Server

```bash
# Make sure your virtual environment is activated
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```


## 📖 Usage Guide

### 1. Adding Documents

**Option A: Text Input**
1. Click on "Text Input" tab
2. Enter document title
3. Paste or type document content
4. Click "Add Document"

**Option B: File Upload**
1. Click on "File Upload" tab
2. Drag and drop files or click to select
3. Review selected files
4. Click "Upload Files"

Supported formats: PDF, DOCX, TXT, MD

### 2. Asking Questions

1. Type your question in the "Ask Questions" section
2. Click "Get Answer"
3. View the AI-generated answer with source references
4. Check similarity scores to see relevance

### 3. Managing Documents

- View all indexed documents in the "Indexed Documents" section
- See document metadata (ID, chunks count)
- Delete documents you no longer need
- Refresh the list to see updates

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend interface |
| POST | `/documents` | Add documents via JSON |
| POST | `/upload` | Upload document files |
| POST | `/query` | Ask questions |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| GET | `/health` | Check API status |

### Example API Calls

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "title": "Sample Document",
      "content": "This is the document content..."
    }]
  }'
```

**Upload files:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf" \
```

**Query documents:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "top_k": 3
  }'
```

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [OpenAI](https://openai.com/) - AI models and embeddings
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## 📧 Contact

Mohammad Haseeb Khan - haseebfarooq200@gmail.com

Project Link: [https://github.com/yourusername/document-qa-system](https://github.com/yourusername/document-qa-system)

---

⭐ If you find this project useful, please consider giving it a star!

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vector Search Explained](https://www.pinecone.io/learn/vector-search/)

## 🔄 Version History

- **v1.0.0** (2024)
  - Initial release
  - Text and file upload support
  - Basic Q&A functionality
  - Vector search implementation
