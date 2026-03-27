# 📄 DocuMind — RAG-Powered PDF Q&A with Mistral 7B

> Ask questions about your PDF documents using a locally-run Mistral 7B model, HuggingFace embeddings, and LlamaIndex — all on Google Colab.

---

## 🧠 Overview

**DocuMind** is a Retrieval-Augmented Generation (RAG) pipeline that enables natural language question answering over uploaded PDF documents. It combines:

- 🤖 **Mistral-7B-Instruct** — a powerful open-source LLM loaded in 8-bit quantization for memory efficiency
- 🔢 **sentence-transformers/all-mpnet-base-v2** — for generating high-quality document embeddings
- 📚 **LlamaIndex** — to orchestrate document ingestion, vector indexing, and query routing
- ☁️ **Google Colab** — as the runtime environment (free GPU supported)

---

## 🏗️ Architecture

```
PDF Documents
     │
     ▼
SimpleDirectoryReader (LlamaIndex)
     │
     ▼
Text Chunking (chunk_size=1024)
     │
     ▼
HuggingFace Embeddings (all-mpnet-base-v2)
     │
     ▼
VectorStoreIndex
     │
     ▼
Query Engine  ◄──── User Question
     │
     ▼
Mistral-7B-Instruct (8-bit quantized)
     │
     ▼
Answer
```

---

## ⚙️ Tech Stack

| Component | Tool / Model |
|---|---|
| LLM | `mistralai/Mistral-7B-Instruct-v0.1` |
| Embedding Model | `sentence-transformers/all-mpnet-base-v2` |
| RAG Framework | `llama_index == 0.9.39` |
| Quantization | `BitsAndBytes` (8-bit) |
| PDF Loader | `pypdf` via `SimpleDirectoryReader` |
| Orchestration | `LangChain + LlamaIndex` |
| Runtime | Google Colab (GPU) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/documind.git
cd documind
```

### 2. Open in Google Colab

Upload the notebook to [Google Colab](https://colab.research.google.com/) and enable a **GPU runtime**:

> Runtime → Change runtime type → T4 GPU

### 3. Install Dependencies

Run the setup cells in the notebook, or manually install:

```bash
pip install pypdf
pip install -q transformers einops accelerate langchain bitsandbytes
pip install sentence_transformers
pip install llama_index==0.9.39
pip install langchain-community
```

### 4. Authenticate with HuggingFace

You need a HuggingFace account with access to the Mistral model. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

> ⚠️ **Never hardcode your token in public repositories.** Use environment variables or Colab Secrets instead.

### 5. Upload Your PDFs

Create a `Data/` directory and upload your PDF files:

```python
import os
os.makedirs("Data", exist_ok=True)
# Then upload your PDFs to the Data/ folder
```

### 6. Run the Pipeline

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("/content/Data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

response = query_engine.query("What is correlation?")
print(response)
```

---

## 🔐 Security Note

The original notebook contains a **hardcoded HuggingFace API token**. Before pushing to any public repository:

1. Revoke the exposed token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Use environment variables or Colab's built-in secret manager:

```python
from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))
```

---

## 📁 Project Structure

```
documind/
├── mistral_with_llamaindex.ipynb   # Main Colab notebook
├── Data/                           # Directory for uploaded PDFs
└── README.md
```

---

## 💡 Example Queries

Once your PDFs are indexed, you can ask questions like:

```python
query_engine.query("Summarize the main findings of the document.")
query_engine.query("What is the definition of correlation?")
query_engine.query("List the key recommendations from the report.")
```

---

## 📌 Requirements

- Python 3.8+
- Google Colab with GPU (T4 or better recommended)
- HuggingFace account with Mistral model access
- At least 12GB GPU VRAM (8-bit quantization reduces this significantly)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Mistral AI](https://mistral.ai/) for the Mistral-7B model
- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for model hosting and transformers
