# 🔎 Hybrid Search with LangChain

A practical implementation of **Hybrid Search** using **LangChain**, combining **semantic vector retrieval** and **BM25 keyword retrieval** to improve document search quality.

This project demonstrates how multiple retrieval strategies can be combined using LangChain's `EnsembleRetriever` to retrieve more relevant context for RAG-based applications.

---

## 🚀 Overview

Traditional keyword search is good at finding **exact terms**, while vector search is good at understanding **semantic meaning**.

This project combines both approaches:

```text
                    User Query
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Semantic Search         BM25 Search
        (Vector Search)       (Keyword Search)
              │                   │
              └─────────┬─────────┘
                        ▼
                Ensemble Retriever
                        │
                        ▼
                 Ranked Results
```

The semantic retriever uses **Chroma + `all-MiniLM-L6-v2`**, while the sparse retriever uses **BM25**. Their results are combined using weighted retrieval through `EnsembleRetriever`.

---

## ✨ Key Features

* Semantic retrieval using Hugging Face embeddings
* BM25-based sparse/keyword retrieval
* Chroma vector store
* LangChain `EnsembleRetriever`
* Weighted combination of semantic and lexical retrieval
* Recursive text chunking
* Simple architecture suitable for understanding Hybrid Search concepts
* Jupyter Notebook implementation

---

## 🧠 How It Works

### 1. Text Chunking

The source text is split into smaller chunks using:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

split_docs = text_splitter.split_text(text)
```

Chunking allows the retrievers to work with smaller, more focused pieces of text instead of one large document.

---

### 2. Dense Embeddings

The project uses:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

`all-MiniLM-L6-v2` converts text into numerical embeddings that capture semantic meaning.

These embeddings are stored in Chroma for vector similarity search.

---

### 3. Chroma Vector Store

The chunks are indexed in Chroma:

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_texts(
    split_docs,
    embeddings
)
```

The semantic retriever is then created with:

```python
semantic_ret = db.as_retriever()
```

---

### 4. BM25 Retrieval

A second retrieval strategy is created using BM25:

```python
from langchain_community.retrievers import BM25Retriever

sparse_ret = BM25Retriever.from_texts(split_docs)
```

BM25 focuses on **lexical matching**, making it useful when exact words or terms in the query are important.

For example, if a query contains a specific name, technical term, or phrase, BM25 can provide strong results even when semantic similarity is not sufficient.

---

### 5. Ensemble Retrieval

The two retrievers are combined:

```python
from langchain_classic.retrievers import EnsembleRetriever

final_ret = EnsembleRetriever(
    retrievers=[semantic_ret, sparse_ret],
    weights=[0.7, 0.3]
)
```

The weights determine the contribution of each retriever:

```text
Semantic Search → 70%
BM25            → 30%
```

This allows semantic and keyword-based retrieval signals to contribute to the final ranking.

---

## 🧪 Example

The project uses a text passage about **Subhash Chandra Bose**.

A query such as:

```python
res = final_ret.invoke(
    "Who is Subhash Chandra Bose?"
)
```

returns the most relevant document chunks from the combined retrieval system.

---

## 🛠️ Tech Stack

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| Python              | Programming language       |
| LangChain           | Retrieval orchestration    |
| LangChain Community | Chroma & BM25 integrations |
| LangChain Classic   | Ensemble retrieval         |
| Hugging Face        | Embedding model            |
| `all-MiniLM-L6-v2`  | Dense embedding model      |
| Chroma              | Vector database            |
| BM25                | Sparse keyword retrieval   |
| Jupyter Notebook    | Development environment    |

---

## 📁 Project Structure

```text
hybrid-search/
│
├── hybrid.ipynb
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/hybrid-search.git
cd hybrid-search
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -U langchain langchain-community langchain-classic langchain-huggingface langchain-text-splitters chromadb rank-bm25 sentence-transformers
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
hybrid.ipynb
```

---

## 📌 Retrieval Architecture

```text
                        DOCUMENT
                            │
                    Recursive Chunking
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
            Hugging Face           BM25
             Embeddings
                   │                 │
                   ▼                 ▼
                Chroma          BM25 Retriever
                   │                 │
                   └────────┬────────┘
                            ▼
                  EnsembleRetriever
                      0.7 / 0.3
                            │
                            ▼
                    Retrieved Chunks
```

---

## 🔍 Why Hybrid Search?

### Vector Search

Understands semantic similarity.

For example:

```text
"freedom fighter"
```

can retrieve content discussing:

```text
"Indian independence leader"
```

even when the exact phrase does not appear.

### BM25

Focuses on actual terms present in the query and documents.

For example:

```text
"Subhash Chandra Bose"
```

can strongly favor chunks containing the exact name.

### Hybrid Search

Combining both approaches can provide a better balance between:

```text
Semantic Understanding + Exact Keyword Matching
```

---

## 🎯 Learning Objectives

This project was built to understand:

* How document chunking works
* How embedding models convert text into vectors
* How vector similarity search works
* How BM25 performs lexical retrieval
* How multiple retrievers can be combined
* How weighted retrieval works with `EnsembleRetriever`
* The role of retrieval in RAG pipelines

---

## 🔮 Future Improvements

Potential extensions include:

* Add a PDF/document loader
* Use metadata-aware retrieval
* Experiment with different chunk sizes and overlap
* Tune ensemble weights
* Add MMR for more diverse semantic results
* Add a cross-encoder reranker
* Integrate the retriever into a complete RAG pipeline
* Compare BM25, vector, and hybrid retrieval quantitatively
* Add evaluation metrics such as Recall@K and MRR

---

## 👨‍💻 Author

**Pranshu Awasthy**

Focused on learning and building practical applications with:

```text
Generative AI • RAG • LangChain • LLM Applications • AI Agents
```

---

## ⭐ If You Find This Useful

Give the repository a ⭐ and feel free to experiment with different retrieval strategies and ensemble weights.
