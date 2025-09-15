# 🩺 MedPrivAgent – Privacy-Aware Clinical AI Assistant

**MedPrivAgent** is a local, privacy-preserving clinical assistant built using **Retrieval-Augmented Generation (RAG)** to help healthcare professionals query and summarize clinical information. It uses **BioGPT-Large** for generation, **Qdrant** for semantic retrieval, and supports **secure remote access** through **Tailscale**.

All computation runs locally (e.g., RTX 4060 laptop), ensuring no sensitive patient data leaves the device. The architecture simulates **Multi-Party Computation (MPC)** concepts for privacy-preserving workflows.

---

## 🧩 Core Features

### 🔍 Input
- Textual medical queries typed via a web UI
- Ingested files (PDF or TXT) auto-chunked into the vector DB

### 🧠 RAG Pipeline (BioGPT + Qdrant)
- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in **Qdrant**
- **Generator:** **BioGPT-Large** (by Microsoft) for medically accurate, context-constrained answers
- **Prompting:** Strict clinical prompt to avoid hallucinations

### 🔐 Privacy via MPC (Simulated)
- Key modules like vector search, file ingestion, and LLM invocation can be isolated or containerized to simulate privacy-preserving computation using MPC principles.

### 🌐 Remote Access via Tailscale
- **FastAPI** backend accessible via your secure **Tailscale IP**
- Web UI allows upload and querying from phone or laptop

---

## 🧪 Example Use Case (Layman Terms)

A doctor types:
> _"What is the treatment protocol for Type 2 diabetes?"_

- MedPrivAgent retrieves relevant medical text chunks from local files
- Passes them as **context** to BioGPT
- Returns a 1–2 paragraph clinical summary answer
- All computation happens **locally**, with no internet or API calls

---

## 📦 Tools & Components

| Component | Role |
|----------|------|
| BioGPT-Large | Medical text generation |
| MiniLM-L6-v2 | Lightweight embedding for retrieval |
| Qdrant | Local vector database |
| FastAPI | RESTful backend |
| Tailscale | Secure VPN for remote access |
| PyMuPDF / PDFMiner | File parsing and chunking |
| Simulated MPC | Modular privacy-preserving workflow logic |

---

## 📁 Project Structure

```
clinical-rag-agent/
├── app/
│   ├── rag_pipeline.py        # RAG pipeline (retriever + BioGPT)
│   ├── utils.py               # Cleaning, ingestion, preprocessing
│
├── api/
│   └── main.py                # FastAPI backend
│
├── vector_store/              # Qdrant local DB
│
├── static/
│   └── index.html             # Chat UI (HTML + JS)
│
├── uploads/                   # Uploaded PDFs / TXT files
├── requirements.txt
├── README.md
└── tailscale_setup.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/clinical-rag-agent
cd clinical-rag-agent
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
uv run uvicorn api.main:app --reload --port 8000
```

- Open browser: `http://localhost:8000`
- Upload PDF/TXT
- Ask clinical questions

---

## ✅ To-Do / Roadmap

- [x] BioGPT-based context-aware answering
- [x] Qdrant integration for fast retrieval
- [x] File upload + auto-ingestion pipeline
- [x] Clean prompt formatting and output postprocessing
- [ ] Add MPC containerization and benchmarking
- [ ] Enable feedback loop for user correction

---

## 📬 Contact

Made with ❤️ by [QuantumLeap Labs](https://github.com/prakhar105)  
Questions? Collabs? DM [@quantumleap](https://www.linkedin.com/in/...) or raise an issue in the repo.