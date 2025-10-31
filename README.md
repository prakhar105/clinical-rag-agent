# 🩺 MedPrivAgent – Privacy-Aware Clinical RAG Assistant

![](https://github.com/prakhar105/clinical-rag-agent/blob/main/assets/logo.png)

**MedPrivAgent** is a local, privacy-conscious AI assistant designed for healthcare professionals. Built with **Retrieval-Augmented Generation (RAG)**, it enables secure querying over sensitive medical documents using **BioGPT-Large** and **Qdrant**, with advanced features like **continuous ingestion**, **semantic search**, and **MPC-inspired modular design**.

All processing is performed **locally**, and remote access is secured via **Tailscale**, making it ideal for clinics, hospitals, or research labs that need offline, confidential AI assistance.

![UI Preview](https://github.com/prakhar105/clinical-rag-agent/blob/main/assets/app.png)

---

##  Core Features

###  Query & Summarization
- Natural language questions via web UI
- Context-rich answers in clinical tone from **BioGPT**
- Built-in “Show Source Context” toggle for traceability

###  RAG Pipeline (Custom BioGPT + Qdrant)
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Generator**: Microsoft’s **BioGPT-Large**
- **Retriever**: Qdrant-based local vector search
- Strict medical prompt to reduce hallucinations

###  Privacy & Isolation
- No cloud APIs, no internet dependency
- Simulated **MPC-style** separation of modules
- Ideal for confidential patient or research data

###  Continuous Ingestion (Watchdog)
- Auto-detects new PDF/TXT files in `data/clinical_data/`
- Extracts content and pushes to vector DB with `scripts/continuous_ingestion_pipeline.py`
- Optimized for real-time hospital data flows

###  Remote Access via Tailscale
- UI + API served via **FastAPI**
- Access securely from any Tailscale-connected device

---

##  Directory Structure

```
MedPrivAgent/
├── app/
│   ├── main.py                   # FastAPI web server
│   ├── rag_pipeline.py          # End-to-end RAG logic
│   └── templates/index.html     # UI: chat & upload interface
│
├── assets/
│   ├── app.png                  # App screenshot
│   └── logo.png                 # Logo
│
├── data/
│   ├── clinical_data/           # Raw PDFs or .txt inputs
│   ├── processed_data/          # Cleaned and chunked text
│   └── offload_biogpt/          # (Optional) offloaded generations
│
├── scripts/
│   ├── continuous_ingestion_pipeline.py  # Watchdog ingestion
│   ├── data_preperation.py               # PDF/TXT to chunks
│   └── ingest_rag_files.py               # Manual ingestion tool
│
├── vector_store/               # Qdrant DB dump (persistent)
├── run.py                      # Entry script
├── run_server.py               # Dev server runner
├── README.md
├── pyproject.toml
└── .gitignore
```

---

##  Quick Start

```bash
git clone https://github.com/yourusername/MedPrivAgent.git
cd MedPrivAgent
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate
uv pip install -r requirements.txt
uv run run_server.py
```

Then open: [http://localhost:8000](http://localhost:8000)

---

##  Continuous File Ingestion

To enable real-time file ingestion:

```bash
uv run scripts/continuous_ingestion_pipeline.py
```

This will:
- Watch the `data/clinical_data` folder for changes
- Automatically chunk new files
- Update the vector database with fresh content

You can optionally schedule this with **systemd**, **supervisord**, or a **cron job** for background ingestion.

---

##  Sample Query

> _"What is the treatment protocol for Type 2 diabetes?"_

- 🔍 Retrieves relevant passages from uploaded files
- 🧠 BioGPT generates 1–2 paragraph clinical summary
- 🛡️ Everything stays local

---

##  Tech Stack

| Component            | Purpose                           |
|---------------------|-----------------------------------|
| `BioGPT-Large`       | Domain-specific generation        |
| `MiniLM-L6-v2`       | Embedding for semantic retrieval  |
| `Qdrant`             | Local vector DB                   |
| `FastAPI`            | Backend + file handling           |
| `Tailscale`          | Secure VPN                        |
| `watchdog`           | Real-time directory monitoring    |
| `PDFMiner`, `PyMuPDF`| PDF text extraction               |

---

##  Roadmap

- [x] RAG with BioGPT + Qdrant
- [x] Chunked PDF/TXT ingestion
- [x] Show/hide source context in UI
- [x] Watchdog-based live ingestion
- [ ] Add MPC container support
- [ ] Admin dashboard for clinical audit logs
- [ ] Fine-tuning option for local BioGPT retraining

---

##  Ideal Use Cases

- Local AI assistant in hospitals or clinics
- Medical record summarization in offline settings
- Secure RAG over proprietary research datasets
- Internal tools for regulatory or compliance audits

---

Feel free to contribute or fork this repo to build your own **private, offline clinical GPT agent**!

> Made with ❤️ for doctors, data scientists & privacy geeks.
