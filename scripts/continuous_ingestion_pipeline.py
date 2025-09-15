import os
import json
import time
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# --- Config ---
INPUT_DIR = Path("data/clinical_data")
OUTPUT_DIR = Path("data/processed_data")
QDRANT_PATH = "vector_store/qdrant_local"
COLLECTION_NAME = "medprivagent_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RAG_JSON = OUTPUT_DIR / "rag_chunks.json"
LORA_JSON = OUTPUT_DIR / "lora_dataset.json"
INGESTED_TRACKER = OUTPUT_DIR / "ingested_files.json"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- Setup Qdrant ---
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client = QdrantClient(path=QDRANT_PATH)

if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    print("🧠 Creating new Qdrant collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    collection_name=COLLECTION_NAME,
    client=client,
    embedding=embeddings,
)

# --- Utility Functions ---
def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"❌ Failed to extract {file_path.name}: {e}")
    return ""

def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)

def append_to_json(path: Path, new_data: list):
    if path.exists():
        try:
            with path.open("r+", encoding="utf-8") as f:
                data = json.load(f)
                data.extend(new_data)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            print(f"⚠️ Corrupt JSON in {path.name}, overwriting...")
            with path.open("w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

def hash_file(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()

def load_ingested_hashes():
    if INGESTED_TRACKER.exists():
        with INGESTED_TRACKER.open("r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_ingested_hashes(hashes):
    with INGESTED_TRACKER.open("w", encoding="utf-8") as f:
        json.dump(list(hashes), f, indent=2)

# --- Ingest Logic ---
def process_and_ingest(file_path: Path, seen_hashes: set):
    file_hash = hash_file(file_path)
    if file_hash in seen_hashes:
        print(f"⏩ Already processed {file_path.name}, skipping.")
        return

    print(f"📄 Processing: {file_path.name}")
    text = extract_text(file_path)

    if not text.strip():
        print("⚠️ Empty or unreadable file. Skipping.")
        return

    chunks = chunk_text(text)
    source = file_path.name

    rag_data = [{"text": chunk, "source": source} for chunk in chunks]
    lora_data = [{
        "instruction": "Summarize the following clinical text:",
        "input": chunk,
        "output": "Domain-specific summary to be generated here."
    } for chunk in chunks]

    append_to_json(RAG_JSON, rag_data)
    append_to_json(LORA_JSON, lora_data)

    texts = [d["text"] for d in rag_data]
    metadatas = [{"source": d["source"]} for d in rag_data]
    vector_store.add_texts(texts=texts, metadatas=metadatas)

    seen_hashes.add(file_hash)
    save_ingested_hashes(seen_hashes)

    print(f"✅ Ingested {len(chunks)} chunks from {source}")

# --- Watcher Handler ---
class ClinicalFileHandler(FileSystemEventHandler):
    def __init__(self, seen_hashes):
        self.seen_hashes = seen_hashes

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".pdf", ".txt")):
            file_path = Path(event.src_path)
            process_and_ingest(file_path, self.seen_hashes)

# --- Initial Bootstrapping ---
def bootstrap_existing_files(seen_hashes):
    print("🚀 Bootstrapping existing files...")
    for file in INPUT_DIR.glob("*.pdf"):
        process_and_ingest(file, seen_hashes)
    for file in INPUT_DIR.glob("*.txt"):
        process_and_ingest(file, seen_hashes)

# --- Main Loop ---
if __name__ == "__main__":
    seen_hashes = load_ingested_hashes()

    print(f"👀 Watching {INPUT_DIR.resolve()} for new files...")
    bootstrap_existing_files(seen_hashes)

    handler = ClinicalFileHandler(seen_hashes)
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
