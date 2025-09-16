import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import Qdrant  # ✅ NEW, non-deprecated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load data
with open("data/processed_data/rag_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
metadatas = [{"source": item["source"]} for item in data]

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Setup Qdrant client
collection_name = "medprivagent_chunks"
client = QdrantClient(path="vector_store/qdrant_local")

# Create collection if needed
existing = [col.name for col in client.get_collections().collections]
if collection_name not in existing:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# #  Correctly initialize Qdrant vector store
# qdrant = Qdrant(
#     client=client,
#     collection_name=collection_name,
#     embeddings=embeddings  #  Use 'embeddings=', NOT 'embedding_function='
# )

# #  Ingest the texts
# qdrant.add_texts(texts=texts, metadatas=metadatas)

# Create vector store using updated class
qdrant = QdrantVectorStore(
    collection_name=collection_name,
    client=client,
    embedding=embeddings,  #  not `embeddings`
)

# Add your data
qdrant.add_texts(texts=texts, metadatas=metadatas)

print(" Qdrant vector store created and populated.")
