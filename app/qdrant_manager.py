# app/qdrant_manager.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient


class QdrantManager:
    def __init__(
        self,
        collection_name: str = "medprivagent_chunks",
        vector_store_path: str = "vector_store/qdrant_local",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.qdrant_client = QdrantClient(path=vector_store_path)
        self.collection_name = collection_name
        self.vectorstore = self._init_vector_store()

    def _init_vector_store(self):
        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embedding_model
        )

    def get_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
