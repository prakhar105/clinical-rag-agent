from pathlib import Path
import re

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from app.qdrant_manager import QdrantManager


#  Clean up BioGPT output
def clean_biogpt_output(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)                # Remove <TAGS>
    text = re.sub(r'▃+', '', text)                     # Remove separators
    text = re.sub(r'\s+', ' ', text).strip()           # Collapse whitespace
    text = re.sub(r'doi:\s*[\d\s./a-zA-Z-]+', '', text)
    text = re.sub(r'\d{4};\s*\d+:\s*\d+-\d+', '', text)
    return text.strip()

def extract_answer_from_output(output: str) -> str:
    # Assume BioGPT may repeat prompt — slice at 'Answer:' if needed
    if "Answer:" in output:
        return output.split("Answer:")[-1].strip()
    return output.strip()


#  Prompt Template
prompt_template = PromptTemplate.from_template("""
You are a helpful clinical assistant using BioGPT to answer questions about diabetes, inflammation, and related treatments.

You MUST only use the provided context to answer. Do not hallucinate. If the answer is not in the context, respond with:
"The answer is not available in the provided context."

Format your answer in 1-2 short clinical paragraphs.

Context:
{context}

Question:
{question}

Answer:
""")


# #  Embeddings & Vector DB
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# qdrant = QdrantClient(path="vector_store/qdrant_local")
# vectorstore = QdrantVectorStore(
#     client=qdrant,
#     collection_name="medprivagent_chunks",
#     embeddings=embedding_model
# )

qdrant_manager = QdrantManager()
retriever = qdrant_manager.get_retriever()

#  Load BioGPT (with CPU-safe config)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")

offload_dir = Path("offload_biogpt")
offload_dir.mkdir(exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/BioGPT-Large",
    torch_dtype="auto",
    device_map="auto",
    offload_folder=str(offload_dir)
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=generator)

#  RAG + Retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,  # ✅ Enable to get `context`
    chain_type_kwargs={"prompt": prompt_template}
)


#  Inference API
def answer_question(query: str) -> dict:
    try:
        if query.strip().lower() in {"hello", "hi", "hey"}:
            return {
                "answer": "Hi! I’m your clinical assistant. Ask me about Type 2 diabetes, inflammation, or treatment options.",
                "context": ""
            }

        result = qa_chain({"query": query})

        answer_clean = extract_answer_from_output(clean_biogpt_output(result["result"]))
        print("Answer clean :", answer_clean)
        context_docs = result.get("source_documents", [])

        context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs) if context_docs else "No context found."
        print("Context Text :", context_text)

        return {
            "answer": answer_clean,
            "context": context_text
        }

    except Exception as e:
        return {
            "answer": f"❌ An internal error occurred: {str(e)}",
            "context": ""
        }
