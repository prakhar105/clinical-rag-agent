from pathlib import Path
import re

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient


# 🧹 Clean up BioGPT output
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


# 🧠 Prompt Template
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


# 🔎 Embeddings & Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant = QdrantClient(path="vector_store/qdrant_local")
vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name="medprivagent_chunks",
    embeddings=embedding_model
)


# 🧬 Load BioGPT (with CPU-safe config)
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
    #return_full_text=False,  # ⛔ Prevent echoing input
    eos_token_id=tokenizer.eos_token_id,  # ✅ Optional
)

llm = HuggingFacePipeline(pipeline=generator)

# 🧠 RAG + Retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)


# 🤖 Inference API
def answer_question(query: str) -> str:
    try:
        if query.strip().lower() in {"hello", "hi", "hey"}:
            return "Hi! I’m your clinical assistant. Ask me about Type 2 diabetes, inflammation, or treatment options."

        raw_answer = qa_chain.run(query)
        return extract_answer_from_output(clean_biogpt_output(raw_answer))

    except Exception as e:
        return f"An internal error occurred: {str(e)}"
