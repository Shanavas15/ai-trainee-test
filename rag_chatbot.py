import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resume_file_path = r"C:\Users\shana\Desktop\Ai_Trainee_Test\Shanavas N - Resume.pdf"

if os.path.exists(resume_file_path):
    reader = PdfReader(resume_file_path)
    extracted_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    resume_text = "\n".join(extracted_text)
else:
    resume_text = "No resume found. Please provide a resume file in the current directory."

chunk_size = 100
chunk_overlap = 20

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks

all_chunks = []
for doc in [resume_text]:
    all_chunks.extend(chunk_text(doc, chunk_size, chunk_overlap))

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="resume_collection")

chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
collection.add(
    ids=chunk_ids,
    documents=all_chunks
)

user_query = "What frameworks are you familiar with?"
TOP_K = 3

results = collection.query(
    query_texts=[user_query],
    n_results=TOP_K
)

retrieved_chunks = results["documents"][0] if results["documents"] else []
context_for_llm = " ".join(retrieved_chunks)

FALLBACK_PROMPT = "I'm sorry, but I don't have information about that in my knowledge base."

system_prompt = (
    "You are a hiring assistant reviewing a candidate's resume. Use the following context from the resume to answer the user's question.\n "
    "Rules:\n"
    "1. Answer using only the information provided in the context. \n"
    "2. Do not make up information. \n"
    f"3. If the specific information is not available in the context, respond with: \'{FALLBACK_PROMPT}\'"
)

user_prompt = f"""Context from Resume:
{context_for_llm}

Question: {user_query}
Answer:"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0,
)

print(f"Question: {user_query}")
print(f"Answer: {response.choices[0].message.content}")