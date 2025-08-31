import os
import torch
import warnings
import vertexai
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from vertexai.generative_models import GenerativeModel, GenerationConfig
# import gradio as gr
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN")
FILTER_MODEL_ID = os.getenv("FILTER_MODEL", "google/flan-t5-small")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gahld-469906")
LOCATION = os.getenv("GCP_REGION", "global")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
TOP_P = float(os.getenv("TOP_P", 0.95))

def load_filter_model(model_id=FILTER_MODEL_ID):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text_gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=text_gen)

filter_llm = load_filter_model()

vertexai.init(project=PROJECT_ID, location=LOCATION)
gemini_model = GenerativeModel(GEMINI_MODEL)
gen_config = GenerationConfig(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_output_tokens=8192,
)

def pii_filter(chunk: str):
    prompt = f"""
    You are a privacy filter. Review the following text and redact or anonymize any sensitive data
    (names, emails, phone numbers, IDs, addresses, or any personally identifiable information).
    Keep the meaning intact.

    If the chunk only contains a section title like "1. Bank Policies", return it unchanged.

    Text:
    {chunk}

    Return the safe-to-share version:
    """
    result = filter_llm(prompt)
    return result

def summarizer(texts):
    SUMMARY_PROMPT = """You are a helpful legal assistant to help summarize the anonymized document in this prompt. 
    Structure the output as follows:
    **SUMMARY**
    1) TLDR (2-3 lines)
    2) Key Policies
    3) Prohibited Actions
    4) Exceptions/Notes (if necessary)
    5) Clauses Mentioned (if necessary)
    """

    joined_text = "\n\n".join([t.page_content for t in texts])
    response = gemini_model.generate_content(
        SUMMARY_PROMPT + "\n\nDocument:\n" + joined_text,
        generation_config=gen_config,
    )
    return response.text

def summarize_document(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7000,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    safe_chunks = []
    for doc in texts:
        safe_text = pii_filter(doc.page_content)
        doc.page_content = safe_text
        safe_chunks.append(doc)

    summary = summarizer(safe_chunks)
    return summary

def process_file(file_obj):
    if file_obj is None:
        return "Please upload a document."
    return summarize_document(file_obj.name)

# demo = gr.Interface(
#     fn=process_file,
#     inputs=gr.File(label="Upload a document", file_types=[".txt", ".md", ".pdf"]),
#     outputs="text",
#     title="Legal Document Summarizer",
#     description="Upload a document. The app will anonymize sensitive info and summarize it."
# )

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello Cloud Run!"}

@app.post("/summarize/")
async def summarize_api(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    summary = summarize_document(file_path)
    os.remove(file_path)
    return {"summary": summary}

# if __name__ == "__main__":
#     import sys
#     if "api" in sys.argv:  # run as API
#         uvicorn.run(app, host="0.0.0.0", port=8000)
#     else:  # run as Gradio UI
#         demo.launch()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
