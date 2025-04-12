import os
from dotenv import load_dotenv
from typing import List
from langchain_docling.loader import ExportType
load_dotenv()  # Load environment variables from .env file

class Config:
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    LLM = "gemma3:12b"
    EXPORT_TYPE = ExportType.DOC_CHUNKS
    RAY_ADDRESS = "auto"
    #PWD = '$2b$12$C6NurfvKnwkQSyi//CGBIu4KFCtidItSb8wLLYoF2lfGaViEKwTbe'
    ACTION_LIST: List[str] =["context_retrieval: retrieve context from database contain scientific papers",
                            # "context_analysis: use retrieved context from context_retrieval to generate refined context for LLM agent's final response",
                            "show_pdf: show pdf content if the user asks for it or the query is to vague or complex",
                            "chat_response: respond to user's msg."]
    NOUGAT_URL="http://localhost:8503"
    QDRANT_URL="http://localhost:6333"
    
    # API Keys
    OPENAI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ALIBABA_API_KEY: str = ""

config = Config()