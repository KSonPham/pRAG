from langchain.schema.runnable import RunnableConfig
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser
from langchain_ollama import OllamaLLM
from typing import Dict, Tuple, Any, Callable, Optional
from promp_template import PromptTemplate, SimpleResponseTemplate, ContextResponseTemplate
from db import VectorDatabase
from config import Config
import os
import re   
import ray

class ChatHistory:
    def __init__(self, limit=10):
        self.history = []
        self.limit = limit
    
    def add_message(self, message):
        if len(self.history) >= self.limit:
            self.history.pop(0)
        self.history.append(message)

def create_runnable(llm: str, vector_db: VectorDatabase, collection_name: str, pdf_callback: Optional[Callable] = None):
    # Initialize LLM with streaming enabled
    ollama = OllamaLLM(model=llm, streaming=True)
    
    # Decision templates
    show_pdf_template = PromptTemplate({
        "Query": "{query}",
        "Instructions": "Determine if this query requires showing PDF documents. Return True if the query is complex and would benefit from visual context, or if user explicitly asks for PDFs. Return False otherwise.",
        "Output Format": "Output: True or False, Reason: <reason>"
    })

    need_context_template = PromptTemplate({
        "Query": "{query}", 
        "Instructions": "Determine if this query requires retrieving context from the database. Return True if the query asks about specific information that would need context. Return False for general questions or greetings.",
        "Output Format": "Output: True or False, Reason: <reason>"
    })

    # Decision functions
    def decide_show_pdf(query: str) -> bool:
        prompt = show_pdf_template.render(query)
        response = ollama.invoke(prompt).strip().lower()
        match = re.search(r"Output:\s*(True|False)", response, re.IGNORECASE)
        if match:
            response = match.group(1).lower()
        return response
        
    def decide_need_context(query: str) -> bool:
        prompt = need_context_template.render(query)
        response = ollama.invoke(prompt).strip().lower()
        match = re.search(r"Output:\s*(True|False)", response, re.IGNORECASE)
        if match:
            response = match.group(1).lower()   
        return response

    def get_context(query: str) -> Tuple[list, set]:
        context, pdfs = ray.get(vector_db.context_retrieval.remote(query, collection_name))
        return context, pdfs

    def generate_response(inputs: Dict) -> str:
        query = inputs["query"]
        need_context = inputs["need_context"]
        context = inputs.get("context", [])
        history = inputs.get("history", [])
        
        # If pdfs and callback exist, trigger the callback
        if pdf_callback and inputs.get("show_pdf") == "true" and inputs.get("pdfs"):
            # Store pdfs for external async handling via callback
            pdf_callback(inputs["pdfs"])

        if need_context:
            prompt = ContextResponseTemplate()
            prompt = prompt.render({
                "query": query,
                "context": context,
                "history": history,
                "pdf": inputs.get("pdfs", [])
            })
        else:
            prompt = SimpleResponseTemplate()
            prompt = prompt.render({
                "query": query,
                "history": history
            })

        return ollama.invoke(prompt)

    # Build the runnable chain
    chain = (
        RunnablePassthrough()
        .assign(
            show_pdf=RunnableLambda(decide_show_pdf),
            need_context=RunnableLambda(decide_need_context)
        )
        .assign(
            context=lambda x: get_context(x["query"])[0] if x["need_context"] else [],
            pdfs=lambda x: get_context(x["query"])[1] if x["need_context"] else set()
        )
        .assign(
            response=RunnableLambda(generate_response)
        )
        | RunnableLambda(lambda x: x["response"])
        | StrOutputParser()
    )

    return chain

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize vector database
    vector_db = VectorDatabase.remote(
        qdrant_url=Config.QDRANT_URL,
        embed_model_id=Config.EMBED_MODEL_ID
    )

    pdf_to_display = []
    def pdf_callback(pdfs):
        global pdf_to_display
        pdf_to_display = pdfs
    runnable = create_runnable(Config.LLM, vector_db, "admin.collection", pdf_callback)
    query = "show me pdf?"
    response = runnable.invoke({"query": query})
    print("Regular response:", response)
    
    # For streaming
    # for chunk in runnable.stream({"query": query}):
    #     print("Streaming chunk:", chunk, end="", flush=True)
    # print()
    
    # For async streaming (to be used with Chainlit or similar)
    # import asyncio
    
    # async def demo_async_streaming():
    #     async for chunk in runnable.astream({"query": query}):
    #         print("Async chunk:", chunk, end="", flush=True)
    
    # # Run the async demo if needed
    # asyncio.run(demo_async_streaming())