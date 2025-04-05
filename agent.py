from langchain_ollama import OllamaLLM
from promp_template import ActionListTemplate, ContextAnalyserTemplate, SimpleResponseTemplate, ContextResponseTemplate, ContextTemplate
from db import VectorDatabase
from config import Config
import os
import re



class Agent:
    def __init__(self, llm:str):
        self.llm = OllamaLLM(model=llm)
        self.llm_name = llm
        self.action_list_template = ActionListTemplate()
        self.context_analyser = ContextAnalyserTemplate()
        self.simple_response_template = SimpleResponseTemplate()
        self.context_response_template = ContextResponseTemplate()
        self.context_template = ContextTemplate()
        
    def parse_deepseek(self, output: str) -> bool:
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        return output
        
    def __call__(self, prompt: str, db:VectorDatabase, collection:str, context:str, chat_history:ChatHistory):
        """Call the LLM with the prompt and database context."""
        planning_prompt = self.action_list_template.create_prompt({
            "query": prompt,
            "action_list": Config.ACTION_LIST
        })
        response = self.llm.invoke(planning_prompt).strip()
        if "deepseek" in self.llm_name:
            response = self.parse_deepseek(response)
        context = []
        for name in response.split("/"):
            method = getattr(self, name, None)  # Get method from object
            if callable(method):
                if name == "context_retrieval":
                    context, pdfs = method(db, prompt, collection)
                elif name == "context_analysis":
                    context = self.context_analysis(prompt, context)
                elif name == "show_pdf":
                    pdfs = method(pdfs)
                elif name == "chat_response":
                    response = method(prompt, context, chat_history)
            else:
                print(f"Method {name} not found")
        
        return response
    
    def chat_response(self, prompt:str, context:list, chat_history:ChatHistory):
        if len(context) == 0:
            response_prompt = self.simple_response_template.create_prompt({
                "query": prompt,
                "history": chat_history.history
            })
            response = self.llm.invoke(response_prompt).strip()
            if "deepseek" in self.llm_name:
                response = self.parse_deepseek(response)
        else:
            response_prompt = self.context_response_template.create_prompt({
                "query": prompt,
                "context": context,
                "history": chat_history.history
            })
            response = self.llm.invoke(response_prompt).strip()
            if "deepseek" in self.llm_name:
                response = self.parse_deepseek(response)
        return response
    
    def context_analysis(self, prompt:str, context:list):
        analysis_prompt = self.context_analyser.create_prompt({
            "query": prompt,
            "context": context
        })
        response = self.llm.invoke(analysis_prompt).strip()
        if "deepseek" in self.llm_name:
            response = self.parse_deepseek(response)
        return response
    
    def show_pdf(self, pdfs:set):
        return pdfs
        
    
if __name__ == "__main__":
    # Initialize the agent with a specific LLM model
    agent = Agent(llm="gemma3:12b")
    
    # Initialize the vector database
    vector_db = VectorDatabase(
        qdrant_url=os.getenv('QDRANT_URL'),
        embed_model_id=Config.EMBED_MODEL_ID
    )

    # Example usage
    #prompt = "Hi"
    prompt = "explain Polyline Sequence Representation in SMERF?"
    collection = "admin.collection"
    chat_history = []
    
    # Call the agent with the database, prompt, and collection
    result = agent(db=vector_db, prompt=prompt, collection=collection, chat_history=chat_history)
    print(result)