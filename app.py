import chainlit as cl
import os
from passlib.context import CryptContext
from config import Config
from db import VectorDatabase
from agent import Agent
import datetime

class ChatHistory:
    def __init__(self, limit=10):
        self.history = []
        self.limit = limit
    
    def add_message(self, message):
        if len(self.history) >= self.limit:
            self.history.pop(0)
        self.history.append(message)
    
    
# Create a password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    """Hash a password for storing."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

# Pre-hash the admin password (do this once and store the hash)
# hashed_admin_password = hash_password("admin")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # In real applications, fetch the user and hashed password from your database
    if username == "admin":
        if verify_password(password, Config.PWD):
            return cl.User(
                identifier="admin", 
                metadata={"role": "admin", "provider": "credentials"}
            )
    return None


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=f"Hello {app_user.identifier}, please upload pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
            max_files=5,
        ).send()
    collection_name = f"{app_user.identifier}.collection"

    vector_db = VectorDatabase(qdrant_url=os.getenv('QDRANT_URL'), embed_model_id=Config.EMBED_MODEL_ID)
    agent = Agent(llm="gemma3:12b")
    # Init chat history
    cl.user_session.set("chat_history", ChatHistory(limit=10))
    cl.user_session.set("agent", agent)
    
    # Create personal data collection
    vector_db.create_collection(collection_name=collection_name)
    cl.user_session.set("vector_db", vector_db)
    cl.user_session.set("collection_name", collection_name)
    # Wait for the user to upload a file
    for file in files:
        # Process the uploaded file
        await cl.Message(content=f"Processing {file.name}...").send()
        vector_db.add_documents3(file_path=file.path, collection=collection_name)
        await cl.Message(content=f"Finished processing {file.name}...").send()

    
    # await cl.Message(f"Hello {app_user.identifier}, how can i help you today?").send()
    # elements = [
    #   cl.Pdf(name="pdf1", display="side", path="./data/2311-SMERF.04079v1.pdf", page=1) # page, side, inline
    # ]
    # # Reminder: The name of the pdf must be in the content of the message
    # await cl.Message(content="Look at this local pdf1!", elements=elements).send()

# @cl.on_chat_end
# def end():
#     # print("goodbye", cl.user_session.get("id"))
#     client = cl.user_session.get("vector_db").client
#     collection_name = cl.user_session.get("collection_name")
#     client.delete_collection(collection_name=collection_name)

@cl.on_message
async def on_message(message: cl.Message):
    msg_timestamp = datetime.datetime.now().isoformat()
    history = cl.user_session.get("chat_history")
    prompt = message.content
    agent = cl.user_session.get("agent")
    vector_db = cl.user_session.get("vector_db")
    collection_name = cl.user_session.get("collection_name")
    response = agent(db=vector_db, prompt=prompt, collection=collection_name, chat_history=history)
    history.add_message({
        "timestamp": msg_timestamp,
        "user": message.content,
        "assistant": response
    })    
    cl.user_session.set("chat_history", history)
    # Send the response back to the user
    await cl.Message(content=response).send()
    

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)