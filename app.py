import chainlit as cl
import os
from passlib.context import CryptContext
from config import Config
from db import VectorDatabase

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
    collection_name = f"{app_user.identifier}.collection"
    files = None
    vector_db = VectorDatabase(qdrant_url=os.getenv('QDRANT_URL'), embed_model_id=Config.EMBED_MODEL_ID)
    
    # Init chat history
    cl.user_session.set("chat_history", [])
    
    # Create personal data collection
    vector_db.create_collection(collection_name=collection_name)
    cl.user_session.set("vector_db", vector_db)
    cl.user_session.set("collection_name", collection_name)
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=f"Hello {app_user.identifier}, please upload pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
            max_files=5,
        ).send()
    
    
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

@cl.on_chat_end
def end():
    # print("goodbye", cl.user_session.get("id"))
    client = cl.user_session.get("vector_db").client
    collection_name = cl.user_session.get("collection_name")
    client.delete_collection(collection_name=collection_name)

@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("chat_history")
    history.append(message)
    cl.user_session.set("chat_history", history)
    a = cl.user_session.get("vector_db")
    await cl.Message(content=f"Your vector db is {a}").send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)